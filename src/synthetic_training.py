#!/usr/bin/env python
import pathlib
from dataclasses import dataclass
import sys
import json

import numpy as np
import polars as pl

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from sbi.utils import BoxUniform
from sbi.inference import NPE, simulate_for_sbi
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import (
    FCEmbedding,
    PermutationInvariantEmbedding
)
from sbi.analysis import pairplot, sbc_rank_plot
from sbi.diagnostics import run_sbc

import matplotlib.pyplot as plt

from scipy.stats import lognorm

from data_processing import DataProcessing, FeatureConfig, T100DataProcessing


@dataclass(slots=True)
class SyntheticGeneratorConfig:
    """
    Configuration for the synthetic data generation pipeline.

    Attributes
    ----------
    seed : int
        Seed for pseudo‑random generators.
    n_flights : int
        Number of flights to sample from T100.
    k_swaps : int
        Number of weight sweeps per flight.
    loguniform_low : float
        Lower bound for the log‑uniform weight sweep.
    """
    seed: int = 42
    n_flights: int = 1
    k_swaps: int = 5
    loguniform_low: float = 100.


class HyperPrior:
    def __init__(self, psi_min: float = 1e-1, psi_max: float = 1e1) -> None:
        self.psi_min = psi_min
        self.psi_max = psi_max

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Sample from a log-uniform distribution."""
        return np.exp(rng.uniform(np.log(self.psi_min), np.log(self.psi_max), size=size))

    def signed_sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Sample from a signed log-uniform distribution."""
        signs = rng.choice([-1, 1], size=size)
        return signs * self.sample(rng, size)
    

class SyntheticDataGenerator:
    """
    A self‑contained generator for synthetic cargo‑flights data.

    The constructor loads  T100 and stores the full feature tensor
    and per‑flight metadata. Public methods allow sampling a subset of flights,
    generating slack samples, performing the weight sweep, and persisting the
    resulting synthetic training set.
    """

    def __init__(self, config: SyntheticGeneratorConfig | None = None, shipping_data: DataProcessing | None = None) -> None:
        self.config = config or SyntheticGeneratorConfig()
        self.device = device
        # Random number stream – keep a single PRNG instance for deterministic runs.
        self.rng = np.random.default_rng(self.config.seed)
        self.proc = T100DataProcessing()

        self.shipping_proc = shipping_data or DataProcessing(separate_target=True)

        self._load_and_preprocess()
        self._epsilon_design_matr()
        self._load_shipping_data()
        self._weight_dist()

        self.hyperprior = HyperPrior()

    # -------------------------------------------------------------------
    #  Loading / preprocessing
    # -------------------------------------------------------------------

    def _augment_shipping_data(self, max_rows: int = 100, num_rows: int = 100) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if max_rows > len(self.ship_dist):
            max_rows = len(self.ship_dist)
            num_rows = len(self.ship_dist)

        if max_rows % 5 != 0:
            max_rows = (max_rows // 5) * 5  # round down to nearest multiple of 5
            num_rows = (num_rows // 5) * 5  # round down to nearest multiple of 5

        # Create a local copy of shipping data to prevent overwriting the class attribute
        aug_ship_dist = self.ship_dist[:max_rows].clone()
        aug_ship_dist = aug_ship_dist.with_columns(pl.col("AW (lbs)").cast(pl.Float64))

        aug_obs_ship_tens = self.observed_ship_tensor[:max_rows].clone()
        aug_obs_weight_tens = self.observed_weight_tensor[:max_rows].clone()

        for _ in range(num_rows):
            # randomly select a row from the shipping data
            idx = self.rng.choice(max_rows)
            df_row = aug_ship_dist[idx]

            tens_row = aug_obs_ship_tens[idx]
            tens_weight = aug_obs_weight_tens[idx]

            match = self.t100_df.filter((pl.col("ORIGIN") == df_row["ORIGIN"]) & (pl.col("DEST") == df_row["DEST"]))
            assert len(match) == 1, f"got {match} matches for {df_row['ORIGIN']} -> {df_row['DEST']}, expected 1"

            # create a new row with the same values but with a weight greater than or equal to the slack value
            new_row = df_row.clone()
            # new_row is a Series, so we access the value using .item() or [0]
            weight_val = float(new_row["AW (lbs)"][0])
            slack_val = match['SLACK'].item()

            assert weight_val <= slack_val, breakpoint()# f"weight_val={weight_val} is greater than slack_val={slack_val} for {df_row['ORIGIN']} -> {df_row['DEST']}"
            new_weight = slack_val - 1e3
            assert new_weight >= 0, f"new_weight={new_weight} is negative for slack_val={slack_val}"
            assert new_weight >= weight_val, f"new_weight={new_weight} is less than original weight_val={weight_val}"

            # new_row = new_row.with_columns(pl.lit(max(weight_val, slack_val - 3e3)).alias("AW (lbs)"))
            new_row = new_row.with_columns(pl.lit(new_weight).alias("AW (lbs)"))
            # append the new row to the local augmented dataframe
            aug_ship_dist = aug_ship_dist.vstack(new_row)

            # append the new tensor row to the local augmented tensor
            aug_obs_ship_tens = torch.cat([aug_obs_ship_tens, tens_row.unsqueeze(0)], dim=0)
            aug_obs_weight_tens = torch.cat([aug_obs_weight_tens, torch.tensor([[new_weight]], dtype=torch.float32)], dim=0)

        # Derive U and W from the local augmented dataframe, sampled to max_rows
        # U = torch.tensor(self._apply_one_hot_mapping(aug_ship_dist[:(max_rows + num_rows)]), dtype=torch.float32)
        # NOTE we add 2 manually to account for the conversion to OD XYZ from OD lat-lon
        U = aug_obs_ship_tens[:(max_rows + num_rows)]
        W = aug_obs_weight_tens[:(max_rows + num_rows)].to(torch.float32)
        
        # S = 1 for the original rows and 0 for the augmented rows
        s_labels = [1.0 if i < max_rows else 0.0 for i in range(max_rows + num_rows)]
        S = torch.tensor(s_labels, dtype=torch.float32).unsqueeze(1)
        return U, W, S

    def _load_shipping_data(self) -> None:
        """Load and preprocess the shipping data."""
        self.shipping_proc.align_from(self.proc)
        
        df_shipping = self.shipping_proc.load_shipping_data()
        ship_geo = self.shipping_proc._geolocate_nodes(df_shipping)
        ship_dist = self.shipping_proc._calculate_distance(ship_geo)

        self.ship_dist = self.shipping_proc.get_od_overlap(ship_dist, self.t100_df)

        self.observed_ship_tensor, self.observed_weight_tensor = self.shipping_proc.to_tensor(
            self.ship_dist, is_training=False, simple=True, just_num=True, normalize_and_convert=True
        )

        print(f"Observed shipping tensor shape: {self.observed_ship_tensor.shape}, {self.observed_weight_tensor.shape}")

    def _load_and_preprocess(self) -> None:
        """Load the T100 data, clean it, and compute the feature tensor."""
        df = self.proc.filter_data()
        df = df.filter(pl.col("SLACK").is_not_null()).filter(~pl.col("SLACK").is_nan())

        # Persist the per‑flight scalar columns – they are needed during synthesis.
        # self.u_max_all = df["PAYLOAD_PER_FLIGHT"].to_numpy().astype(np.float32)
        # self.u_cargo_all = df["FREIGHT_PER_FLIGHT"].to_numpy().astype(np.float32)
        # self.u_mail_all = df["MAIL_PER_FLIGHT"].to_numpy().astype(np.float32)
        # self.u_pax_all = df["PASSENGERS_PER_FLIGHT"].to_numpy().astype(np.float32)

        # Geolocate nodes, compute distances, and encode all features into a
        # single numerical tensor.
        df_geo = self.proc._geolocate_nodes(df)
        self.t100_df = self.proc._calculate_distance(df_geo)

        self.mu_z = self.t100_df["MU_SLACK"].to_numpy().astype(np.float32)
        self.sigma_z = self.t100_df["SIGMA_SLACK"].to_numpy().astype(np.float32)

        self.features_tensor, _ = self.proc.to_tensor(
            self.t100_df, is_training=True, simple=True, just_num=True, normalize_and_convert=True
        )
        self.X_full = self.features_tensor.detach().cpu().numpy().astype(np.float32)

        self.n_features = self.X_full.shape[1]

    # -------------------------------------------------------------------
    #  Sample a subset of flights
    # -------------------------------------------------------------------

    def _sample_flights(self, n_samples: int) -> None:
        total = self.X_full.shape[0]
        if n_samples > total:
            raise ValueError(
                f"Requested n_samples={n_samples} exceeds "
                f"available flights={total}"
            )
        idx = self.rng.choice(total, size=n_samples, replace=False)
        self.idx = idx
        self.X_sel = self.X_full[idx]

        # self.u_max_sel = self.u_max_all[idx]
        # self.u_cargo_sel = self.u_cargo_all[idx]
        # self.u_mail_sel = self.u_mail_all[idx]
        # self.u_pax_sel = self.u_pax_all[idx]

        self.design_sell = self.design_full[idx]

    # -------------------------------------------------------------------
    #  Generate slack samples
    # -------------------------------------------------------------------
    
    def _epsilon_design_matr(self) -> None:
        self.design_full = self.X_full
        self.n_design = self.design_full.shape[1]
        # categorical_cols = FeatureConfig().categorical_features
        # numeric_cols = FeatureConfig().numerical_features

        # dummies = self.df.select(categorical_cols).to_dummies()
        # self.dummy_cols = dummies.columns

        # numeric = self.df.select(numeric_cols).to_numpy().astype(np.float32)
        # self.num_mean = numeric.mean(axis=0, keepdims=True)
        # self.num_std = numeric.std(axis=0, keepdims=True) + 1e-6

        # num_std_scaled = (numeric - self.num_mean) / (self.num_std)

        # self.design_full = np.concatenate([dummies.to_numpy().astype(np.float32), num_std_scaled], axis=1)
        # self.n_design = self.design_full.shape[1]
        
    def _draw_hyperprior_coeffs(self) -> None:
        """Draw hyperprior coefficients for the gamma distribution."""
        self.psi_alpha = self.hyperprior.signed_sample(self.rng, self.n_design)
        self.psi_beta = self.hyperprior.signed_sample(self.rng, self.n_design)

    def _set_hyperprior_from_theta(self, theta: np.ndarray) -> None:
        n_design = self.design_full.shape[1]
        assert theta.shape[0] == 2 * n_design, f"Expected {2 * n_design} hyperprior coefficients, got {theta.shape[0]}"

        self.psi_alpha = theta[:n_design]
        self.psi_beta = theta[n_design:]

    def _sample_theta(self):
        theta = np.concatenate(
            [self.hyperprior.signed_sample(self.rng, self.n_design),
             self.hyperprior.signed_sample(self.rng, self.n_design)]
        )
        return torch.tensor(theta, dtype=torch.float32)

    def _alpha_beta(self, design_matr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        log_alpha = (design_matr @ self.psi_alpha) / np.sqrt(self.n_design)
        log_beta  = (design_matr @ self.psi_beta)  / np.sqrt(self.n_design)
        return np.exp(log_alpha), np.exp(log_beta)

    def _draw_down(self, design_matr: np.ndarray, mu_z: np.ndarray) -> np.ndarray:
        scaling_factors = mu_z

        alpha_i, beta_i = self._alpha_beta(design_matr)
        return (self.rng.gamma(alpha_i, beta_i) * scaling_factors).astype(np.float32)

    def _standardize_features(self) -> None:
        self.X_mean = self.X_full.mean(axis=0, keepdims=True)
        self.X_std  = self.X_full.std(axis=0, keepdims=True) + 1e-6
        self.X_full_std = (self.X_full - self.X_mean) / self.X_std

    def _generate_slack_samples(self) -> None:
        """Create truncated‑normal slack (U) samples for the selected flights."""
        mu_z = self.mu_z[self.idx]
        sigma_z = self.sigma_z[self.idx]

        self.z_gen_sel = self.proc._truncated_slack_samples(
            mu_z, sigma_z, n_draws=1
        ).reshape(-1).astype(np.float32)  # Subtract the drawn down values

        # NOTE need to verify standardization
        # self._standardize_features()
        # self._epsilon_design_matr()
        # self.design_sell = self.design_full[self.idx]
        # self._draw_hyperprior_coeffs()
        eps = self._draw_down(self.design_sell, mu_z)
        # ensure non-negative slack values
        self.z_gen_sel = np.maximum(self.z_gen_sel - eps, 0.0).astype(np.float32)

    # -------------------------------------------------------------------
    #  Weight sweep
    # -------------------------------------------------------------------

    def _weight_dist(self) -> None:
        x = self.ship_dist["AW (lbs)"].cast(pl.Float64).drop_nulls().to_numpy()
        x = x[x > 0]  # Remove zeros for log-normal fitting
        self.max_weight = x.max()

        self.weight_dist_shape, self.weight_dist_loc, self.weight_dist_scale = lognorm.fit(x, floc=0)

    def _sample_weight(self, n_weight_samples: int) -> np.ndarray:
        samples = lognorm.rvs(
            self.weight_dist_shape,
            loc=self.weight_dist_loc,
            scale=self.weight_dist_scale,
            size=n_weight_samples,
            random_state=self.rng,
        )

        # self.weight_min = samples.min()
        # self.weight_max = samples.max()
        
        return samples

    def _weight_sweep(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        U, W, S = [], [], []
        w_s = np.array([1e2, 1e3, 3e3, 1e4, 3e4])

        for u_vec, z_gen in zip(self.design_sell, self.z_gen_sel):
            # w_s = self._sample_weight(self.config.k_swaps)
            # Standardize the analyzed weights
            s = (w_s <= z_gen).astype(np.float32)

            U.append(np.repeat(u_vec[np.newaxis, :], self.config.k_swaps, axis=0))
            W.append(w_s[:, np.newaxis])
            S.append(s[:, np.newaxis])

        # self.U = torch.tensor(np.concatenate(U, axis=0), dtype=torch.float32)
        # self.W = torch.tensor(np.concatenate(W, axis=0), dtype=torch.float32)
        # self.S = torch.tensor(np.concatenate(S, axis=0), dtype=torch.float32)
        U = torch.tensor(np.concatenate(U, axis=0), dtype=torch.float32)
        W = torch.tensor(np.concatenate(W, axis=0), dtype=torch.float32)
        S = torch.tensor(np.concatenate(S, axis=0), dtype=torch.float32)
        return U, W, S

    # -------------------------------------------------------------------
    #  Public API
    # -------------------------------------------------------------------
    
    def simulator(self, theta: np.ndarray):
        if isinstance(theta, torch.Tensor):
            theta = theta.detach().cpu().numpy()
        elif not isinstance(theta, np.ndarray):
            theta = np.asarray(theta, dtype=np.float32)

        if theta.ndim == 1:
            # 1. Redraw a new set of 100 flights for this simulation
            self._sample_flights(n_samples=34)

            # 1.5 Update the design matrix for the selected flights
            # self.design_sell = self.design_full[self.idx]
            
            # 2. Set the global hyperprior coefficients (same across all 100 flights)
            self._set_hyperprior_from_theta(theta)
            
            # 3. Generate slack samples for the 100 flights
            self._generate_slack_samples()
            
            # 4. Perform weight sweep for the 100 flights
            U, W, S = self._weight_sweep()
            
            return torch.cat([U, W, S], dim=1)

        outputs = [self.simulator(theta_i) for theta_i in theta]
        return torch.stack(outputs, dim=0)

    def _apply_one_hot_mapping(self, df: pl.DataFrame) -> np.ndarray:
        categorical_cols = FeatureConfig().categorical_features
        numerical_cols = FeatureConfig().numerical_features

        dummies = df.select(categorical_cols).to_dummies()
        dummy_dat = dummies.to_numpy().astype(np.float32)

        final_dummies = np.zeros((len(df), len(self.dummy_cols)), dtype=np.float32)
        # TODO: there are more cols in dummy_cols than in the current df, so we need to map them correctly, 
        # they still need to be added even if zero
        for col in dummies.columns:
            if col in self.dummy_cols:
                idx = self.dummy_cols.index(col)
                final_dummies[:, idx] = dummy_dat[:, dummies.columns.index(col)]

        numeric = df.select(numerical_cols).to_numpy().astype(np.float32)
        num_std_scaled = (numeric - self.num_mean) / (self.num_std)

        return np.concatenate([final_dummies, num_std_scaled], axis=1)

    def observation(self) -> torch.Tensor:
        """Build a fixed observation tensor for posterior conditioning."""
        max_rows = 100 # 500 reduced to 100 due to smaller true overlap size

        # select first 500 rows of the shipping data for the observation
        # U = self.observed_ship_tensor[:max_rows]
        # U = torch.tensor(self._apply_one_hot_mapping(self.ship_dist[:max_rows]), dtype=torch.float32)
        # W = self.observed_weight_tensor[:max_rows].to(torch.float32)
        # S = torch.ones_like(W)  # all weights are valid for the observation

        # print(f"U dtype: {U.dtype}, W dtype: {W.dtype}, S dtype: {S.dtype}")
        U, W, S = self._augment_shipping_data(max_rows=max_rows, num_rows=100)

        return torch.cat([U, W, S], dim=1).to(self.device)

    def generate(self) -> dict:
        """
        Run the full pipeline and return a dictionary suitable for torch.save.
        """
        self._sample_flights(n_samples=100)
        self._generate_slack_samples()

        self._load_shipping_data()
        self._weight_sweep()
        return {"features": self.U, "weights": self.W, "labels": self.S}

    def save(self, out_dir: pathlib.Path | str | None = None) -> pathlib.Path:
        """
        Persist the synthetic training set to disk.

        Parameters
        ----------
        out_dir : pathlib.Path | str | None
            Directory to write the ``synthetic_set.pt`` file.  If omitted,
            ``data/synthetic`` is used relative to the project root.
        """
        out_dir = pathlib.Path(out_dir) if out_dir else (
            pathlib.Path(__file__).resolve().parent.parent / "data" / "synthetic"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "synthetic_set.pt"
        torch.save(self.generate(), out_path)
        print(f"Synthetic training set saved to {out_path} – {self.U.shape[0]} samples")
        return out_path

    def print_sample(self, n: int = 5) -> None:
        """Print a short human‑readable sample of the synthetic data."""
        print("Sample of generated data:")
        print(
            "Features: [ORIGIN, DEST, UNIQUE_CARRIER, Origin Latitude, Origin Longitude, Destination Latitude, Destination Longitude, DISTANCE] | Weight | Label"
        )
        for i in range(min(n, self.U.shape[0])):
            print(
                f"Features: {self.U[i].numpy()}, Weight: {self.W[i].item():.4f}, Label: {self.S[i].item()}"
            )

        # print range of weights, number of positive labels, and number of negative labels
        print(f"Weight range: {self.W.min().item():.4f}                         {self.W.max().item():.4f}")
        print(f"Number of positive labels: {int(self.S.sum().item())}, Number of negative labels: {int((1 - self.S).sum().item())}")


class EmbeddingNet(nn.Module):
    def __init__(self, categorical_sizes: list[int],
                 embedding_dim: int = 10,
                 hidden_features: int = 50,
                 num_continuous_features: int = 5,
                 output_dim: int = 20,
                ):
        super().__init__()

        self.categories = len(categorical_sizes)
        self.num_continuous_features = num_continuous_features
        self.output_dim = output_dim
        print(f"Number of categorical features: {self.categories}")
        print(f"Sizes of categorical features: {categorical_sizes}")
        print(f"Number of continuous features: {self.num_continuous_features}")

        self.embeddings = nn.ModuleList(
            nn.Embedding(num_embeddings=size, embedding_dim=embedding_dim) for size in categorical_sizes
        )

        total_dim = self.categories * embedding_dim + num_continuous_features + 2

        self.fc = FCEmbedding(input_dim = total_dim, num_hiddens=hidden_features, output_dim=output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0)

        categorical_inputs = x[..., :self.categories].long()
        continuous_inputs = x[..., self.categories:]

        embedded_cats = []
        for i, emb in enumerate(self.embeddings):
            cat_col = categorical_inputs[..., i].clamp(min=0, max=emb.num_embeddings - 1)
            embedded_cats.append(emb(cat_col))

        combined = torch.cat(embedded_cats + [continuous_inputs], dim=-1)
        return self.fc(combined)
    

def _demo() -> None:
    cfg = SyntheticGeneratorConfig()
    generator = SyntheticDataGenerator(cfg)
    U, W, S = generator.generate()
    generator.save()
    generator.print_sample()


import optuna
def objective(trial: optuna.Trial):
    pass


if __name__ == "__main__":
    # _demo()
    generator = SyntheticDataGenerator()

    n_cats = len(generator.proc.cat_cols)
    print("cat_cols:", generator.proc.cat_cols)
    print("cat_sizes:", generator.proc.cat_sizes)
    print("actual max codes:", generator.X_full[:, :n_cats].max(axis=0))
    print("actual min codes:", generator.X_full[:, :n_cats].min(axis=0))

    # Uniform prior over the range [-10, 10] for each of the 2 * n_design hyperprior coefficients
    # prior = sbi_utils.BoxUniform(
    #     low=-2 * torch.ones(2 * generator.n_design),
    #     high=2 * torch.ones(2 * generator.n_design),
    # )
    std = 1.0
    prior = MultivariateNormal(
        loc=torch.zeros(2 * generator.n_design, device=device),
        covariance_matrix=torch.eye(2 * generator.n_design, device=device) * std,
    )

    simulator = generator.simulator
    '''
    generator._sample_flights(n_samples=100)  # populates generator.design_sell

    mu_z = (generator.u_max_sel - generator.u_cargo_sel - generator.u_mail_sel
        - generator.u_pax_sel * generator.proc.mu_pax)
    print("z_T100 (nominal slack) range:", mu_z.min(), mu_z.max(), "median:", np.median(mu_z))

    for theta_i in [prior.sample((1,))[0].numpy() for _ in range(3)]:
        generator._set_hyperprior_from_theta(theta_i)
        a, b = generator._alpha_beta(generator.design_sell)
        eps_sample = generator.rng.gamma(a, b)
        print("eps range:", eps_sample.min(), eps_sample.max(), "median:", np.median(eps_sample))

    # sys.exit(0)

    design_matr = generator.design_sell
    for _ in range(5):
        theta_i = prior.sample((1,))[0].numpy()
        generator._set_hyperprior_from_theta(theta_i)
        a, b = generator._alpha_beta(design_matr)
        print("alpha range:", a.min(), a.max(), "| beta range:", b.min(), b.max())

    for _ in range(5):
        theta_i = prior.sample((1,))[0]
        x_i = simulator(theta_i)  # simulator() itself calls _sample_flights(100) internally
        S_col = x_i[:, -1]
        print("S=1 fraction:", S_col.mean().item())

    sys.exit(0)
    '''

    # single_trial_embedding = EmbeddingNet(
    #     categorical_sizes=generator.proc.cat_sizes,
    #     embedding_dim=10,
    #     hidden_features=generator.n_design * 2,
    #     num_continuous_features=len(generator.proc.num_cols),
    # )

    single_trial_embedding = FCEmbedding(
        generator.n_design + 2,# 668,
        generator.n_design * 2,
    )

    embedding_net = PermutationInvariantEmbedding(
        single_trial_embedding,
        generator.n_design * 2,
    )

    neural_posterior = posterior_nn(
        model="maf",
        embedding_net=embedding_net,
        hidden_features=min(generator.n_design, 256),  # was n_design*2
        num_transforms=5,  # was 5
        z_score_x='independent',
        z_score_theta='independent',
    )

    inference = NPE(prior, density_estimator=neural_posterior, device=device)

    x_o = generator.observation()

    print("x_o dtype:", x_o.dtype)
    print("x_o shape:", x_o.shape)


    # Cheap pre-flight checks — abort early if something's structurally wrong
    assert x_o.dtype == torch.float32, f"x_o dtype wrong: {x_o.dtype}"
    assert not torch.isnan(x_o).any(), "x_o contains NaNs"
    theta_probe, x_probe = simulate_for_sbi(simulator, prior, num_simulations=20, simulation_batch_size=1, seed=0)
    assert not torch.isnan(x_probe).any(), "simulator produced NaNs"
    print("S=1 fraction (probe):", x_probe[:, :, -1].mean().item())
    print("theta probe std:", theta_probe.std(dim=0)[:5])

    # generator._sample_flights(n_samples=40)  # match your actual n_flights per sim
    # mu_z = (generator.u_max_sel - generator.u_cargo_sel - generator.u_mail_sel
    #         - generator.u_pax_sel * generator.proc.mu_pax)
    # print("z_T100 median:", np.median(generator.mu_z))

    eps_medians = []
    for _ in range(10):
        theta_i = prior.sample((1,))[0].cpu().numpy()
        generator._set_hyperprior_from_theta(theta_i)
        a, b = generator._alpha_beta(generator.design_sell)
        eps_sample = generator.rng.gamma(a, b) * 1e4  # match your scaling_factor
        eps_medians.append(np.median(eps_sample))

    print("eps median range across theta draws:", min(eps_medians), max(eps_medians))
    # print("z_T100 median for comparison:", np.median(mu_z))
    # sanity: eps medians should span a meaningful fraction of z_T100's scale,
    # not be uniformly tiny or uniformly huge relative to it

    num_rounds = 1
    posteriors = []
    proposal = prior  # Use the prior as the initial proposal distribution

    for _ in range(num_rounds):
        theta, x = simulate_for_sbi(
            simulator,
            prior,
            num_simulations=100_000,
            simulation_batch_size=1,
            seed=generator.config.seed,
        )

        # theta = theta.to(device)
        # x = x.to(device)

        density_estimator = inference.append_simulations(theta, x, proposal=proposal, data_device='cpu').train(show_train_summary=True, training_batch_size=512) # set max epochs here

        posterior = inference.build_posterior(density_estimator)

        posteriors.append(posterior)

        proposal = posterior.set_default_x(x_o)

        print("Fraction S=1 in synthetic training x:", x[:, :, -1].mean().item())

        # ============================================================
        # DIAGNOSTICS BLOCK
        # ============================================================
        run_dir = pathlib.Path("run_outputs") / f"run_{generator.config.seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # persist artifacts
        torch.save(density_estimator.state_dict(), run_dir / "density_estimator.pt")
        torch.save({"theta": theta.cpu(), "x": x.cpu()}, run_dir / "training_data.pt")
        with open(run_dir / "training_summary.json", "w") as f:
            json.dump({k: v for k, v in inference.summary.items()}, f, default=str)

        diagnostics = {}
        diagnostics["S_fraction_train"] = x[:, :, -1].mean().item()

        theta_test = prior.sample((1,))
        x_test = simulator(theta_test[0]).to(device)
        post_test = posterior.sample((2000,), x=x_test).detach().cpu()
        prior_samp = prior.sample((2000,)).cpu()
        diagnostics["contraction_ratio"] = (post_test.std(dim=0) / prior_samp.std(dim=0))[:10].tolist()

        samples_xo = posterior.sample((2000,), x=x_o).detach().cpu()
        diagnostics["posterior_std_xo"] = samples_xo.std(dim=0)[:10].tolist()
        diagnostics["posterior_mean_xo"] = samples_xo.mean(dim=0)[:10].tolist()

        diagnostics["z_gen_stats"] = {
            "median": float(np.median(generator.z_gen_sel)),
            "frac_zero": float((generator.z_gen_sel == 0).mean()),
        }

        with open(run_dir / "diagnostics.json", "w") as f:
            json.dump(diagnostics, f, indent=2)

        theta_sbc, x_sbc = simulate_for_sbi(simulator, prior, num_simulations=300,
                                              simulation_batch_size=1, seed=123)
        ranks, _ = run_sbc(theta_sbc, x_sbc.to(device), posterior, num_posterior_samples=1000)
        fig, ax = sbc_rank_plot(ranks, num_posterior_samples=1000, num_bins=20)
        fig.savefig(run_dir / "sbc_rank_plot.png", dpi=150)
        plt.close(fig)

    print("Posterior ready for conditioning on x_o with shape:", tuple(x_o.shape))

    print(inference.summary["training_loss"])
    print(inference.summary["validation_loss"])

    fig, ax = plt.subplots()
    ax.plot(inference.summary["training_loss"], label="train")
    ax.plot(inference.summary["validation_loss"], label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()
    fig.savefig("loss_curve.png", dpi=150)

    print("Saved to loss_curve.png")

    # posterior_samples = posterior.sample((1,), x=x_o)

    samples = posterior.sample((2_000,), x=x_o).detach().cpu()
    print("posterior std (34 overlapping rows only):", samples.std(dim=0)[:5])
    print("posterior mean (34 overlapping rows only):", samples.mean(dim=0)[:5])

    subset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    n_bound = 4 * std

    fig, axes = pairplot(
        samples[:, subset],
        limits=[[-n_bound, n_bound]] * len(subset),
        labels=[f"psi_{i}" for i in subset],
        figsize=(6, 6),
    )

    fig.savefig("pairplot.png", dpi=150)
    print("Saved to pairplot.png")

    theta_test = prior.sample((1,))
    x_test = simulator(theta_test[0])  # in-distribution, has real simulator's S mix
    post_test = posterior.sample((2_000,), x=x_test.to(device)).detach().cpu()
    print("posterior std (in-dist x):", post_test.std(dim=0)[:5])
    print("prior std:", prior.sample((2_000,)).std(dim=0)[:5])
