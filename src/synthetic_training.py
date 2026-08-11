#!/usr/bin/env python
import pathlib
from dataclasses import dataclass

import numpy as np
import polars as pl

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

torch.manual_seed(42)
np.random.seed(42)

from sbi.utils import BoxUniform
from sbi.inference import NPE, simulate_for_sbi
from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import (
    FCEmbedding,
    PermutationInvariantEmbedding
)

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
    def __init__(self, psi_min: float = 1e-3, psi_max: float = 1e3) -> None:
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

    def _load_shipping_data(self) -> None:
        """Load and preprocess the shipping data."""
        self.shipping_proc.align_from(self.proc)
        
        df_shipping = self.shipping_proc.load_shipping_data()
        ship_geo = self.shipping_proc._geolocate_nodes(df_shipping)
        self.ship_dist = self.shipping_proc._calculate_distance(ship_geo)

        self.observed_ship_tensor, self.observed_weight_tensor = self.shipping_proc.to_tensor(
            self.ship_dist, is_training=False, simple=True
        )

    def _load_and_preprocess(self) -> None:
        """Load the T100 data, clean it, and compute the feature tensor."""
        df = self.proc.filter_data()
        df = df.filter(pl.col("SLACK").is_not_null()).filter(~pl.col("SLACK").is_nan())

        # Persist the per‑flight scalar columns – they are needed during synthesis.
        self.u_max_all = df["PAYLOAD_PER_FLIGHT"].to_numpy().astype(np.float32)
        self.u_cargo_all = df["FREIGHT_PER_FLIGHT"].to_numpy().astype(np.float32)
        self.u_mail_all = df["MAIL_PER_FLIGHT"].to_numpy().astype(np.float32)
        self.u_pax_all = df["PASSENGERS_PER_FLIGHT"].to_numpy().astype(np.float32)

        # Geolocate nodes, compute distances, and encode all features into a
        # single numerical tensor.
        df_geo = self.proc._geolocate_nodes(df)
        df_dist = self.proc._calculate_distance(df_geo)

        self.features_tensor, _ = self.proc.to_tensor(
            df_dist, is_training=True, simple=True
        )
        self.X_full = self.features_tensor.detach().cpu().numpy().astype(np.float32)
        self.df = df_dist

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
        self.u_max_sel = self.u_max_all[idx]
        self.u_cargo_sel = self.u_cargo_all[idx]
        self.u_mail_sel = self.u_mail_all[idx]
        self.u_pax_sel = self.u_pax_all[idx]

        self.design_sell = self.design_full[idx]

    # -------------------------------------------------------------------
    #  Generate slack samples
    # -------------------------------------------------------------------
    
    def _epsilon_design_matr(self) -> None:
        categorical_cols = FeatureConfig().categorical_features
        numeric_cols = FeatureConfig().numerical_features

        dummies = self.df.select(categorical_cols).to_dummies()
        self.dummy_cols = dummies.columns

        numeric = self.df.select(numeric_cols).to_numpy().astype(np.float32)
        self.num_mean = numeric.mean(axis=0, keepdims=True)
        self.num_std = numeric.std(axis=0, keepdims=True) + 1e-6

        num_std_scaled = (numeric - self.num_mean) / (self.num_std)

        self.design_full = np.concatenate([dummies.to_numpy().astype(np.float32), num_std_scaled], axis=1)
        self.n_design = self.design_full.shape[1]
        
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

    def _draw_down(self, design_matr: np.ndarray) -> np.ndarray:
        alpha_i, beta_i = self._alpha_beta(design_matr)
        return self.rng.gamma(alpha_i, beta_i).astype(np.float32)

    def _standardize_features(self) -> None:
        self.X_mean = self.X_full.mean(axis=0, keepdims=True)
        self.X_std  = self.X_full.std(axis=0, keepdims=True) + 1e-6
        self.X_full_std = (self.X_full - self.X_mean) / self.X_std

    def _generate_slack_samples(self) -> None:
        """Create truncated‑normal slack (U) samples for the selected flights."""
        mu_z = (
            self.u_max_sel
            - self.u_cargo_sel
            - self.u_mail_sel
            - (self.u_pax_sel * self.proc.mu_pax)
        )
        sigma_z = np.sqrt(np.maximum(self.u_pax_sel, 0.0)) * self.proc.sigma_pax
        self.z_gen_sel = self.proc._truncated_slack_samples(
            mu_z, sigma_z, n_draws=1
        ).reshape(-1).astype(np.float32)  # Subtract the drawn down values

        # NOTE need to verify standardization
        # self._standardize_features()
        # self._epsilon_design_matr()
        # self.design_sell = self.design_full[self.idx]
        # self._draw_hyperprior_coeffs()
        eps = self._draw_down(self.design_sell)
        # ensure non-negative slack values
        self.z_gen_sel = np.maximum(self.z_gen_sel - eps, 0.0).astype(np.float32)
        # breakpoint()  # enable interactive debugging when needed

    # -------------------------------------------------------------------
    #  Weight sweep
    # -------------------------------------------------------------------

    def _weight_dist(self) -> None:
        x = self.ship_dist["AW (kg)"].cast(pl.Float64).drop_nulls().to_numpy()
        x = x[x > 0]  # Remove zeros for log-normal fitting

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
        for u_vec, z_gen in zip(self.design_sell, self.z_gen_sel):
            w_s = self._sample_weight(self.config.k_swaps)

            s = (w_s <= z_gen).astype(np.float32)
            # breakpoint()  # enable interactive debugging when needed
            U.append(np.repeat(u_vec[np.newaxis, :], self.config.k_swaps, axis=0))
            W.append(w_s[:, np.newaxis])
            S.append(s[:, np.newaxis])

        # self.U = torch.tensor(np.concatenate(U, axis=0), dtype=torch.float32)
        # self.W = torch.tensor(np.concatenate(W, axis=0), dtype=torch.float32)
        # self.S = torch.tensor(np.concatenate(S, axis=0), dtype=torch.float32)
        U = torch.tensor(np.concatenate(U, axis=0), dtype=torch.float32)
        W = torch.tensor(np.concatenate(W, axis=0), dtype=torch.float32)
        S = torch.tensor(np.concatenate(S, axis=0), dtype=torch.float32)
        # breakpoint()  # enable interactive debugging when needed
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
            self._sample_flights(n_samples=100)

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
        # breakpoint()  # enable interactive debugging when needed
        # TODO: there are more cols in dummy_cols than in the current df, so we need to map them correctly, 
        # they still need to be added even if zero
        for col in dummies.columns:
            if col in self.dummy_cols:
                idx = self.dummy_cols.index(col)
                final_dummies[:, idx] = dummy_dat[:, dummies.columns.index(col)]

        # breakpoint()  # enable interactive debugging when needed
        numeric = df.select(numerical_cols).to_numpy().astype(np.float32)
        num_std_scaled = (numeric - self.num_mean) / (self.num_std)

        return np.concatenate([final_dummies, num_std_scaled], axis=1)

    def observation(self) -> torch.Tensor:
        """Build a fixed observation tensor for posterior conditioning."""
        max_rows = 500

        # select first 500 rows of the shipping data for the observation
        # U = self.observed_ship_tensor[:max_rows]
        U = torch.tensor(self._apply_one_hot_mapping(self.ship_dist[:max_rows]), dtype=torch.float32)
        W = self.observed_weight_tensor[:max_rows].to(torch.float32)
        S = torch.ones_like(W)  # all weights are valid for the observation

        print(f"U dtype: {U.dtype}, W dtype: {W.dtype}, S dtype: {S.dtype}")

        return torch.cat([U, W, S], dim=1)

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
    # breakpoint()  # enable interactive debugging when needed
    generator.save()
    generator.print_sample()
    # breakpoint() # enable interactive debugging when needed


if __name__ == "__main__":
    # _demo()
    generator = SyntheticDataGenerator()

    n_cats = len(generator.proc.cat_cols)
    print("cat_cols:", generator.proc.cat_cols)
    print("cat_sizes:", generator.proc.cat_sizes)
    print("actual max codes:", generator.X_full[:, :n_cats].max(axis=0))
    print("actual min codes:", generator.X_full[:, :n_cats].min(axis=0))

    # breakpoint()  # enable interactive debugging when needed

    # Uniform prior over the range [-10, 10] for each of the 2 * n_design hyperprior coefficients
    # prior = sbi_utils.BoxUniform(
    #     low=-2 * torch.ones(2 * generator.n_design),
    #     high=2 * torch.ones(2 * generator.n_design),
    # )

    prior = MultivariateNormal(
        loc=torch.zeros(2 * generator.n_design),
        covariance_matrix=torch.eye(2 * generator.n_design) * 9.0,
    )

    simulator = generator.simulator

    # single_trial_embedding = EmbeddingNet(
    #     categorical_sizes=generator.proc.cat_sizes,
    #     embedding_dim=10,
    #     hidden_features=generator.n_design * 2,
    #     num_continuous_features=len(generator.proc.num_cols),
    # )

    single_trial_embedding = FCEmbedding(
        668,
        generator.n_design, # * 2,
    )

    embedding_net = PermutationInvariantEmbedding(
        single_trial_embedding,
        generator.n_design, # * 2,
    )

    neural_posterior = posterior_nn(
        model="maf",
        embedding_net=embedding_net,
        hidden_features=min(generator.n_design, 64),  # was n_design*2
        num_transforms=3,  # was 5
        z_score_x='none',
        z_score_theta='none'
    )

    # breakpoint()  # enable interactive debugging when needed
    inference = NPE(prior, density_estimator=neural_posterior, device="cpu")

    x_o = generator.observation()

    print("x_o dtype:", x_o.dtype)
    print("x_o shape:", x_o.shape)

    # breakpoint()  # enable interactive debugging when needed

    theta, x = simulate_for_sbi(
        simulator,
        prior,
        num_simulations=10000,
        simulation_batch_size=1,
        seed=generator.config.seed,
    )

    density_estimator = inference.append_simulations(theta, x).train(show_train_summary=True) # set max epochs here

    posterior = inference.build_posterior(density_estimator)
    posterior.set_default_x(x_o)
    print("Posterior ready for conditioning on x_o with shape:", tuple(x_o.shape))

    # posterior_samples = posterior.sample((1,), x=x_o)
    # breakpoint()  # enable interactive debugging when needed
    from sbi.analysis import pairplot
    import matplotlib.pyplot as plt

    samples = posterior.sample((2000,), x=x_o)

    subset = [0, 1, 2]
    std = 3.0  # match your MVN prior's std
    n_bound = 4 * std

    fig, axes = pairplot(
        samples[:, subset],
        limits=[[-n_bound, n_bound]] * len(subset),
        labels=[f"psi_{i}" for i in subset],
        figsize=(6, 6),
    )

    fig.savefig("pairplot.png", dpi=150)
    print("Saved to pairplot.png")
