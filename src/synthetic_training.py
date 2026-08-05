#!/usr/bin/env python
import pathlib
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
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
    n_flights: int = 500
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

        self.shipping_proc = shipping_data or DataProcessing()


        self._load_and_preprocess()

        self.hyperprior = HyperPrior()

    # -------------------------------------------------------------------
    #  Loading / preprocessing
    # -------------------------------------------------------------------

    def _load_shipping_data(self) -> None:
        """Load and preprocess the shipping data."""
        self.shipping_proc.align_from(self.proc)
        df_shipping = self.shipping_proc.load_shipping_data()
        ship_geo = self.shipping_proc._geolocate_nodes(df_shipping)
        self.ship_dist =self.shipping_proc._calculate_distance(ship_geo)

        self.ship_tensor, _ = self.shipping_proc.to_tensor(
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

    def _sample_flights(self) -> None:
        total = self.X_full.shape[0]
        if self.config.n_flights > total:
            raise ValueError(
                f"Requested n_flights={self.config.n_flights} exceeds "
                f"available flights={total}"
            )
        idx = self.rng.choice(total, size=self.config.n_flights, replace=False)
        self.idx = idx
        self.X_sel = self.X_full[idx]
        self.u_max_sel = self.u_max_all[idx]
        self.u_cargo_sel = self.u_cargo_all[idx]
        self.u_mail_sel = self.u_mail_all[idx]
        self.u_pax_sel = self.u_pax_all[idx]

    # -------------------------------------------------------------------
    #  Generate slack samples
    # -------------------------------------------------------------------

    def _epsilon_design_matr(self) -> None:
        categorical_cols = FeatureConfig().categorical_features
        numeric_cols = FeatureConfig().numerical_features

        dummies = self.df.select(categorical_cols).to_dummies()
        numeric = self.df.select(numeric_cols).to_numpy().astype(np.float32)
        # TODO fix this as stds are nans
        num_std = (numeric - numeric.mean(axis=0, keepdims=True)) / (numeric.std(axis=0, keepdims=True) + 1e-6)

        self.design_full = np.concatenate([dummies.to_numpy().astype(np.float32), num_std], axis=1)
        self.n_design = self.design_full.shape[1]
        # breakpoint()  # enable interactive debugging when needed

    def _draw_hyperprior_coeffs(self) -> None:
        """Draw hyperprior coefficients for the gamma distribution."""
        self.psi_alpha = self.hyperprior.signed_sample(self.rng, self.n_design)
        self.psi_beta = self.hyperprior.signed_sample(self.rng, self.n_design)

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
        self._epsilon_design_matr()
        self.design_sell = self.design_full[self.idx]
        self._draw_hyperprior_coeffs()
        eps = self._draw_down(self.design_sell)
        # ensure non-negative slack values
        self.z_gen_sel = np.maximum(self.z_gen_sel - eps, 0.0).astype(np.float32)
        # breakpoint()  # enable interactive debugging when needed

    # -------------------------------------------------------------------
    #  Weight sweep
    # -------------------------------------------------------------------

    def _weight_dist(self, n_weight_samples: int) -> None:
        x = self.ship_dist["AW (kg)"].cast(pl.Float64).drop_nulls().to_numpy()
        x = x[x > 0]  # Remove zeros for log-normal fitting

        shape, loc, scale = lognorm.fit(x, floc=0)
        samples = lognorm.rvs(shape, loc=loc, scale=scale, size=n_weight_samples, random_state=self.config.seed)

        self.weight_min = samples.min()
        self.weight_max = samples.max()

        return samples

    def _weight_sweep(self) -> None:
        U, W, S = [], [], []
        for u_vec, z_gen in zip(self.X_sel, self.z_gen_sel):

            # w_max = max(z_gen, self.config.loguniform_low)
            # w_s = np.exp(
            #     self.rng.uniform(
            #         np.log(self.config.loguniform_low), np.log(w_max), size=self.config.k_swaps
            #     )
            # )
            # NOTE: may consider standardizing sample set of weights across each to allow 
            # for standardized comparison
            w_s = self._weight_dist(self.config.k_swaps)

            s = (w_s <= z_gen).astype(np.float32)

            U.append(np.repeat(u_vec[np.newaxis, :], self.config.k_swaps, axis=0))
            W.append(w_s[:, np.newaxis])
            S.append(s[:, np.newaxis])

        self.U = torch.tensor(np.concatenate(U, axis=0), dtype=torch.float32)
        self.W = torch.tensor(np.concatenate(W, axis=0), dtype=torch.float32)
        self.S = torch.tensor(np.concatenate(S, axis=0), dtype=torch.float32)

    # -------------------------------------------------------------------
    #  Public API
    # -------------------------------------------------------------------

    def generate(self) -> dict:
        """
        Run the full pipeline and return a dictionary suitable for torch.save.
        """
        self._sample_flights()
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
        print(f"Weight range: {self.W.min().item():.4f} – {self.W.max().item():.4f}")
        print(f"Number of positive labels: {int(self.S.sum().item())}, Number of negative labels: {int((1 - self.S).sum().item())}")


def _demo() -> None:
    cfg = SyntheticGeneratorConfig()
    generator = SyntheticDataGenerator(cfg)
    U, W, S = generator.generate()
    # breakpoint()  # enable interactive debugging when needed
    generator.save()
    generator.print_sample()
    # breakpoint() # enable interactive debugging when needed


if __name__ == "__main__":
    _demo()
