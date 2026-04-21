from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class RouteInput:
    origin: str
    destination: str
    distance_miles: float
    route_id: str | None = None


@dataclass(frozen=True)
class ShipmentInput:
    weight_kg: float
    pallets: float = 0.0
    equivalent_cost: float = 0.0
    commodity: str | None = None


@dataclass(frozen=True)
class CarrierInput:
    carrier_id: str
    carrier_name: str | None = None
    fleet_tag: str | None = None


@dataclass(frozen=True)
class SampledAirliftOutcome:
    accepted: bool
    acceptance_probability: float
    aircraft_type: str | None
    aircraft_probability: float | None
    aircraft_probabilities: np.ndarray | None
    feature_vector: np.ndarray


class FeatureEncoder(Protocol):
    def encode(self, route: RouteInput, shipment: ShipmentInput, carrier: CarrierInput) -> np.ndarray:
        raise NotImplementedError


class FlatFeatureEncoder:
    def __init__(
        self,
        route_fields: Sequence[str] = ("distance_miles",),
        shipment_fields: Sequence[str] = ("weight_kg", "pallets", "equivalent_cost"),
        carrier_fields: Sequence[str] = (),
        categorical_maps: Mapping[str, Mapping[str, int]] | None = None,
    ):
        self.route_fields = tuple(route_fields)
        self.shipment_fields = tuple(shipment_fields)
        self.carrier_fields = tuple(carrier_fields)
        self.categorical_maps = {} if categorical_maps is None else dict(categorical_maps)

    def _extract(self, obj: Any, field: str) -> float:
        if isinstance(obj, Mapping):
            value = obj[field]
        else:
            value = getattr(obj, field)

        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        if isinstance(value, str):
            mapping = self.categorical_maps.get(field)
            if mapping is None:
                raise TypeError(
                    f"Field '{field}' is categorical. Provide a categorical mapping for FlatFeatureEncoder."
                )
            if value not in mapping:
                raise KeyError(f"Unknown category '{value}' for field '{field}'.")
            return float(mapping[value])
        raise TypeError(f"Field '{field}' must be numeric to use FlatFeatureEncoder.")

    def encode(self, route: RouteInput, shipment: ShipmentInput, carrier: CarrierInput) -> np.ndarray:
        values = [
            *(self._extract(route, field) for field in self.route_fields),
            *(self._extract(shipment, field) for field in self.shipment_fields),
            *(self._extract(carrier, field) for field in self.carrier_fields),
        ]
        return np.asarray(values, dtype=np.float32)


class CarrierAcceptanceNetwork(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    @classmethod
    def from_sklearn(cls, model: Any) -> "CarrierAcceptanceNetwork":
        network = cls(input_dim=model.coef_.shape[1])
        with torch.no_grad():
            network.linear.weight.copy_(torch.as_tensor(model.coef_, dtype=torch.float32))
            network.linear.bias.copy_(torch.as_tensor(model.intercept_, dtype=torch.float32).view(-1))
        return network

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))

    @torch.no_grad()
    def predict_proba(self, x: np.ndarray | torch.Tensor) -> float:
        tensor = torch.as_tensor(x, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return float(self.forward(tensor).squeeze().item())

    @torch.no_grad()
    def sample(self, x: np.ndarray | torch.Tensor, rng: np.random.Generator | None = None) -> tuple[float, bool]:
        probability = self.predict_proba(x)
        generator = rng if rng is not None else np.random.default_rng()
        accepted = bool(generator.binomial(1, probability))
        return probability, accepted


class MultinomialLogitAircraftModel:
    def __init__(
        self,
        coef_: np.ndarray,
        intercept_: np.ndarray,
        classes_: Sequence[str],
        scaler_mean: np.ndarray | None = None,
        scaler_scale: np.ndarray | None = None,
    ):
        self.coef_ = np.asarray(coef_, dtype=np.float64)
        self.intercept_ = np.asarray(intercept_, dtype=np.float64)
        self.classes_ = tuple(str(cls) for cls in classes_)
        self.scaler_mean_ = None if scaler_mean is None else np.asarray(scaler_mean, dtype=np.float64)
        self.scaler_scale_ = None if scaler_scale is None else np.asarray(scaler_scale, dtype=np.float64)

    @classmethod
    def from_sklearn(cls, model: Any, scaler: Any | None = None) -> "MultinomialLogitAircraftModel":
        scaler_mean = None if scaler is None else getattr(scaler, "mean_", None)
        scaler_scale = None if scaler is None else getattr(scaler, "scale_", None)
        return cls(
            coef_=np.asarray(model.coef_),
            intercept_=np.asarray(model.intercept_),
            classes_=np.asarray(model.classes_).tolist(),
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
        )

    def _transform(self, x: np.ndarray | Sequence[float]) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("Aircraft feature vectors must be one-dimensional.")

        if self.scaler_mean_ is not None and self.scaler_scale_ is not None:
            values = (values - self.scaler_mean_) / (self.scaler_scale_ + 1e-12)

        return values

    def predict_proba(self, x: np.ndarray | Sequence[float]) -> np.ndarray:
        values = self._transform(x)

        if len(self.classes_) == 2 and self.coef_.shape[0] == 1:
            logit = float(values @ self.coef_[0] + self.intercept_[0])
            prob_1 = 1.0 / (1.0 + np.exp(-logit))
            return np.asarray([1.0 - prob_1, prob_1], dtype=np.float64)

        logits = values @ self.coef_.T + self.intercept_
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum()

    def sample(
        self,
        x: np.ndarray | Sequence[float],
        rng: np.random.Generator | None = None,
        compatibility_fn: Callable[[str], bool] | None = None,
    ) -> tuple[str, float, np.ndarray]:
        probabilities = self.predict_proba(x)
        generator = rng if rng is not None else np.random.default_rng()

        if compatibility_fn is None:
            chosen = generator.choice(self.classes_, p=probabilities)
            chosen_probability = float(probabilities[self.classes_.index(str(chosen))])
            return str(chosen), chosen_probability, probabilities

        feasible = np.asarray([bool(compatibility_fn(cls)) for cls in self.classes_], dtype=bool)
        if not feasible.any():
            raise ValueError("No feasible aircraft types remain after applying the compatibility filter.")

        filtered = probabilities * feasible.astype(np.float64)
        filtered = filtered / filtered.sum()
        chosen = generator.choice(self.classes_, p=filtered)
        chosen_probability = float(filtered[self.classes_.index(str(chosen))])
        return str(chosen), chosen_probability, filtered


class StochasticAirliftSampler:
    def __init__(
        self,
        feature_encoder: FeatureEncoder,
        acceptance_model: CarrierAcceptanceNetwork,
        aircraft_model: MultinomialLogitAircraftModel | None = None,
        compatibility_fn: Callable[[str, RouteInput, ShipmentInput, CarrierInput], bool] | None = None,
    ):
        self.feature_encoder = feature_encoder
        self.acceptance_model = acceptance_model
        self.aircraft_model = aircraft_model
        self.compatibility_fn = compatibility_fn

    def _aircraft_compatibility(self, aircraft_type: str, route: RouteInput, shipment: ShipmentInput, carrier: CarrierInput) -> bool:
        if self.compatibility_fn is None:
            return True
        return bool(self.compatibility_fn(aircraft_type, route, shipment, carrier))

    def sample(
        self,
        route: RouteInput,
        shipment: ShipmentInput,
        carrier: CarrierInput,
        n_samples: int = 1,
        rng: np.random.Generator | None = None,
    ) -> list[SampledAirliftOutcome]:
        generator = rng if rng is not None else np.random.default_rng()
        feature_vector = self.feature_encoder.encode(route, shipment, carrier)

        outcomes: list[SampledAirliftOutcome] = []
        for _ in range(n_samples):
            acceptance_probability, accepted = self.acceptance_model.sample(feature_vector, rng=generator)

            aircraft_type: str | None = None
            aircraft_probability: float | None = None
            aircraft_probabilities: np.ndarray | None = None

            if accepted and self.aircraft_model is not None:
                aircraft_type, aircraft_probability, aircraft_probabilities = self.aircraft_model.sample(
                    feature_vector,
                    rng=generator,
                    compatibility_fn=lambda aircraft: self._aircraft_compatibility(aircraft, route, shipment, carrier),
                )

            outcomes.append(
                SampledAirliftOutcome(
                    accepted=accepted,
                    acceptance_probability=acceptance_probability,
                    aircraft_type=aircraft_type,
                    aircraft_probability=aircraft_probability,
                    aircraft_probabilities=aircraft_probabilities,
                    feature_vector=feature_vector.copy(),
                )
            )

        return outcomes


__all__ = [
    "RouteInput",
    "ShipmentInput",
    "CarrierInput",
    "SampledAirliftOutcome",
    "FeatureEncoder",
    "FlatFeatureEncoder",
    "CarrierAcceptanceNetwork",
    "MultinomialLogitAircraftModel",
    "StochasticAirliftSampler",
]