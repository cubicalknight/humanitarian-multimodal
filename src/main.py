"""Run cost or feasible-route sensitivity analyses for ``stoc_optimod``.

Workflow:
1. Run ``--prepare`` once to build the network and a shared scenario sample.
2. Submit one Slurm array task per contiguous chunk of cost combinations.
3. Run ``--merge`` after the array is complete to create one summary JSON file.

Each array task reads the prepared problem, builds one Gurobi model, and writes
only its own result files. This avoids repeated data preparation, repeated model
construction within a task, and concurrent writes to shared results.

Use ``--k-sensitivity`` to independently prepare and solve every configured
feasible-route limit with the baseline cost parameters.
Use ``--matched-k-sensitivity`` to compare each configured seed-union
path limit against a per-shipment, strictly distance-ranked path
count.
Use ``--base-case-comparison`` for integrated and myopic baseline solves
with restricted and unrestricted first-stage routing.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import pickle
import sys
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gurobipy as gp
import numpy as np
import polars as pl
from sklearn.neighbors import BallTree

from data_processing import DataProcessing, T100DataProcessing
from stoc_optimod import (
    LegOption,
    Node,
    RouteKey,
    Shipment,
    StochasticOptimizationParameters,
    TwoStageSolver,
    UncertaintyRealization,
    build_feasible_routes_by_shipment,
    build_seed_union_feasible_routes_by_shipment,
    build_seed_union_paths_by_shipment,
    k_shortest_route_paths,
)

COST_FIELDS = (
    "cost_flight",
    "cost_ground",
    "cost_penalty_incompatibility",
)

# Sweep set for --k-sensitivity and --matched-k-sensitivity.
ROUTE_PATH_LIMITS = (1, 2, 5, 10, 20)
# ROUTE_PATH_LIMITS = (10,)

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "num_scenarios": 100,
    "shipment_limit": None,
    "base_parameters": {
        "cost_flight": 4.0,
        "cost_ground": 2.0,
        "cost_penalty_incompatibility": 5.0,
    },
    "parameter_values": {
        "cost_flight": [4.0],
        "cost_ground": [2.0],
        "cost_penalty_incompatibility": [5.0],
    },
    "solver": {
        "threads": None,
        "time_limit_seconds": None,
        "mip_gap": None,
        "quiet": False,
    },
    "network": {
        "nearby_city_radius_miles": 100.0,
        "max_city_attempts": 10,
        "max_ground_distance_miles": 500.0,
        "recourse_path_limit": 10,
        "feasible_path_mode": "distance_only",
        "restrict_first_stage": False,
    },
}

SCRIPT_DIR = Path(__file__).resolve().parent


def _relative_to_script(path: Path) -> Path:
    """Resolve relative CLI paths from this script's directory."""
    return path if path.is_absolute() else SCRIPT_DIR / path


@dataclass(slots=True)
class PreparedProblem:
    """The immutable input shared by all sensitivity-array tasks."""

    shipments: dict[str, Shipment]
    legs: dict[tuple[str, str, str], LegOption]
    scenarios: dict[tuple[str, str, str], UncertaintyRealization]
    seed: int
    num_scenarios: int
    feasible_routes_by_shipment: dict[str, tuple[RouteKey, ...]] | None = None
    feasible_path_mode: str = "distance_only"
    recourse_path_limit: int | None = None
    recourse_path_limits_by_shipment: dict[str, int] | None = None
    cost_seed_pairs: tuple[tuple[float, float], ...] = ()


def _merge_defaults(given: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge a user config with the supported defaults."""
    merged = deepcopy(defaults)
    for key, value in given.items():
        if isinstance(value, dict) and isinstance(defaults.get(key), dict):
            merged[key] = _merge_defaults(value, defaults[key])
        else:
            merged[key] = value
    return merged


class SensitivityConfig:
    """Loads, validates, and expands the JSON sensitivity configuration."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = _merge_defaults(json.loads(path.read_text()), DEFAULT_CONFIG)
        self._validate()

    def _validate(self) -> None:
        unknown_fields = (
            set(self.data["parameter_values"]) | set(self.data["base_parameters"])
        ) - set(COST_FIELDS)
        if unknown_fields:
            raise ValueError(f"Unknown cost fields: {sorted(unknown_fields)}")

        for field in COST_FIELDS:
            values = self.data["parameter_values"].get(
                field,
                [self.data["base_parameters"][field]],
            )
            if not isinstance(values, list) or not values:
                raise ValueError(f"parameter_values.{field} must be a non-empty list.")
            if any(
                value is None or float(value) < 0 for value in values
            ):
                raise ValueError(f"parameter_values.{field} must contain non-negative values.")

        path_limit = self.data["network"]["recourse_path_limit"]
        if (
            not isinstance(path_limit, int)
            or isinstance(path_limit, bool)
            or path_limit <= 0
        ):
            raise ValueError("network.recourse_path_limit must be a positive integer.")

        path_mode = self.data["network"]["feasible_path_mode"]
        if path_mode not in {"distance_only", "seed_union"}:
            raise ValueError(
                "network.feasible_path_mode must be 'distance_only' or 'seed_union'."
            )

        if not isinstance(self.data["network"]["restrict_first_stage"], bool):
            raise ValueError("network.restrict_first_stage must be a boolean.")

    def combinations(self) -> list[dict[str, float | None]]:
        """Return the Cartesian product of configured parameter values."""
        value_lists = [
            self.data["parameter_values"].get(
                field,
                [self.data["base_parameters"][field]],
            )
            for field in COST_FIELDS
        ]
        return [
            dict(zip(COST_FIELDS, combination, strict=True))
            for combination in itertools.product(*value_lists)
        ]

    def with_recourse_path_limit(self, path_limit: int) -> SensitivityConfig:
        """Return an independently configurable copy with one route limit."""
        copied_data = deepcopy(self.data)
        copied_data["network"]["recourse_path_limit"] = path_limit

        copied_config = object.__new__(SensitivityConfig)
        copied_config.path = self.path
        copied_config.data = copied_data
        copied_config._validate()
        return copied_config

    def with_path_preprocessing(
        self,
        *,
        path_limit: int,
        path_mode: str,
        cost_seed_pairs: tuple[tuple[float, float], ...] | None = None,
    ) -> SensitivityConfig:
        """Copy this config with the route-preprocessing inputs replaced.

        ``seed_union`` obtains its seeds from ``parameter_values``.  Supplying
        ``cost_seed_pairs`` is therefore useful for analyses that intentionally
        use one cost vector rather than the entire sensitivity grid.
        """
        copied_data = deepcopy(self.data)
        copied_data["network"]["recourse_path_limit"] = path_limit
        copied_data["network"]["feasible_path_mode"] = path_mode
        if cost_seed_pairs is not None:
            copied_data["parameter_values"]["cost_flight"] = [
                pair[0] for pair in cost_seed_pairs
            ]
            copied_data["parameter_values"]["cost_ground"] = [
                pair[1] for pair in cost_seed_pairs
            ]

        copied_config = object.__new__(SensitivityConfig)
        copied_config.path = self.path
        copied_config.data = copied_data
        copied_config._validate()
        return copied_config

    def cost_seed_pairs(self) -> tuple[tuple[float, float], ...]:
        """Return stable, de-duplicated air/ground cost seeds for Yen paths."""
        air_costs = self.data["parameter_values"].get(
            "cost_flight", [self.data["base_parameters"]["cost_flight"]]
        )
        ground_costs = self.data["parameter_values"].get(
            "cost_ground", [self.data["base_parameters"]["cost_ground"]]
        )
        return tuple(
            dict.fromkeys(
                (float(air_cost), float(ground_cost))
                for air_cost, ground_cost in itertools.product(air_costs, ground_costs)
            )
        )


def _write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON without exposing a partially written result file."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary_path.replace(path)


class _StoreOutputDirectory(argparse.Action):
    """Store a CLI output path and record that the user explicitly chose it."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Path,
        option_string: str | None = None,
    ) -> None:
        setattr(namespace, self.dest, values)
        setattr(namespace, "output_dir_explicit", True)


class ProblemPreparer:
    """Build the original ``stoc_optimod.__main__`` input once."""

    MILES_PER_METER = 0.000621371
    EARTH_RADIUS_MILES = 3958.8

    def __init__(self, config: SensitivityConfig) -> None:
        self.config = config.data
        self.cost_seed_pairs = config.cost_seed_pairs()
        self.rng = np.random.default_rng(self.config["seed"])

    def prepare(self) -> PreparedProblem:
        """Prepare shipments, network legs, and common random-number scenarios."""
        t100_processor = T100DataProcessing()
        t100_processor.rng = np.random.default_rng(self.config["seed"])

        t100_data = t100_processor.filter_data()
        t100_data = t100_processor._geolocate_nodes(t100_data)
        t100_data = t100_processor._calculate_distance(t100_data)

        shipping_processor = DataProcessing()
        shipping_processor.align_from(t100_processor)
        shipping_data = shipping_processor.load_shipping_data()
        shipping_data = shipping_processor._geolocate_nodes(shipping_data)
        shipping_data = shipping_processor._calculate_distance(shipping_data)

        overlap = shipping_processor.get_od_overlap(shipping_data, t100_data)
        overlap = overlap.with_columns(
            pl.col("AW (lbs)").cast(pl.Float64, strict=False),
            pl.col("Commercial Cost for First Mile").cast(pl.Float64, strict=False),
            pl.col("Commercial Cost for Last Mile").cast(pl.Float64, strict=False),
        )

        legs, nodes = self._build_air_network(t100_data)
        shipments = self._build_shipments_and_ground_links(overlap, nodes, legs)

        if self.config["shipment_limit"] is not None:
            shipments = dict(
                itertools.islice(shipments.items(), int(self.config["shipment_limit"]))
            )
        if not shipments:
            raise ValueError("No valid overlapping shipments were found.")

        path_limit = int(self.config["network"]["recourse_path_limit"])
        path_mode = self.config["network"]["feasible_path_mode"]
        cost_seed_pairs = self.cost_seed_pairs
        if path_mode == "distance_only":
            feasible_routes_by_shipment = build_feasible_routes_by_shipment(
                shipments, legs, max_paths=path_limit
            )
        else:
            feasible_routes_by_shipment = build_seed_union_feasible_routes_by_shipment(
                shipments,
                legs,
                cost_seed_pairs=cost_seed_pairs,
                max_paths=path_limit,
            )
        feasible_air_routes = {
            route
            for routes in feasible_routes_by_shipment.values()
            for route in routes
            if route[2] == "air"
        }
        scenarios = self._build_scenarios(
            legs,
            t100_processor,
            feasible_air_routes,
        )
        return PreparedProblem(
            shipments=shipments,
            legs=legs,
            scenarios=scenarios,
            seed=int(self.config["seed"]),
            num_scenarios=int(self.config["num_scenarios"]),
            feasible_routes_by_shipment=feasible_routes_by_shipment,
            feasible_path_mode=path_mode,
            recourse_path_limit=path_limit,
            cost_seed_pairs=cost_seed_pairs if path_mode == "seed_union" else (),
        )

    @staticmethod
    def _build_air_network(
        t100_data: pl.DataFrame,
    ) -> tuple[dict[tuple[str, str, str], LegOption], dict[str, Node]]:
        """Create airport nodes and one air leg for every T100 origin/destination."""
        legs: dict[tuple[str, str, str], LegOption] = {}
        nodes: dict[str, Node] = {}

        for row in t100_data.iter_rows(named=True):
            origin = str(row["ORIGIN"])
            destination = str(row["DEST"])

            nodes.setdefault(
                origin,
                Node(origin, float(row["Origin_Lat"]), float(row["Origin_Lon"]), "air"),
            )
            nodes.setdefault(
                destination,
                Node(
                    destination,
                    float(row["Destination_Lat"]),
                    float(row["Destination_Lon"]),
                    "air",
                ),
            )

            route = (origin, destination, "air")
            legs[route] = LegOption(
                route_id=f"{origin}_{destination}",
                origin=nodes[origin],
                destination=nodes[destination],
                distance_miles=float(row["DISTANCE"]),
                mode="air",
                mu_slack=float(row["MU_SLACK"]),
                sigma_slack=float(row["SIGMA_SLACK"]),
            )

        return legs, nodes

    def _build_shipments_and_ground_links(
        self,
        overlap: pl.DataFrame,
        nodes: dict[str, Node],
        legs: dict[tuple[str, str, str], LegOption],
    ) -> dict[str, Shipment]:
        """Build shipment endpoint nodes and cached OSRM ground connections."""
        import geonamescache
        import requests

        network_config = self.config["network"]
        airport_ids = sorted(nodes)
        airport_tree = BallTree(
            np.radians([(nodes[airport].latitude, nodes[airport].longitude) for airport in airport_ids]),
            metric="haversine",
        )
        cities = [
            {
                "lat": city["latitude"],
                "lon": city["longitude"],
                "country": city["countrycode"],
            }
            for city in geonamescache.GeonamesCache().get_cities().values()
        ]
        city_tree = BallTree(
            np.radians([(city["lat"], city["lon"]) for city in cities]),
            metric="haversine",
        )

        cache_path = Path(__file__).parent / "cache" / "routing_cache.json"
        routing_cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        cache_is_dirty = False
        airport_countries: dict[str, str] = {}

        def airport_country(airport_id: str) -> str:
            if airport_id not in airport_countries:
                airport = nodes[airport_id]
                nearest_city = city_tree.query(
                    np.radians([(airport.latitude, airport.longitude)]),
                    k=1,
                )[1][0][0]
                airport_countries[airport_id] = cities[nearest_city]["country"]
            return airport_countries[airport_id]

        def drivable_route(
            origin_lat: float,
            origin_lon: float,
            destination_lat: float,
            destination_lon: float,
        ) -> dict[str, Any]:
            """Use a local cache around one OSRM driving-route request."""
            nonlocal cache_is_dirty
            raw_key = (
                f"{origin_lat:.5f},{origin_lon:.5f},"
                f"{destination_lat:.5f},{destination_lon:.5f}"
            )
            key = hashlib.sha1(raw_key.encode()).hexdigest()

            if key not in routing_cache:
                try:
                    url = (
                        "https://router.project-osrm.org/route/v1/driving/"
                        f"{origin_lon},{origin_lat};{destination_lon},{destination_lat}"
                        "?overview=false"
                    )
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    routes = data.get("routes", [])
                    routing_cache[key] = {
                        "feasible": data.get("code") == "Ok" and bool(routes),
                        "distance_m": routes[0]["distance"] if routes else None,
                    }
                except requests.RequestException:
                    routing_cache[key] = {"feasible": False, "distance_m": None}
                cache_is_dirty = True

            return routing_cache[key]

        endpoint_nodes: dict[tuple[str, str, str], Node] = {}
        for row in overlap.iter_rows(named=True):
            ngo_id = str(row["NGO ID"])
            endpoint_specs = (
                ("O", str(row["ORIGIN"]), "Commercial Cost for First Mile"),
                ("D", str(row["DEST"]), "Commercial Cost for Last Mile"),
            )
            for side, airport_id, cost_column in endpoint_specs:
                endpoint_key = (ngo_id, airport_id, side)
                if row[cost_column] is None or endpoint_key in endpoint_nodes:
                    continue

                airport = nodes[airport_id]
                nearby_city_indices = city_tree.query_radius(
                    np.radians([(airport.latitude, airport.longitude)]),
                    r=float(network_config["nearby_city_radius_miles"])
                    / self.EARTH_RADIUS_MILES,
                )[0]
                selected_city = next(
                    (
                        cities[index]
                        for index in nearby_city_indices[
                            : int(network_config["max_city_attempts"])
                        ]
                        if cities[index]["country"] == airport_country(airport_id)
                        and drivable_route(
                            cities[index]["lat"],
                            cities[index]["lon"],
                            airport.latitude,
                            airport.longitude,
                        )["feasible"]
                    ),
                    None,
                )
                if selected_city is None:
                    raise ValueError(f"No drivable nearby city for endpoint {endpoint_key}.")

                endpoint = Node(
                    node_id=f"{ngo_id}_{airport_id}_{side}",
                    latitude=selected_city["lat"],
                    longitude=selected_city["lon"],
                    connection_type="gnd",
                )
                endpoint_nodes[endpoint_key] = endpoint
                nodes[endpoint.node_id] = endpoint

        for endpoint in endpoint_nodes.values():
            nearby_airports = airport_tree.query_radius(
                np.radians([(endpoint.latitude, endpoint.longitude)]),
                r=float(network_config["max_ground_distance_miles"])
                / self.EARTH_RADIUS_MILES,
            )[0]
            for airport_index in nearby_airports:
                airport = nodes[airport_ids[airport_index]]
                route = drivable_route(
                    endpoint.latitude,
                    endpoint.longitude,
                    airport.latitude,
                    airport.longitude,
                )
                if not route["feasible"]:
                    continue

                distance_miles = float(route["distance_m"]) * self.MILES_PER_METER
                if distance_miles > float(network_config["max_ground_distance_miles"]):
                    continue

                legs[(endpoint.node_id, airport.node_id, "ground")] = LegOption(
                    f"{endpoint.node_id}_{airport.node_id}_ground",
                    endpoint,
                    airport,
                    distance_miles,
                    "ground",
                )
                legs[(airport.node_id, endpoint.node_id, "ground")] = LegOption(
                    f"{airport.node_id}_{endpoint.node_id}_ground",
                    airport,
                    endpoint,
                    distance_miles,
                    "ground",
                )

        if cache_is_dirty:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(routing_cache))

        shipments: dict[str, Shipment] = {}
        for row in overlap.iter_rows(named=True):
            weight = row["AW (lbs)"]
            if weight is None or float(weight) <= 0:
                continue

            ngo_id = str(row["NGO ID"])
            origin_id = str(row["ORIGIN"])
            destination_id = str(row["DEST"])
            shipment_id = str(row["Shipment ID"])
            shipments[shipment_id] = Shipment(
                shipment_id=shipment_id,
                weight=float(weight),
                origin=endpoint_nodes.get((ngo_id, origin_id, "O"), nodes[origin_id]),
                destination=endpoint_nodes.get(
                    (ngo_id, destination_id, "D"),
                    nodes[destination_id],
                ),
            )

        return shipments

    def _build_scenarios(
        self,
        legs: dict[tuple[str, str, str], LegOption],
        t100_processor: T100DataProcessing,
        feasible_air_routes: set[tuple[str, str, str]],
    ) -> dict[tuple[str, str, str], UncertaintyRealization]:
        """Create shared scenarios only for air routes used by sparse recourse."""
        npz = np.load(Path(__file__).parent / "psi_bar.npz")
        psi_bar = npz["psi_bar"]
        n_design = int(npz["n_design"])
        psi_alpha, psi_beta = psi_bar[:n_design], psi_bar[n_design:]
        scenario_count = int(self.config["num_scenarios"])

        scenarios: dict[tuple[str, str, str], UncertaintyRealization] = {}
        for route, leg in legs.items():
            if route not in feasible_air_routes:
                continue

            slack = t100_processor._truncated_slack_samples(
                np.array([leg.mu_slack]),
                np.array([leg.sigma_slack]),
                scenario_count,
            ).ravel()
            design = np.asarray(leg.u_vec)
            alpha = np.exp(design @ psi_alpha / np.sqrt(n_design))
            beta = np.exp(design @ psi_beta / np.sqrt(n_design))
            drawdown = self.rng.gamma(alpha, beta, scenario_count) * float(leg.mu_slack)
            realizations = np.maximum(slack - drawdown, 0.0).astype(float).tolist()

            scenarios[route] = UncertaintyRealization(
                leg=leg,
                num_scenarions=scenario_count,
                scenario_realize=realizations,
            )

        return scenarios


class SensitivityRunner:
    """Solve sensitivity combinations over one fixed constraint matrix.

    The prepared problem fixes the network, shipments, and scenario realizations.
    Cost sensitivity therefore changes only the linear objective coefficients, so a
    worker can build the (potentially large) model once and re-use it for several
    combinations.
    """

    def __init__(self, config: SensitivityConfig, prepared: PreparedProblem) -> None:
        self.config = config
        self.prepared = prepared

    def run(self, run_index: int, output_dir: Path) -> Path:
        """Run one combination (kept for backwards-compatible array jobs)."""
        return self.run_many((run_index,), output_dir)[0]

    def run_many(self, run_indices: Sequence[int], output_dir: Path) -> list[Path]:
        """Run combinations sequentially while reusing one Gurobi model.

        Only the objective coefficients vary across the configured combinations.
        The variable and constraint matrix are consequently constructed exactly
        once for this method call.
        """
        combinations = self.config.combinations()
        indices = tuple(run_indices)
        if not indices:
            return []
        for run_index in indices:
            if not 0 <= run_index < len(combinations):
                raise IndexError(f"run index must be in 0..{len(combinations) - 1}")

        model, solver, first_stage, keep, reassign = self._build_model()

        output_dir.mkdir(parents=True, exist_ok=True)
        result_paths: list[Path] = []
        for run_index in indices:
            parameter_values = (
                dict(self.config.data["base_parameters"]) | combinations[run_index]
            )
            self._set_cost_objective(
                model,
                solver,
                first_stage,
                keep,
                reassign,
                StochasticOptimizationParameters(**parameter_values),
            )
            model.optimize()
            if self.config.data["network"]["restrict_first_stage"]:
                self._validate_first_stage_routes(
                    run_index,
                    model,
                    solver,
                    first_stage,
                )

            result = self._build_result(
                run_index,
                parameter_values,
                model,
                solver,
                first_stage,
                keep,
                reassign,
            )
            result_path = output_dir / f"run_{run_index:06d}.json"
            _write_json_atomic(result_path, result)
            result_paths.append(result_path)

        return result_paths

    def solve_once(
        self,
        parameters: dict[str, float | None],
        *,
        run_label: str,
    ) -> dict[str, Any]:
        """Build and solve one prepared problem for explicitly supplied costs.

        This intentionally does not use ``parameter_values``. It supports modes
        whose inputs change the constraint matrix, such as route-limit
        sensitivity, where reusing a model across cases is not valid.
        """
        recourse_connectivity = self._recourse_connectivity()
        model, solver, first_stage, keep, reassign = self._build_model()
        self._set_cost_objective(
            model,
            solver,
            first_stage,
            keep,
            reassign,
            StochasticOptimizationParameters(**parameters),
        )
        model.optimize()
        # breakpoint()
        if self.config.data["network"]["restrict_first_stage"]:
            self._validate_first_stage_routes(run_label, model, solver, first_stage)
        result = self._build_result(
            None,
            parameters,
            model,
            solver,
            first_stage,
            keep,
            reassign,
        )
        result["recourse_connectivity"] = recourse_connectivity
        return result

    def solve_comparison_case(self, *, method: str, run_label: str) -> dict[str, Any]:
        """Solve a baseline case, preserving myopic decisions if recourse fails."""
        if method not in {"integrated", "myopic"}:
            raise ValueError(f"Unknown comparison method: {method}")
        parameters = dict(self.config.data["base_parameters"])
        params = StochasticOptimizationParameters(**parameters)
        allow_air_leg_dropping = (
            method == "myopic" and not self.config.data["network"]["restrict_first_stage"]
        )
        model, solver, first_stage, keep, reassign = self._build_model(
            first_stage_only=method == "myopic"
        )
        stages = {}
        saved_decisions = {}
        first_stage_cost = None

        def record_stage(name: str) -> None:
            stages[name] = {
                "status": int(model.Status),
                "solution_count": int(model.SolCount),
                "runtime_seconds": float(model.Runtime),
            }

        try:
            if method == "myopic":
                self._set_first_stage_cost_objective(model, solver, first_stage, params)
            else:
                self._set_cost_objective(model, solver, first_stage, keep, reassign, params)
            model.optimize()
            record_stage("first_stage" if method == "myopic" else "integrated")
            diagnostics = self._first_stage_route_diagnostics(model, solver, first_stage)
            if self.config.data["network"]["restrict_first_stage"]:
                self._validate_first_stage_routes(run_label, model, solver, first_stage)
            elif diagnostics["violations"]:
                print(f"Warning: {run_label}: first-stage routes outside recourse sets "
                      f"for {len(diagnostics['violations'])} shipment(s); continuing.")

            if model.SolCount:
                # Snapshot before modifying the model invalidates solution attributes.
                values = {key: round(variable.X) for key, variable in first_stage.items()}
                first_stage_cost = sum(
                    values[s, *route] * solver.shipments[s].weight
                    * solver.legs[route].distance_miles
                    * (params.cost_flight if route[2] == "air" else params.cost_ground)
                    for s in solver.S for route in solver.R
                )
                for shipment_id in solver.S:
                    saved_decisions[shipment_id] = {"first_stage": [], "recourse": {}}
                    self._add_first_stage_decisions(
                        saved_decisions[shipment_id], shipment_id, solver, first_stage
                    )
                if method == "myopic":
                    for key, variable in first_stage.items():
                        variable.LB = values[key]
                        variable.UB = values[key]
                    model, keep, reassign, _ = solver.stage_two_setup(
                        model, first_stage, range(self.prepared.num_scenarios),
                        self.prepared.scenarios,
                        allow_air_leg_dropping=allow_air_leg_dropping,
                    )
                    self._configure_gurobi(model, self.config.data["solver"])
                    self._set_cost_objective(
                        model, solver, first_stage, keep, reassign, params
                    )
                    model.optimize()
                    record_stage("recourse")

            result = self._build_result(
                None, parameters, model, solver, first_stage, keep, reassign
            )
            if not model.SolCount and saved_decisions:
                result["decision_variables_by_shipment"] = saved_decisions
            total_cost = result["objective_value"]
            result.update({
                "case": run_label,
                "method": method,
                "allow_air_leg_dropping": allow_air_leg_dropping,
                "stages": stages,
                "first_stage_validation": diagnostics,
                "first_stage_cost": first_stage_cost,
                "expected_net_recourse_cost": (
                    total_cost - first_stage_cost if total_cost is not None else None
                ),
                "total_expected_cost": total_cost,
                "runtime_seconds": sum(stage["runtime_seconds"] for stage in stages.values()),
                "recourse_connectivity": self._recourse_connectivity(),
            })
            return result
        finally:
            model.dispose()

    def _recourse_connectivity(self) -> dict[str, Any]:
        """Report scenario/path-set combinations with no available OD path.

        The second stage requires a complete route in every sampled scenario.
        Thus one unavailable cut in a small candidate-path union is sufficient
        to make the corresponding model infeasible.  This check uses the same
        air-leg availability rule as ``TwoStageSolver.stage_two_setup``.
        """
        feasible_routes = self.prepared.feasible_routes_by_shipment
        if feasible_routes is None:
            return {"checked": False, "unreachable_scenario_count": None}

        examples: list[dict[str, Any]] = []
        unreachable_count = 0
        for shipment_id, shipment in self.prepared.shipments.items():
            routes = feasible_routes[shipment_id]
            for scenario_index in range(self.prepared.num_scenarios):
                outgoing: dict[str, list[str]] = {}
                for route in routes:
                    is_available = route[2] == "ground" or (
                        self.prepared.scenarios[route].scenario_realize[scenario_index]
                        >= shipment.weight
                    )
                    if is_available:
                        outgoing.setdefault(route[0], []).append(route[1])

                reached = {shipment.origin.node_id}
                frontier = [shipment.origin.node_id]
                while frontier:
                    node = frontier.pop()
                    for next_node in outgoing.get(node, ()):
                        if next_node not in reached:
                            reached.add(next_node)
                            frontier.append(next_node)

                if shipment.destination.node_id not in reached:
                    unreachable_count += 1
                    if len(examples) < 20:
                        examples.append(
                            {
                                "shipment_id": shipment_id,
                                "scenario_index": scenario_index,
                                "origin": shipment.origin.node_id,
                                "destination": shipment.destination.node_id,
                            }
                        )

        return {
            "checked": True,
            "unreachable_scenario_count": unreachable_count,
            "unreachable_examples": examples,
        }

    def _build_model(self, *, first_stage_only: bool = False) -> tuple[Any, TwoStageSolver, Any, Any, Any]:
        """Create the fixed model and decision variables for one prepared input."""
        solver_options = self.config.data["solver"]
        feasible_routes = getattr(
            self.prepared,
            "feasible_routes_by_shipment",
            None,
        )
        if feasible_routes is None:
            print(
                "Prepared problem has no stored recourse route sets; computing them "
                "now. Re-run --prepare to also prune stored scenario realizations."
            )
        solver = TwoStageSolver(
            shipments=self.prepared.shipments,
            legs=self.prepared.legs,
            # Cost parameters do not appear in the constraints. The base values
            # are only used while the fixed model is built; each solve updates the
            # objective below.
            params=StochasticOptimizationParameters(
                **self.config.data["base_parameters"]
            ),
            solver_quiet=bool(solver_options["quiet"]),
            feasible_routes_by_shipment=feasible_routes,
            recourse_path_limit=int(
                self.config.data["network"]["recourse_path_limit"]
            ),
            restrict_first_stage=bool(
                self.config.data["network"]["restrict_first_stage"]
            ),
        )

        model, first_stage, first_stage_cost = solver.stage_one_setup(
            gp.Model("sensitivity")
        )
        if first_stage_only:
            self._configure_gurobi(model, solver_options)
            model.setObjective(0.0, gp.GRB.MINIMIZE)
            return model, solver, first_stage, None, None
        model, keep, reassign, recourse_cost = solver.stage_two_setup(
            model,
            first_stage,
            range(self.prepared.num_scenarios),
            self.prepared.scenarios,
        )
        self._configure_gurobi(model, solver_options)

        # The setup methods return expressions for their standalone callers, but
        # this runner updates objective coefficients in place instead. Explicitly
        # clear the objective before applying the first parameter combination.
        del first_stage_cost, recourse_cost
        model.setObjective(0.0, gp.GRB.MINIMIZE)
        return model, solver, first_stage, keep, reassign

    @staticmethod
    def _first_stage_route_diagnostics(
        model: gp.Model,
        solver: TwoStageSolver,
        first_stage: Any,
    ) -> dict[str, Any]:
        """Describe selected legs outside shipment-specific recourse sets."""
        if model.SolCount == 0:
            return {"checked": False, "passed": None, "violations": []}
        violations = []
        for shipment_id in solver.S:
            feasible_routes = solver.feasible_route_sets[shipment_id]
            selected_routes = tuple(
                route
                for route in solver.R
                if first_stage[shipment_id, *route].X > 0.5
            )
            routes_outside_recourse_set = tuple(
                route for route in selected_routes if route not in feasible_routes
            )
            if routes_outside_recourse_set:
                violations.append({
                    "shipment_id": shipment_id,
                    "intended_origin": solver.shipments[shipment_id].origin.node_id,
                    "intended_destination": solver.shipments[shipment_id].destination.node_id,
                    "selected_first_stage_routes": selected_routes,
                    "routes_outside_recourse_set": routes_outside_recourse_set,
                })
        return {"checked": True, "passed": not violations, "violations": violations}

    @staticmethod
    def _validate_first_stage_routes(
        run_index: int | str,
        model: gp.Model,
        solver: TwoStageSolver,
        first_stage: Any,
    ) -> None:
        """Fail fast when the first stage leaves sparse recourse."""
        violations = SensitivityRunner._first_stage_route_diagnostics(
            model, solver, first_stage
        )["violations"]
        if violations:
            details = "\n".join(
                "\n".join(
                    (
                        f"  shipment={violation['shipment_id']!r}",
                        "    intended_origin="
                        f"{violation['intended_origin']!r}",
                        "    intended_destination="
                        f"{violation['intended_destination']!r}",
                        f"    selected_first_stage_routes={violation['selected_first_stage_routes']!r}",
                        "    routes_outside_recourse_set="
                        f"{violation['routes_outside_recourse_set']!r}",
                    )
                )
                for violation in violations
            )
            raise RuntimeError(
                f"Sensitivity run {run_index} selected first-stage routes outside "
                f"the shipment-specific recourse sets:\n{details}"
            )

    @staticmethod
    def _set_first_stage_cost_objective(
        model: gp.Model,
        solver: TwoStageSolver,
        first_stage: Any,
        params: StochasticOptimizationParameters,
        *,
        include_shipment_weight: bool = True,
    ) -> None:
        """Set first-stage objective coefficients for one cost combination.

        ``include_shipment_weight=False`` is a diagnostic option that leaves
        shipment weight out of first-stage leg costs.
        """
        first_stage_variables = []
        first_stage_coefficients = []
        for shipment_id in solver.S:
            shipment = solver.shipments[shipment_id]
            shipment_cost_multiplier = shipment.weight if include_shipment_weight else 1.0
            for route in solver.R:
                unit_cost = (
                    params.cost_flight
                    if route[2] == "air"
                    else params.cost_ground
                )
                first_stage_variables.append(first_stage[shipment_id, *route])
                first_stage_coefficients.append(
                    unit_cost
                    * shipment_cost_multiplier
                    * solver.legs[route].distance_miles
                )
        model.setAttr(
            gp.GRB.Attr.Obj,
            first_stage_variables,
            first_stage_coefficients,
        )

    def _set_cost_objective(
        self,
        model: gp.Model,
        solver: TwoStageSolver,
        first_stage: Any,
        keep: Any,
        reassign: Any,
        params: StochasticOptimizationParameters,
        *,
        include_shipment_weight: bool = True,
    ) -> None:
        """Replace objective coefficients for one two-stage cost combination."""
        self._set_first_stage_cost_objective(
            model,
            solver,
            first_stage,
            params,
            include_shipment_weight=include_shipment_weight,
        )

        scenario_probability = 1.0 / self.prepared.num_scenarios

        # Subtracting the expected unused-assignment credit makes the net
        # coefficient of each recourse-eligible first-stage variable zero:
        # C(x) - E[C(x)] = 0. Routes outside the sparse recourse set retain
        # their ordinary first-stage coefficients.
        credited_first_stage_variables = [
            first_stage[shipment_id, *route]
            for shipment_id in solver.S
            for route in solver.feasible_by_shipment[shipment_id]
        ]
        model.setAttr(
            gp.GRB.Attr.Obj,
            credited_first_stage_variables,
            [0.0] * len(credited_first_stage_variables),
        )

        kept_coefficients = []
        reassignment_coefficients = []
        for shipment_id in solver.S:
            shipment = solver.shipments[shipment_id]
            shipment_cost_multiplier = (
                shipment.weight if include_shipment_weight else 1.0
            )
            for route in solver.feasible_by_shipment[shipment_id]:
                original_unit_cost = (
                    params.cost_flight
                    if route[2] == "air"
                    else params.cost_ground
                )
                distance = solver.legs[route].distance_miles
                original_leg_cost = (
                    original_unit_cost * shipment_cost_multiplier * distance
                )
                kept_coefficients.append(original_leg_cost * scenario_probability)
                reassignment_coefficients.append(
                    (original_leg_cost + params.cost_penalty_incompatibility)
                    * scenario_probability
                )

        # Updating a scenario at a time avoids retaining a second S x R x Omega
        # Python list of Gurobi variable objects solely for coefficient updates.
        scenario_coefficients = kept_coefficients + reassignment_coefficients
        for scenario_index in range(self.prepared.num_scenarios):
            kept_variables = [
                keep[shipment_id, *route, scenario_index]
                for shipment_id in solver.S
                for route in solver.feasible_by_shipment[shipment_id]
            ]
            reassignment_variables = [
                reassign[shipment_id, *route, scenario_index]
                for shipment_id in solver.S
                for route in solver.feasible_by_shipment[shipment_id]
            ]
            model.setAttr(
                gp.GRB.Attr.Obj,
                kept_variables + reassignment_variables,
                scenario_coefficients,
            )

        model.ModelSense = gp.GRB.MINIMIZE

    @staticmethod
    def _configure_gurobi(model: gp.Model, options: dict[str, Any]) -> None:
        """Respect Slurm's CPU allocation unless a config override is supplied."""
        allocated_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        model.Params.Threads = int(options["threads"] or allocated_threads)
        if options["time_limit_seconds"] is not None:
            model.Params.TimeLimit = float(options["time_limit_seconds"])
        if options["mip_gap"] is not None:
            model.Params.MIPGap = float(options["mip_gap"])

    def _build_result(
        self,
        run_index: int | None,
        parameters: dict[str, float | None],
        model: gp.Model,
        solver: TwoStageSolver,
        first_stage: Any,
        keep: Any,
        reassign: Any,
    ) -> dict[str, Any]:
        """Keep only non-zero values, grouped by shipment, to control JSON size."""
        has_solution = model.SolCount > 0
        decisions = {
            shipment_id: {"first_stage": [], "recourse": {}}
            for shipment_id in solver.S
        }

        if has_solution:
            for shipment_id in solver.S:
                self._add_first_stage_decisions(
                    decisions[shipment_id],
                    shipment_id,
                    solver,
                    first_stage,
                )
                self._add_recourse_decisions(
                    decisions[shipment_id],
                    shipment_id,
                    solver,
                    keep,
                    reassign,
                )

        result: dict[str, Any] = {
            "parameters": parameters,
            "seed": self.prepared.seed,
            "num_scenarios": self.prepared.num_scenarios,
            "feasible_path_mode": getattr(
                self.prepared, "feasible_path_mode", "distance_only"
            ),
            "recourse_path_limit": getattr(
                self.prepared, "recourse_path_limit", None),
            "recourse_path_limits_by_shipment": getattr(
                self.prepared, "recourse_path_limits_by_shipment", None
            ),
            "cost_seed_pairs": [
                list(pair)
                for pair in getattr(self.prepared, "cost_seed_pairs", ())
            ],
            "restrict_first_stage": bool(
                self.config.data["network"]["restrict_first_stage"]
            ),
            "status": int(model.Status),
            "objective_value": float(model.ObjVal) if has_solution else None,
            "runtime_seconds": float(model.Runtime),
            "decision_variables_by_shipment": decisions,
        }
        if run_index is not None:
            result = {"run_index": run_index} | result
        return result

    @staticmethod
    def _add_first_stage_decisions(
        shipment_result: dict[str, Any],
        shipment_id: str,
        solver: TwoStageSolver,
        first_stage: Any,
    ) -> None:
        for route in solver.R:
            value = first_stage[shipment_id, *route].X
            if value > 1e-6:
                shipment_result["first_stage"].append(
                    {
                        "origin": route[0],
                        "destination": route[1],
                        "mode": route[2],
                        "value": value,
                    }
                )

    def _add_recourse_decisions(
        self,
        shipment_result: dict[str, Any],
        shipment_id: str,
        solver: TwoStageSolver,
        keep: Any,
        reassign: Any,
    ) -> None:
        for scenario_index in range(self.prepared.num_scenarios):
            scenario_decisions = []
            for route in solver.feasible_by_shipment[shipment_id]:
                keep_value = keep[shipment_id, *route, scenario_index].X
                reassign_value = reassign[shipment_id, *route, scenario_index].X
                if keep_value > 1e-6 or reassign_value > 1e-6:
                    scenario_decisions.append(
                        {
                            "origin": route[0],
                            "destination": route[1],
                            "mode": route[2],
                            "keep": keep_value,
                            "reassign": reassign_value,
                        }
                    )
            if scenario_decisions:
                shipment_result["recourse"][str(scenario_index)] = scenario_decisions


def _routes_from_paths(paths: Sequence[tuple[RouteKey, ...]]) -> tuple[RouteKey, ...]:
    """Return an ordered route union without changing the path count."""
    return tuple(dict.fromkeys(route for path in paths for route in path))


def _cost_weighted_path_unions(
    shipments: dict[str, Shipment],
    legs: dict[RouteKey, LegOption],
    *,
    air_cost: float,
    ground_cost: float,
    max_paths: int,
) -> dict[str, tuple[tuple[RouteKey, ...], ...]]:
    """Get each shipment's unique K-path union under baseline transport cost."""
    unions: dict[str, tuple[tuple[RouteKey, ...], ...]] = {}
    for shipment_id, shipment in shipments.items():
        def route_cost(route: RouteKey, leg: LegOption) -> float:
            if route[2] == "air":
                unit_cost = air_cost
            elif route[2] == "ground":
                unit_cost = ground_cost
            else:
                raise ValueError(f"Route {route!r} has unsupported mode")
            return shipment.weight * leg.distance_miles * unit_cost

        paths = tuple(dict.fromkeys(k_shortest_route_paths(
            legs,
            shipment.origin.node_id,
            shipment.destination.node_id,
            max_paths=max_paths,
            route_cost=route_cost,
        )))
        if not paths or not any(paths):
            raise ValueError(f"No directed route path for shipment {shipment_id!r}")
        unions[shipment_id] = paths
    return unions


def _distance_paths_with_matching_counts(
    shipments: dict[str, Shipment],
    legs: dict[RouteKey, LegOption],
    path_counts: dict[str, int],
) -> dict[str, tuple[tuple[RouteKey, ...], ...]]:
    """Return exactly the requested number of strictly distance-ranked paths."""
    matched: dict[str, tuple[tuple[RouteKey, ...], ...]] = {}
    for shipment_id, shipment in shipments.items():
        requested_count = path_counts[shipment_id]
        paths = k_shortest_route_paths(
            legs,
            shipment.origin.node_id,
            shipment.destination.node_id,
            max_paths=requested_count,
        )
        if len(paths) != requested_count:
            raise ValueError(
                f"Shipment {shipment_id!r} has only {len(paths)} strictly distance-"
                f"ranked paths; cannot match its cost-weighted union of "
                f"{requested_count} paths."
            )
        matched[shipment_id] = paths
    return matched


class MatchedPathSensitivityRunner:
    """Compare every configured seed-union K with equal-count distance paths.

    The cost branch uses all configured cost seeds plus raw distance. Its distinct paths
    form a per-shipment union, whose size becomes the K used by the distance
    branch for that shipment.  This avoids treating a shared global K as if it
    represented an equal candidate-path count for all shipments.
    """

    def __init__(self, config: SensitivityConfig) -> None:
        self.config = config

    def run(self, output_dir: Path) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        baseline = dict(self.config.data["base_parameters"])
        cost_pairs = self.config.cost_seed_pairs()

        results: list[dict[str, Any]] = []
        result_paths: list[Path] = []
        for path_limit in ROUTE_PATH_LIMITS:
            # Prepare exactly the same seed union and scenarios as K sensitivity.
            cost_config = self.config.with_path_preprocessing(
                path_limit=path_limit,
                path_mode="seed_union",
            )
            cost_prepared = ProblemPreparer(cost_config).prepare()
            cost_paths = build_seed_union_paths_by_shipment(
                cost_prepared.shipments,
                cost_prepared.legs,
                cost_seed_pairs=cost_pairs,
                max_paths=path_limit,
            )
            path_counts = {
                shipment_id: len(paths) for shipment_id, paths in cost_paths.items()
            }
            cost_prepared.recourse_path_limits_by_shipment = path_counts

            distance_config = self.config.with_path_preprocessing(
                path_limit=max(path_counts.values()), path_mode="distance_only"
            )
            distance_prepared = ProblemPreparer(distance_config).prepare()
            distance_paths = _distance_paths_with_matching_counts(
                distance_prepared.shipments, distance_prepared.legs, path_counts
            )
            distance_prepared.feasible_routes_by_shipment = {
                shipment_id: _routes_from_paths(paths)
                for shipment_id, paths in distance_paths.items()
            }
            distance_prepared.recourse_path_limit = None
            distance_prepared.recourse_path_limits_by_shipment = path_counts

            case_results: list[dict[str, Any]] = []
            for filename, label, case_config, prepared in (
                (
                    f"cost_weighted_k_{path_limit:03d}.json",
                    f"Seed-union K sensitivity (K={path_limit})",
                    cost_config,
                    cost_prepared,
                ),
                (
                    f"distance_only_matched_k_{path_limit:03d}.json",
                    f"Distance-only matched-K sensitivity (cost K={path_limit})",
                    distance_config,
                    distance_prepared,
                ),
            ):
                result = SensitivityRunner(case_config, prepared).solve_once(
                    baseline, run_label=label
                )
                result["cost_weighted_path_limit"] = path_limit
                result["matched_distance_path_limits_by_shipment"] = path_counts
                path = output_dir / filename
                _write_json_atomic(path, result)
                case_results.append(result)
                result_paths.append(path)
            results.append({"cost_weighted_path_limit": path_limit, "cases": case_results})

        summary_path = output_dir / "matched_k_summary.json"
        _write_json_atomic(
            summary_path,
            {
                "analysis": "matched_cost_weighted_vs_distance_path_sensitivity",
                "base_parameters": baseline,
                "route_path_limits": list(ROUTE_PATH_LIMITS),
                "results": [
                    {
                        "cost_weighted_path_limit": row["cost_weighted_path_limit"],
                        "cases": [
                            {
                                "feasible_path_mode": result["feasible_path_mode"],
                                "status": result["status"],
                                "objective_value": result["objective_value"],
                                "runtime_seconds": result["runtime_seconds"],
                            }
                            for result in row["cases"]
                        ],
                    }
                    for row in results
                ],
            },
        )
        result_paths.append(summary_path)
        return result_paths


class RouteLimitSensitivityRunner:
    """Independently solve a fixed set of feasible-route preprocessing limits."""

    def __init__(self, config: SensitivityConfig) -> None:
        self.config = config

    def run(self, output_dir: Path) -> list[Path]:
        """Prepare, solve, and summarize every configured route-path limit.

        Each iteration constructs a new ``ProblemPreparer`` with the original
        configured seed. That makes each case independent while keeping scenario
        draws comparable for routes shared by multiple K values.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        baseline_parameters = dict(self.config.data["base_parameters"])
        results: list[dict[str, Any]] = []
        result_paths: list[Path] = []

        for path_limit in ROUTE_PATH_LIMITS:
            case_config = self.config.with_recourse_path_limit(path_limit)
            prepared = ProblemPreparer(case_config).prepare()
            result = SensitivityRunner(case_config, prepared).solve_once(
                baseline_parameters,
                run_label=f"K sensitivity (K={path_limit})",
            )
            result["recourse_path_limit"] = path_limit

            result_path = output_dir / f"k_{path_limit:03d}.json"
            _write_json_atomic(result_path, result)
            results.append(result)
            result_paths.append(result_path)

        summary = {
            "analysis": "recourse_path_limit_sensitivity",
            "base_parameters": baseline_parameters,
            "route_path_limits": list(ROUTE_PATH_LIMITS),
            "results": [
                {
                    "recourse_path_limit": result["recourse_path_limit"],
                    "status": result["status"],
                    "objective_value": result["objective_value"],
                    "runtime_seconds": result["runtime_seconds"],
                }
                for result in results
            ],
        }
        summary_path = output_dir / "summary.json"
        _write_json_atomic(summary_path, summary)
        result_paths.append(summary_path)
        return result_paths


class BaseCaseComparisonRunner:
    """Compare integrated and sequential decisions using one shared sample."""

    def __init__(self, config: SensitivityConfig) -> None:
        self.config = config

    def run(self, output_dir: Path) -> list[Path]:
        config = self.config.with_path_preprocessing(
            path_limit=self.config.data["network"]["recourse_path_limit"],
            path_mode="seed_union",
        )
        prepared = ProblemPreparer(config).prepare()
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        paths = []
        for method in ("integrated", "myopic"):
            for restricted in (True, False):
                label = f"{method}_{'restricted' if restricted else 'unrestricted'}"
                case_config = config.with_recourse_path_limit(
                    config.data["network"]["recourse_path_limit"]
                )
                case_config.data["network"]["restrict_first_stage"] = restricted
                result = SensitivityRunner(case_config, prepared).solve_comparison_case(
                    method=method, run_label=label
                )
                path = output_dir / f"{label}.json"
                _write_json_atomic(path, result)
                paths.append(path)
                results[label] = result
        differences = {}
        for restriction in ("restricted", "unrestricted"):
            integrated = results[f"integrated_{restriction}"]["total_expected_cost"]
            myopic = results[f"myopic_{restriction}"]["total_expected_cost"]
            delta = myopic - integrated if integrated is not None and myopic is not None else None
            differences[restriction] = {
                "myopic_minus_integrated": delta,
                "percentage_of_integrated": 100 * delta / integrated
                if delta is not None and integrated != 0 else None,
            }
        summary = {
            "analysis": "base_case_comparison",
            "base_parameters": dict(config.data["base_parameters"]),
            "results": [
                {key: value for key, value in result.items()
                 if key not in {"decision_variables_by_shipment", "recourse_connectivity",
                                "first_stage_validation"}}
                | {"first_stage_validation": {
                    "checked": result["first_stage_validation"]["checked"],
                    "passed": result["first_stage_validation"]["passed"],
                    "violation_count": len(result["first_stage_validation"]["violations"]),
                }}
                for result in results.values()
            ],
            "cost_differences": differences,
        }
        summary_path = output_dir / "summary.json"
        _write_json_atomic(summary_path, summary)
        return paths + [summary_path]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(output_dir_explicit=False)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("sensitivity_config.json"),
        help="Configuration JSON, relative to src/ (default: sensitivity_config.json).",
    )
    parser.add_argument(
        "--prepared-problem",
        type=Path,
        default=Path("output/sensitivity/prepared_problem.pkl"),
        help=(
            "Shared prepared input. Relative paths are resolved from this script's "
            "directory (default: output/sensitivity/prepared_problem.pkl)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/sensitivity/results"),
        action=_StoreOutputDirectory,
        help=(
            "Result directory. Defaults to output/sensitivity/results for cost "
            "commands, output/k_sensitivity for K sensitivity, and "
            "output/base_case_comparison for --base-case-comparison."
        ),
    )

    command = parser.add_mutually_exclusive_group(required=True)
    command.add_argument("--prepare", action="store_true")
    command.add_argument("--run-index", type=int)
    command.add_argument(
        "--run-index-range",
        nargs=2,
        type=int,
        metavar=("START", "STOP"),
        help="Run indices in the half-open interval [START, STOP) using one model.",
    )
    command.add_argument("--print-run-count", action="store_true")
    command.add_argument("--merge", action="store_true")
    command.add_argument(
        "--base-case-comparison", action="store_true",
        help="Compare integrated and myopic baseline routing, restricted and unrestricted.",
    )
    command.add_argument(
        "--k-sensitivity",
        action="store_true",
        help=(
            f"Independently prepare and solve K={', '.join(map(str, ROUTE_PATH_LIMITS))} using only "
            "base_parameters."
        ),
    )
    command.add_argument(
        "--matched-k-sensitivity",
        action="store_true",
        help=(
            "Compare configured seed-union paths with strictly distance-ranked "
            "paths, using each shipment's unique seed-union size as its distance K."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    args.config = _relative_to_script(args.config)
    args.prepared_problem = _relative_to_script(args.prepared_problem)
    if (
        args.k_sensitivity or args.matched_k_sensitivity
    ) and not args.output_dir_explicit:
        args.output_dir = Path("output/k_sensitivity")
    args.output_dir = _relative_to_script(args.output_dir)
    if args.base_case_comparison and not args.output_dir_explicit:
        args.output_dir = SCRIPT_DIR / "output/base_case_comparison"
    config = SensitivityConfig(args.config)

    if args.base_case_comparison:
        for result_path in BaseCaseComparisonRunner(config).run(args.output_dir):
            print(result_path)
        return

    if args.k_sensitivity:
        result_paths = RouteLimitSensitivityRunner(config).run(args.output_dir)
        for result_path in result_paths:
            print(result_path)
        return

    if args.matched_k_sensitivity:
        result_paths = MatchedPathSensitivityRunner(config).run(args.output_dir)
        for result_path in result_paths:
            print(result_path)
        return

    if args.print_run_count:
        print(len(config.combinations()))
        return

    if args.prepare:
        prepared = ProblemPreparer(config).prepare()
        args.prepared_problem.parent.mkdir(parents=True, exist_ok=True)
        with args.prepared_problem.open("wb") as output_file:
            pickle.dump(prepared, output_file, protocol=pickle.HIGHEST_PROTOCOL)
        print(
            f"Prepared {len(prepared.shipments)} shipments, {len(prepared.legs)} legs, "
            f"and {prepared.num_scenarios} shared scenarios."
        )

        print(
            f"Saved prepared problem to {args.prepared_problem}"
        )

        return

    if args.merge:
        results = [
            json.loads(path.read_text())
            for path in sorted(args.output_dir.glob("run_*.json"))
        ]
        summary_path = args.output_dir / "summary.json"
        summary_path.write_text(json.dumps(results, indent=2, allow_nan=False))
        print(f"Merged {len(results)} results into {summary_path}")
        return

    with args.prepared_problem.open("rb") as input_file:
        prepared = pickle.load(input_file)

    runner = SensitivityRunner(config, prepared)
    if args.run_index_range is not None:
        start, stop = args.run_index_range
        if stop < start:
            raise ValueError("--run-index-range requires STOP >= START")
        run_count = len(config.combinations())
        if start < 0 or start >= run_count:
            raise IndexError(f"range start must be in 0..{run_count - 1}")
        # The final Slurm chunk is commonly shorter than CHUNK_SIZE.
        result_paths = runner.run_many(range(start, min(stop, run_count)), args.output_dir)
        for result_path in result_paths:
            print(result_path)
        return

    run_index = args.run_index
    if run_index is None:
        run_index = int(os.environ["SLURM_ARRAY_TASK_ID"])

    result_path = runner.run(run_index, args.output_dir)
    print(result_path)


if __name__ == "__main__":
    main()
