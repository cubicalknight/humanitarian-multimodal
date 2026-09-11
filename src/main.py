"""Run a Slurm-array cost sensitivity analysis for ``stoc_optimod``.

Workflow:
1. Run ``--prepare`` once to build the network and a shared scenario sample.
2. Submit one Slurm array task per cost-parameter combination.
3. Run ``--merge`` after the array is complete to create one summary JSON file.

Each array task reads the prepared problem and writes only its own result file.
This avoids repeated data preparation and any concurrent writes to shared results.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import pickle
import sys
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
    Shipment,
    StochasticOptimizationParameters,
    TwoStageSolver,
    UncertaintyRealization,
)

COST_FIELDS = (
    "cost_flight",
    "cost_ground",
    "cost_penalty_incompatibility",
    "cost_reassignment",
)

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "num_scenarios": 100,
    "shipment_limit": None,
    "base_parameters": {
        "cost_flight": 4.0,
        "cost_ground": 2.0,
        "cost_penalty_incompatibility": 5.0,
        "cost_reassignment": None,
    },
    "parameter_values": {
        "cost_flight": [4.0],
        "cost_ground": [2.0],
        "cost_penalty_incompatibility": [5.0],
        "cost_reassignment": [None],
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


def _merge_defaults(given: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge a user config with the supported defaults."""
    merged = dict(defaults)
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
        unknown_fields = set(self.data["parameter_values"]) - set(COST_FIELDS)
        if unknown_fields:
            raise ValueError(f"Unknown cost fields: {sorted(unknown_fields)}")

        for field in COST_FIELDS:
            values = self.data["parameter_values"].get(
                field,
                [self.data["base_parameters"][field]],
            )
            if not isinstance(values, list) or not values:
                raise ValueError(f"parameter_values.{field} must be a non-empty list.")
            if field != "cost_reassignment" and any(
                value is None or float(value) < 0 for value in values
            ):
                raise ValueError(f"parameter_values.{field} must contain non-negative values.")

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


class ProblemPreparer:
    """Build the original ``stoc_optimod.__main__`` input once."""

    MILES_PER_METER = 0.000621371
    EARTH_RADIUS_MILES = 3958.8

    def __init__(self, config: SensitivityConfig) -> None:
        self.config = config.data
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

        scenarios = self._build_scenarios(legs, t100_processor)
        return PreparedProblem(
            shipments=shipments,
            legs=legs,
            scenarios=scenarios,
            seed=int(self.config["seed"]),
            num_scenarios=int(self.config["num_scenarios"]),
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
    ) -> dict[tuple[str, str, str], UncertaintyRealization]:
        """Create one common random-number scenario set for every cost point."""
        npz = np.load(Path(__file__).parent / "psi_bar.npz")
        psi_bar = npz["psi_bar"]
        n_design = int(npz["n_design"])
        psi_alpha, psi_beta = psi_bar[:n_design], psi_bar[n_design:]
        scenario_count = int(self.config["num_scenarios"])

        scenarios: dict[tuple[str, str, str], UncertaintyRealization] = {}
        for route, leg in legs.items():
            if route[2] != "air":
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
    """Solve one parameter combination and serialize its compact JSON result."""

    def __init__(self, config: SensitivityConfig, prepared: PreparedProblem) -> None:
        self.config = config
        self.prepared = prepared

    def run(self, run_index: int, output_dir: Path) -> Path:
        combinations = self.config.combinations()
        if not 0 <= run_index < len(combinations):
            raise IndexError(f"run index must be in 0..{len(combinations) - 1}")

        parameter_values = dict(self.config.data["base_parameters"]) | combinations[run_index]
        solver_options = self.config.data["solver"]
        solver = TwoStageSolver(
            shipments=self.prepared.shipments,
            legs=self.prepared.legs,
            params=StochasticOptimizationParameters(**parameter_values),
            solver_quiet=bool(solver_options["quiet"]),
        )

        model, first_stage, first_stage_cost = solver.stage_one_setup(
            gp.Model(f"sensitivity_{run_index}")
        )
        model, keep, reassign, recourse_cost = solver.stage_two_setup(
            model,
            first_stage,
            range(self.prepared.num_scenarios),
            self.prepared.scenarios,
        )
        self._configure_gurobi(model, solver_options)
        model.setObjective(
            first_stage_cost + recourse_cost / self.prepared.num_scenarios,
            gp.GRB.MINIMIZE,
        )
        model.optimize()

        result = self._build_result(
            run_index,
            parameter_values,
            model,
            solver,
            first_stage,
            keep,
            reassign,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / f"run_{run_index:06d}.json"
        temporary_path = result_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(result, indent=2, allow_nan=False))
        temporary_path.replace(result_path)
        return result_path

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
        run_index: int,
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

        return {
            "run_index": run_index,
            "parameters": parameters,
            "seed": self.prepared.seed,
            "num_scenarios": self.prepared.num_scenarios,
            "status": int(model.Status),
            "objective_value": float(model.ObjVal) if has_solution else None,
            "runtime_seconds": float(model.Runtime),
            "decision_variables_by_shipment": decisions,
        }

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
            for route in solver.R:
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
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
    )

    command = parser.add_mutually_exclusive_group(required=True)
    command.add_argument("--prepare", action="store_true")
    command.add_argument("--run-index", type=int)
    command.add_argument("--print-run-count", action="store_true")
    command.add_argument("--merge", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    args.config = _relative_to_script(args.config)
    args.prepared_problem = _relative_to_script(args.prepared_problem)
    args.output_dir = _relative_to_script(args.output_dir)
    config = SensitivityConfig(args.config)

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

    run_index = args.run_index
    if run_index is None:
        run_index = int(os.environ["SLURM_ARRAY_TASK_ID"])

    result_path = SensitivityRunner(config, prepared).run(run_index, args.output_dir)
    print(result_path)


if __name__ == "__main__":
    main()
