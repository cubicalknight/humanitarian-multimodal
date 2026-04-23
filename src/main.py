# %%
from __future__ import annotations

import argparse
import sys
# import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import pickle as pkl

from data_processing import DataProcessing
from nce_model import NoiseGeneration, TrainNCE
from reproduce_with_stochastic_airlift import configure_determinism, train_torch_mnlr
from stochastic_airlift import MultinomialLogitAircraftModel
from stoc_optimod import (
    FlightOption,
    Shipment,
    StochasticOptimizationParameters,
    TwoStageSolver,
    UncertaintyRealization,
)
from unified_data_loader import build_shared_split


CONFIG_INDEX = {
    "Narrowbody": 0,
    "Widebody": 1,
    "Freighter": 2,
}


def _config_index(config_name: str) -> int:
    if config_name not in CONFIG_INDEX:
        raise ValueError(
            f"Unexpected aircraft configuration '{config_name}'. "
            "Expected one of ['Narrowbody', 'Widebody', 'Freighter']."
        )
    return CONFIG_INDEX[config_name]


def _build_parser() -> argparse.ArgumentParser:
    # make compatible with notebook by defaulting to no arguments, but allow overrides for CLI use
    parser = argparse.ArgumentParser(
        description="Train NCE + aircraft models on train split and run SAA on test split."
    )
    parser.add_argument("--data", type=Path, default=None, help="Optional data source path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train ratio for shared split.")
    parser.add_argument("--nce-trials", type=int, default=30, help="Optuna trials for NCE training.")
    parser.add_argument("--n-scenarios", type=int, default=8, help="Number of SAA scenarios.")
    parser.add_argument(
        "--config-verbose",
        action="store_true",
        help="Print progress during config prediction model training.",
    )
    parser.add_argument(
        "--config-log-every",
        type=int,
        default=25,
        help="Epoch logging interval for config prediction training when --config-verbose is set.",
    )
    parser.add_argument(
        "--solver-quiet",
        action="store_true",
        help="Disable Gurobi solver output.",
    )
    parser.add_argument("--retrain", action="store_true", help="Whether to retrain models instead of loading from disk.")

    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    if argv is None:
        if "ipykernel" in sys.modules:
            argv = []
        else:
            argv = sys.argv[1:]
    return parser.parse_args(argv)


def _load_split_frames(
    data_path: Path | None,
    train_indices: torch.Tensor,
    test_indices: torch.Tensor,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    data_handler = DataProcessing()
    if data_path is not None:
        data_handler.excel_path = data_path

    df = data_handler.load_shipping_data(data_handler.excel_path)
    df = data_handler._geolocate_nodes(df)
    df = data_handler._calculate_distance(df)

    train_frame = df[train_indices.cpu().numpy().tolist()]
    test_frame = df[test_indices.cpu().numpy().tolist()]
    return train_frame, test_frame


def _explode_airline_names(frame: pl.DataFrame, airline_col: str = "Airline") -> pl.DataFrame:
    if airline_col not in frame.columns:
        raise ValueError(f"Expected '{airline_col}' to be present in the data frame.")

    return (
        frame
        .select(pl.all())
        .with_columns(
            pl.col(airline_col)
            .cast(pl.Utf8)
            .str.split(",")
            .alias("airline_name")
        )
        .explode("airline_name")
        .with_columns(pl.col("airline_name").str.strip_chars())
        .filter(pl.col("airline_name").is_not_null() & (pl.col("airline_name") != ""))
    )


def _build_airline_price_profile(train_frame: pl.DataFrame) -> tuple[dict[str, float], float, int]:
    weight_column_candidates = ["shipment_weight_kg", "AW (kg)"]
    weight_column = next((c for c in weight_column_candidates if c in train_frame.columns), None)

    required_columns = {"Airline", "Commercial Cost for Long-Haul", "distance"}
    missing_columns = sorted(required_columns.difference(train_frame.columns))
    if missing_columns:
        raise ValueError(f"Cannot build airline price profile; missing columns: {missing_columns}")
    if weight_column is None:
        raise ValueError(
            "Cannot build airline price profile; missing shipment weight column. "
            f"Expected one of {weight_column_candidates}."
        )

    priced_rows = (
        _explode_airline_names(
            train_frame.select(["Airline", "Commercial Cost for Long-Haul", "distance", weight_column]),
            airline_col="Airline",
        )
        .with_columns([
            pl.col("Commercial Cost for Long-Haul")
            .cast(pl.Utf8)
            .str.replace_all(",", "")
            .cast(pl.Float64, strict=False)
            .alias("long_haul_cost"),
            pl.col("distance").cast(pl.Utf8).str.replace_all(",", "").cast(pl.Float64, strict=False).alias("distance_km"),
            pl.col(weight_column).cast(pl.Utf8).str.replace_all(",", "").cast(pl.Float64, strict=False).alias("shipment_weight_kg"),
        ])
        .with_columns((pl.col("distance_km") * 0.621371).alias("distance_miles"))
        .filter(
            pl.col("long_haul_cost").is_not_null()
            & pl.col("distance_miles").is_not_null()
            & pl.col("shipment_weight_kg").is_not_null()
            & (pl.col("long_haul_cost") > 0)
            & (pl.col("distance_miles") > 0)
            & (pl.col("shipment_weight_kg") > 0)
        )
        .with_columns((pl.col("long_haul_cost") / (pl.col("distance_miles") * pl.col("shipment_weight_kg"))).alias("price_per_mile_kg"))
    )

    # print(priced_rows.head(5))
    # sys.exit(0)

    if priced_rows.is_empty():
        raise ValueError("No valid airline pricing rows remained after cleaning training data.")

    airline_mean_price_per_mile = (
        priced_rows
        .group_by("airline_name")
        .agg(pl.col("price_per_mile_kg").mean().alias("mean_price_per_mile"))
        .sort("airline_name")
    )

    # print(airline_mean_price_per_mile.head(5))
    # sys.exit(0)

    airline_price_map = {
        str(row[0]): float(row[1])
        for row in airline_mean_price_per_mile.select(["airline_name", "mean_price_per_mile"]).iter_rows()
    }
    if not airline_price_map:
        raise ValueError("No airline price-per-mile statistics could be computed.")

    global_mean_price_per_mile = float(priced_rows.select(pl.col("price_per_mile_kg").mean()).item())
    if not np.isfinite(global_mean_price_per_mile) or global_mean_price_per_mile <= 0:
        raise ValueError(f"Invalid global mean price-per-mile computed: {global_mean_price_per_mile}")

    return airline_price_map, global_mean_price_per_mile, int(priced_rows.height)


def _build_shipments_and_features(
    split_features: torch.Tensor,
    test_indices: torch.Tensor,
    test_frame: pl.DataFrame,
) -> tuple[list[Shipment], dict[str, np.ndarray], dict[str, float], dict[str, float], dict[str, int]]:
    shipments: list[Shipment] = []
    feature_by_shipment: dict[str, np.ndarray] = {}
    weight_by_shipment: dict[str, float] = {}
    pallets_by_shipment: dict[str, float] = {}
    observed_config_by_shipment: dict[str, int] = {}

    test_features_np = split_features[test_indices].detach().cpu().numpy().astype(np.float32)
    rows = test_frame.iter_rows(named=True)

    for idx, (row, feat) in enumerate(zip(rows, test_features_np)):

        shipment_id = f"S{idx:04d}"
        weight = float(row.get("AW (kg)", 0.0) or 0.0)
        pallets = float(row.get("Pallets", 0.0) or 0.0)
        origin = str(row.get("Origin", ""))
        destination = str(row.get("Destination", ""))
        observed_config = str(row.get("Aircraft Type", ""))

        shipments.append(
            Shipment(
                shipment_id=shipment_id,
                weight_kg=weight,
                origin=origin,
                destination=destination,
                pallets=pallets,
                equivalent_cost=max(0.0, 0.15 * weight),
                commodity="humanitarian",
            )
        )
        feature_by_shipment[shipment_id] = feat
        weight_by_shipment[shipment_id] = weight
        pallets_by_shipment[shipment_id] = pallets
        observed_config_by_shipment[shipment_id] = _config_index(observed_config)

    return shipments, feature_by_shipment, weight_by_shipment, pallets_by_shipment, observed_config_by_shipment


def _build_flights(
    train_frame: pl.DataFrame,
    test_frame: pl.DataFrame,
    rng: np.random.Generator,
    airline_price_per_mile: dict[str, float],
    global_mean_price_per_mile: float,
    airlines_per_route: int = 3,
) -> tuple[list[FlightOption], int]:
    routes = (
        test_frame.select(["Origin", "Destination", "distance"]).drop_nulls().unique().sort(["Origin", "Destination"])
    )

    global_airlines = _explode_airline_names(train_frame, airline_col="Airline").select("airline_name").unique().to_series().to_list()
    global_airlines = [str(a) for a in global_airlines]
    if not global_airlines:
        raise ValueError("No airlines were found in the training split to bootstrap test flights.")

    flights: list[FlightOption] = []
    # NOTE use for testing
    fallback_cost_count = 0

    for route_idx, route in enumerate(routes.iter_rows(named=True)):

        origin = str(route["Origin"])
        destination = str(route["Destination"])
        distance_km = float(route["distance"])
        distance_miles = distance_km * 0.621371

        route_airlines = (
            _explode_airline_names(
                train_frame.filter((pl.col("Origin") == origin) & (pl.col("Destination") == destination)),
                airline_col="Airline",
            )
            .select("airline_name")
            .unique()
            .to_series()
            .to_list()
        )
        route_airlines = [str(a) for a in route_airlines]

        # NOTE new bandaid fix, sometimes have 1 airline op on rt so need to shift to global
        pool = route_airlines if (route_airlines and len(route_airlines) != 1) else global_airlines

        n_sample = min(airlines_per_route, len(pool))
        if n_sample == 1:
            raise Exception(
                f"Only one airline '{pool[0]}' found for route {origin} -> {destination}. Consider reducing airlines_per_route or ensuring more airline diversity in the training split."
            )
        sampled_airlines = rng.choice(pool, size=n_sample, replace=False).tolist()

        for airline_id in sampled_airlines:
            price_per_mile = airline_price_per_mile.get(airline_id, global_mean_price_per_mile)
            if airline_id not in airline_price_per_mile:
                fallback_cost_count += 1
            if not np.isfinite(price_per_mile) or price_per_mile <= 0:
                raise ValueError(
                    f"Invalid price-per-mile for airline '{airline_id}': {price_per_mile}."
                )
            flights.append(
                FlightOption(
                    route_id=f"R{route_idx:03d}",
                    airline_id=airline_id,
                    origin=origin,
                    destination=destination,
                    distance_miles=distance_miles,
                    cost_flight=price_per_mile * distance_miles,
                )
            )

    return flights, fallback_cost_count


def _train_predictive_models(args):
    configure_determinism(args.seed)
    rng = np.random.default_rng(args.seed)

    split = build_shared_split(data_path=args.data, train_ratio=args.train_ratio, seed=args.seed)

    # Train NCE acceptance predictor on the training set.
    print("Training NCE acceptance predictor...")
    nce_trainer = TrainNCE(split.train_features, split.test_features)
    nce_study = nce_trainer.run_optimization(n_trials=args.nce_trials)
    nce_model = nce_trainer.train_final_model(nce_study.best_trial.params)
    nce_network: Any = nce_model.model
    noise_gen = NoiseGeneration(split.train_features, inflation=1.5)

    # Train aircraft configuration predictor on the training set.
    print("Training aircraft configuration predictor...")
    torch_mnlr, _, _ = train_torch_mnlr(
        split,
        verbose=args.config_verbose,
        log_every=args.config_log_every,
    )
    aircraft_compat_model = MultinomialLogitAircraftModel(
        coef_=torch_mnlr.linear.weight.detach().cpu().numpy(),
        intercept_=torch_mnlr.linear.bias.detach().cpu().numpy(),
        classes_=split.label_names,
    )

    # Define training and testing frames
    train_frame, test_frame = _load_split_frames(args.data, split.train_indices, split.test_indices)

    # Build airline price profile from training data
    airline_price_per_mile, global_mean_price_per_mile, pricing_rows = _build_airline_price_profile(train_frame)

    # Build shipments from test set
    shipments, feature_by_shipment, weight_by_shipment, pallets_by_shipment, observed_config_by_shipment = _build_shipments_and_features(
        split.features,
        split.test_indices,
        test_frame,
    )

    # Build flight options from test set and airline price profile
    # NOTE: this will be replaced by T100 flight and multimodal options
    flights, fallback_cost_count = _build_flights(
            train_frame,
            test_frame,
            rng=rng,
            airline_price_per_mile=airline_price_per_mile,
            global_mean_price_per_mile=global_mean_price_per_mile,
        )
        
    # 
    shipment_ids = [shipment.shipment_id for shipment in shipments]
    shipment_features = np.stack([feature_by_shipment[shipment_id] for shipment_id in shipment_ids], axis=0)

    # NCE evaluation
    nce_device = next(nce_network.parameters()).device
    shipment_feature_tensor = torch.as_tensor(shipment_features, dtype=torch.float32, device=nce_device)

    # nce_batch_start = time.perf_counter()
    with torch.no_grad():
        nce_scores = nce_network(shipment_feature_tensor).squeeze(-1)
        nce_noise_log_prob = noise_gen.log_prob(shipment_feature_tensor)
        accept_prob_tensor = torch.sigmoid(nce_scores - nce_noise_log_prob)
        accept_prob_tensor = torch.clamp(accept_prob_tensor, 1e-4, 0.9999)
    # nce_batch_seconds = time.perf_counter() - nce_batch_start
    accept_prob_by_shipment = {
        shipment_id: float(accept_prob_tensor[idx].item())
        for idx, shipment_id in enumerate(shipment_ids)
    }

    # save to pkl
    with open("predictive-elements.pkl", "wb") as f:
        pkl.dump(
            {
                "shipments": shipments,
                "flights": flights,
                # "split": split,
                "observed_config_by_shipment": observed_config_by_shipment,
                "feature_by_shipment": feature_by_shipment,
                "accept_prob_by_shipment": accept_prob_by_shipment,
                "aircraft_compat_model": aircraft_compat_model,
                "rng": rng,
            },
            f
        )

    return shipments, flights, observed_config_by_shipment, feature_by_shipment, accept_prob_by_shipment, aircraft_compat_model, rng


def main() -> tuple[Any, Any]:
    args = _parse_args()

    if args.retrain:
        print("Retraining models from scratch...")

        shipments, flights, observed_config_by_shipment, feature_by_shipment, accept_prob_by_shipment, aircraft_compat_model, rng = _train_predictive_models(args)    
       
        # print(f"Built {len(shipments)} shipments and extracted features, weights, pallets, and observed configs for each shipment.")
        
        # print(
        #     f"Flight pricing diagnostics: airlines_with_stats={len(airline_price_per_mile)}, "
        #     f"pricing_rows={pricing_rows}, global_mean_ppm={global_mean_price_per_mile:.4f}, "
        #     f"fallback_flights={fallback_cost_count}"
        # )

        # if not shipments:
        #     raise ValueError("No feasible shipments remain after enforcing OD arc restrictions.")
        # if not flights:
        #         raise ValueError("No flight options were built from the test split.")


        # Define shipment features tensor for NCE evaluation    
    else:
        with open("predictive-elements.pkl", "rb") as f:
            predictive_elements = pkl.load(f)

        shipments = predictive_elements["shipments"]
        flights = predictive_elements["flights"]
        # split = predictive_elements["split"]
        observed_config_by_shipment = predictive_elements["observed_config_by_shipment"]
        feature_by_shipment = predictive_elements["feature_by_shipment"]
        accept_prob_by_shipment = predictive_elements["accept_prob_by_shipment"]
        aircraft_compat_model = predictive_elements["aircraft_compat_model"]
        rng = predictive_elements["rng"]



    print("Building scenarios...")
    # scenario_build_start = time.perf_counter()
    scenarios: list[list[UncertaintyRealization]] = []
    for _ in range(args.n_scenarios):
        scenario: list[UncertaintyRealization] = []
        for shipment in shipments:
            observed_idx = observed_config_by_shipment[shipment.shipment_id]
            base_feat = feature_by_shipment[shipment.shipment_id]
            accept_prob = accept_prob_by_shipment[shipment.shipment_id]

            for flight in flights:
                accepted = bool(rng.binomial(1, accept_prob))

                compatibility = False
                if accepted:
                    realized_class, _, _ = aircraft_compat_model.sample(base_feat, rng=rng)
                    realized_idx = _config_index(realized_class)
                    compatibility = realized_idx >= observed_idx

                scenario.append(
                    UncertaintyRealization(
                        shipment_id=shipment.shipment_id,
                        flight_id=f"{flight.route_id}_{flight.airline_id}",
                        acceptance=accepted,
                        compatibility=compatibility,
                    )
                )
        scenarios.append(scenario)
    # scenario_build_seconds = time.perf_counter() - scenario_build_start

    print(f"Built {len(scenarios)} scenarios with acceptance and compatibility realizations for each shipment-flight pair.")
    # print(
    #     f"NCE batch evaluation took {nce_batch_seconds:.2f}s; scenario construction took {scenario_build_seconds:.2f}s"
    # )

    # \ra
    PARAMS = StochasticOptimizationParameters(
        cost_penalty_rejection=5000.0,
        cost_penalty_incompatibility=0.0,
        cost_reassignment=1200.0,
    )

    solver = TwoStageSolver(shipments, flights, PARAMS)
    print("Solving SAA problem...")

    saa_solution, mod = solver.solve_sample_average(scenarios=scenarios)
    # accepted_counts = [
    #     sum(1 for ur in scenario if ur.acceptance) for scenario in scenarios
    # ]
    # compat_counts = [
    #     sum(1 for ur in scenario if ur.compatibility) for scenario in scenarios
    # ]

    # print("=== Integrated SAA Run ===")
    # print(
    #     f"Data split: train={len(split.train_indices)}, test={len(split.test_indices)}, "
    #     f"ratio={split.train_ratio:.2f}/{1 - split.train_ratio:.2f}, seed={split.seed}"
    # )
    # print(f"Optimization entities: shipments={len(shipments)}, flights={len(flights)}, scenarios={len(scenarios)}")
    # print(f"Scenario diagnostics: avg_acceptances={np.mean(accepted_counts):.1f}, avg_compatibilities={np.mean(compat_counts):.1f}")

    # print(f"Solver status: {saa_solution.first_stage.status}")
    # print(f"First-stage objective: {saa_solution.first_stage.objective_value:,.2f}")
    # if saa_solution.second_stage is not None:
    #     print(f"Expected recourse: {saa_solution.second_stage.objective_value:,.2f}")
    # print(f"Total expected cost: {saa_solution.expected_total_cost:,.2f}")

    # print("Top first-stage assignments:")
    # shown = 0
    # for (shipment_id, flight_id), value in sorted(saa_solution.first_stage.assignments.items()):
    #     if value > 0.5:
    #         print(f"  {shipment_id} -> {flight_id}")
    #         shown += 1
    #         if shown >= 20:
    #             break

    # print("Top second stage reassignments (if any):")
    # if saa_solution.second_stage is not None:
    #     shown = 0
    #     for (shipment_id, flight_id), value in sorted(saa_solution.second_stage.reassignments.items()):
    #         if value > 0.5:
    #             print(f"  {shipment_id} -> {flight_id}")
    #             shown += 1
    #             if shown >= 20:
    #                 break

        # print("Myopic Run")
    myopic_solution, mod2 = solver.solve_myopic(scenarios=scenarios)

    # Diagnostics from exact binary decision variables in the solved models.
    incompatible_lookup = {
        (ur.shipment_id, ur.flight_id, om)
        for om, scenario in enumerate(scenarios)
        for ur in scenario
        if ur.acceptance and not ur.compatibility
    }

    def _parse_triplet_var_name(var_name: str) -> tuple[str, str, int] | None:
        start = var_name.find("[")
        end = var_name.rfind("]")
        if start == -1 or end == -1 or end <= start + 1:
            return None
        parts = [p.strip() for p in var_name[start + 1:end].split(",", 2)]
        if len(parts) != 3:
            return None
        try:
            return parts[0], parts[1], int(parts[2])
        except ValueError:
            return None

    def _totals_from_model(model) -> tuple[int, int]:
        total_reassignments = 0
        total_incompatibles = 0
        for var in model.getVars():
            if var.X <= 0.5:
                continue

            parsed = _parse_triplet_var_name(var.VarName)
            if parsed is None:
                continue
            shipment_id, flight_id, om = parsed
            key = (shipment_id, flight_id, om)

            if var.VarName.startswith("reassign["):
                total_reassignments += 1
                if key in incompatible_lookup:
                    total_incompatibles += 1
            elif var.VarName.startswith("keep["):
                if key in incompatible_lookup:
                    total_incompatibles += 1

        return total_reassignments, total_incompatibles

    saa_total_reassignments, saa_total_incompatibilities = _totals_from_model(mod)
    myopic_total_reassignments, myopic_total_incompatibilities = _totals_from_model(mod2)
    
    #     total_costs.append((saa_solution.expected_total_cost, myopic_solution.expected_total_cost))
    #     first_stage_costs.append((saa_solution.first_stage.objective_value, myopic_solution.first_stage.objective_value))
    #     second_stage_costs.append((saa_solution.second_stage.objective_value, myopic_solution.second_stage.objective_value))

    # avg_total_costs = np.mean(total_costs, axis=0)
    # avg_first_stage_costs = np.mean(first_stage_costs, axis=0)
    # avg_second_stage_costs = np.mean(second_stage_costs, axis=0)

    # print(f"Myopic solver status: {myopic_solution.first_stage.status}")
    # print(f"Myopic first-stage objective: {myopic_solution.first_stage.objective_value:,.2f}")
    # print(f"Myopic expected recourse: {myopic_solution.second_stage.objective_value:,.2f}")
    # print(f"Myopic total expected cost: {myopic_solution.expected_total_cost:,.2f}")

    # Comparing the two results
    print("\n=== Solution Comparison ===")
    print(f"SAA expected total cost: {saa_solution.expected_total_cost:,.2f}")
    print(f"Myopic total expected cost: {myopic_solution.expected_total_cost:,.2f}")
    print(f"Cost difference (Myopic - SAA): {myopic_solution.expected_total_cost - saa_solution.expected_total_cost:,.2f}")
    print(f"Percent reduction: {100.0 * (saa_solution.expected_total_cost - myopic_solution.expected_total_cost) / myopic_solution.expected_total_cost:.2f}%")
    print("First stage difference (Myopic - SAA):", (myopic_solution.first_stage.objective_value - saa_solution.first_stage.objective_value))
    print("Second stage difference (Myopic - SAA):", (myopic_solution.second_stage.objective_value - saa_solution.second_stage.objective_value))

    print("\n=== Totals By Model ===")
    print(f"SAA total reassignments: {saa_total_reassignments}")
    print(f"SAA total incompatible flights: {saa_total_incompatibilities}")
    print(f"Myopic total reassignments: {myopic_total_reassignments}")
    print(f"Myopic total incompatible flights: {myopic_total_incompatibilities}")
    return saa_solution, mod


if __name__ == "__main__":
    saa_solution, mod = main()
