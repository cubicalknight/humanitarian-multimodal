"""Plot feasible-route-limit (K) sensitivity results.

The script expects the detailed ``k_*.json`` files written by
``main.py --k-sensitivity``.  It plots the objective and the final
(post-recourse) ground and air links per shipment.  The modal panel shows the
mean over all shipment--scenario final routings, with min--max bands.

Example::

    poetry run python src/plot_k_sensitivity.py
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

LegKey = tuple[str, str, str]


def leg_key(leg: Mapping[str, Any]) -> LegKey:
    """Return the stable identity used to compare a leg across K values."""
    return str(leg["origin"]), str(leg["destination"]), str(leg["mode"])


def jaccard(left: set[LegKey], right: set[LegKey]) -> float:
    """Jaccard similarity, with two empty routes defined as identical."""
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def load_results(input_dir: Path) -> list[dict[str, Any]]:
    """Load and validate detailed K result files in ascending K order."""
    results: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("k_*.json")):
        with path.open(encoding="utf-8") as stream:
            result = json.load(stream)
        if "recourse_path_limit" not in result:
            raise ValueError(f"{path} has no 'recourse_path_limit'.")
        if "decision_variables_by_shipment" not in result:
            raise ValueError(f"{path} has no shipment decisions.")
        results.append(result)

    if not results:
        raise FileNotFoundError(f"No k_*.json result files found in {input_dir}.")

    limits = [int(result["recourse_path_limit"]) for result in results]
    if len(limits) != len(set(limits)):
        raise ValueError(f"Duplicate K values found: {limits}")
    return sorted(results, key=lambda result: int(result["recourse_path_limit"]))


def build_tables(
    results: Iterable[Mapping[str, Any]],
    *,
    reference_k: int | None = None,
    active_threshold: float = 1e-6,
) -> dict[str, pd.DataFrame]:
    """Extract the objective, final links, and first-stage Jaccard values."""
    result_list = list(results)
    if not result_list:
        raise ValueError("At least one K result is required.")

    limits = sorted(int(result["recourse_path_limit"]) for result in result_list)
    reference_k = max(limits) if reference_k is None else reference_k
    if reference_k not in limits:
        raise ValueError(f"Reference K={reference_k} is not among {limits}.")

    run_rows: list[dict[str, float | int]] = []
    recourse_rows: list[dict[str, Any]] = []
    first_routes: dict[tuple[int, str], set[LegKey]] = {}

    for result in result_list:
        k = int(result["recourse_path_limit"])
        objective = result.get("objective_value")
        run_rows.append(
            {
                "K": k,
                "objective_value": (
                    float(objective) if objective is not None else float("nan")
                ),
            }
        )

        # _build_result leaves every shipment's decision lists empty when the
        # solver has no incumbent.  Do not interpret those placeholders as
        # genuine zero-leg routes.
        if objective is None:
            continue

        shipments = result["decision_variables_by_shipment"]
        for shipment_id, decisions in shipments.items():
            shipment_id = str(shipment_id)
            selected = {
                leg_key(leg)
                for leg in decisions.get("first_stage", [])
                if float(leg.get("value", 0.0)) > 0.5
            }
            first_routes[k, shipment_id] = selected

            scenario_count = int(result.get("num_scenarios", 0))
            scenario_decisions = decisions.get("recourse", {})
            scenario_ids = range(scenario_count) if scenario_count else map(
                int, scenario_decisions.keys()
            )
            for scenario in scenario_ids:
                active_routes = {
                    leg_key(leg)
                    for leg in scenario_decisions.get(str(scenario), [])
                    if float(leg.get("keep", 0.0))
                    + float(leg.get("reassign", 0.0))
                    > active_threshold
                }
                recourse_rows.append(
                    {
                        "K": k,
                        "shipment_id": shipment_id,
                        "scenario": int(scenario),
                        "n_ground_legs": sum(
                            mode == "ground" for _, _, mode in active_routes
                        ),
                        "n_air_legs": sum(
                            mode == "air" for _, _, mode in active_routes
                        ),
                    }
                )

    runs = pd.DataFrame(run_rows).sort_values("K").reset_index(drop=True)

    first_similarity_rows: list[dict[str, Any]] = []
    reference_shipments = {
        shipment for k, shipment in first_routes if k == reference_k
    }
    for k in limits:
        current_shipments = {
            shipment for route_k, shipment in first_routes if route_k == k
        }
        for shipment_id in sorted(reference_shipments & current_shipments):
            current = first_routes[k, shipment_id]
            reference = first_routes[reference_k, shipment_id]
            first_similarity_rows.append(
                {
                    "K": k,
                    "shipment_id": shipment_id,
                    "reference_K": reference_k,
                    "jaccard_to_reference": jaccard(current, reference),
                }
            )
    first_similarity = pd.DataFrame(first_similarity_rows)
    if not first_similarity.empty:
        first_similarity = first_similarity.sort_values(
            ["K", "shipment_id"]
        ).reset_index(drop=True)

    recourse = pd.DataFrame(recourse_rows)
    if not recourse.empty:
        recourse = recourse.sort_values(
            ["K", "shipment_id", "scenario"]
        ).reset_index(drop=True)

    return {
        "runs": runs,
        "first_stage_similarity": first_similarity,
        "recourse_scenarios": recourse,
    }


def _save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_tables(tables: Mapping[str, pd.DataFrame], output_dir: Path) -> None:
    """Create PDF plots for K sensitivity and first-stage route similarity."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    colors = sns.color_palette("colorblind", n_colors=3)

    runs = tables["runs"]
    order = sorted(runs["K"].unique())
    final_routings = tables["recourse_scenarios"]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    sns.lineplot(
        data=runs,
        x="K",
        y="objective_value",
        color=colors[0],
        marker="o",
        ax=axes[0],
    )
    axes[0].set(title="Objective", xlabel="K", ylabel="Objective value")

    ground_axis = axes[1]
    air_axis = ground_axis.twinx()
    for axis, column, label, color in (
        (
            ground_axis,
            "n_ground_legs",
            "Ground",
            colors[1],
        ),
        (
            air_axis,
            "n_air_legs",
            "Air",
            colors[2],
        ),
    ):
        sns.lineplot(
            data=final_routings,
            x="K",
            y=column,
            estimator="mean",
            errorbar=("pi", 100),
            err_style="band",
            marker="o",
            color=color,
            label=label,
            ax=axis,
        )
        axis.set_ylim(-1, 3)
    ground_axis.set(
        title="Final links per shipment",
        xlabel="K",
        ylabel="Average ground links per shipment",
    )
    air_axis.set_ylabel("Average air links per shipment")
    ground_axis.legend(loc="upper left", frameon=False)
    air_axis.legend(loc="upper right", frameon=False)

    for axis in axes:
        axis.set_xticks(order)
    figure.suptitle("K-sensitivity results", y=1.02)
    figure.tight_layout()
    _save_figure(figure, output_dir, "k_sensitivity")

    similarity = tables["first_stage_similarity"]
    if similarity.empty:
        return

    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    sns.lineplot(
        data=similarity,
        x="K",
        y="jaccard_to_reference",
        estimator="mean",
        errorbar=("pi", 100),
        err_style="band",
        color=colors[0],
        marker="o",
        ax=axis,
    )
    axis.set(
        title=f"First-stage similarity to K={similarity['reference_K'].iloc[0]}",
        xlabel="K",
        ylabel="Jaccard similarity",
        ylim=(-0.03, 1.03),
    )
    axis.set_xticks(order)
    figure.tight_layout()
    _save_figure(figure, output_dir, "first_stage_jaccard")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("output/k_sensitivity"),
        help="Directory containing detailed k_*.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/figures/k_sensitivity"),
        help="Directory for PDF figures.",
    )
    parser.add_argument(
        "--reference-k",
        type=int,
        default=None,
        help="K used as the reference route; defaults to the largest available K.",
    )
    parser.add_argument(
        "--active-threshold",
        type=float,
        default=1e-6,
        help="Minimum keep + reassign flow for an active recourse leg.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    results = load_results(args.input_dir)
    tables = build_tables(
        results,
        reference_k=args.reference_k,
        active_threshold=args.active_threshold,
    )
    plot_tables(tables, args.output_dir)
    print(
        f"Processed {len(results)} K values and wrote figures to "
        f"{args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
