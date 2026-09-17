"""Plot feasible-route-limit (K) sensitivity results.

The script reads matched cost/weight and distance-only results, falling back to
``k_*.json`` files when no matched results exist. It plots the objective and final
(post-recourse) ground and air links per shipment. The link figures show the
mean over all shipment--scenario final routings, with min--max bands.
Each objective and metric is saved as a separate figure.

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
from matplotlib.ticker import MaxNLocator
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
    matched = sorted(input_dir.glob("cost_weighted_k_*.json")) + sorted(
        input_dir.glob("distance_only_matched_k_*.json")
    )
    for path in matched or sorted(input_dir.glob("k_*.json")):
        with path.open(encoding="utf-8") as stream:
            result = json.load(stream)
        k = result.get("cost_weighted_path_limit") or result.get("recourse_path_limit")
        if k is None:
            raise ValueError(f"{path} has no 'recourse_path_limit'.")
        result["recourse_path_limit"] = int(k)
        result["case"] = (
            "Distance only" if result.get("feasible_path_mode") == "distance_only"
            else "Cost/weight"
        )
        if "decision_variables_by_shipment" not in result:
            raise ValueError(f"{path} has no shipment decisions.")
        results.append(result)

    if not results:
        raise FileNotFoundError(f"No k_*.json result files found in {input_dir}.")

    limits = [(result["case"], int(result["recourse_path_limit"])) for result in results]
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
    
    if any("case" in result for result in result_list):
        grouped = []
        for case in dict.fromkeys(result["case"] for result in result_list):
            case_results = [
                {key: value for key, value in result.items() if key != "case"}
                for result in result_list if result["case"] == case
            ]
            grouped.append({
                key: table.assign(case=case)
                for key, table in build_tables(
                    case_results, reference_k=reference_k,
                    active_threshold=active_threshold,
                ).items()
            })
        return {key: pd.concat([group[key] for group in grouped], ignore_index=True)
                for key in grouped[0]}

    limits = sorted(int(result["recourse_path_limit"]) for result in result_list)
    reference_k = max(limits) if reference_k is None else reference_k
    
    if reference_k not in limits:
        raise ValueError(f"Reference K={reference_k} is not among {limits}.")

    run_rows: list[dict[str, float | int]] = []
    recourse_rows: list[dict[str, Any]] = []
    first_routes: dict[tuple[int, str], set[LegKey]] = {}
    final_routes: dict[tuple[int, str, int], set[LegKey]] = {}

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
                final_routes[k, shipment_id, int(scenario)] = active_routes
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

    final_similarity = pd.DataFrame([
        {
            "K": k, "shipment_id": shipment_id, "scenario": scenario,
            "reference_K": reference_k,
            "jaccard_to_reference": jaccard(
                routes, final_routes[reference_k, shipment_id, scenario]
            ),
        }
        for (k, shipment_id, scenario), routes in sorted(final_routes.items())
        if (reference_k, shipment_id, scenario) in final_routes
    ])

    return {
        "runs": runs,
        "first_stage_similarity": first_similarity,
        "final_route_similarity": final_similarity,
        "recourse_scenarios": recourse,
    }


def _save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_tables(tables: Mapping[str, pd.DataFrame], output_dir: Path) -> None:
    """Write separate Seaborn figures, with consistent colors for each case."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    palette = {"Cost/weight": "#1f77b4", "Distance only": "#d62728"}
    tables = {
        key: table if "case" in table else table.assign(case="Cost/weight")
        for key, table in tables.items()
    }
    runs = tables["runs"]
    xlabel = "Seed path limit K (distance counts matched per shipment)"

    figure = plt.figure(figsize=(6.5, 4.5))
    axis = figure.add_subplot()
    sns.lineplot(
        data=runs, x="K", y="objective_value", hue="case", palette=palette,
        style="case", dashes={"Cost/weight": "", "Distance only": (4, 2)},
        marker="o", errorbar=None, ax=axis,
    )
    missing_notes = []
    for case, data in runs.groupby("case", sort=False):
        missing = data.loc[data["objective_value"].isna(), "K"].tolist()
        if missing:
            missing_notes.append(f"{case}: no solution at K={', '.join(map(str, missing))}")
    if missing_notes:
        axis.text(0.02, 0.98, "\n".join(missing_notes),
                  transform=axis.transAxes, va="top", fontsize=9)
    axis.set(title="Objective", xlabel=xlabel, ylabel="Objective value")
    axis.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    axis.legend(title="Case", frameon=False)
    figure.tight_layout()
    _save_figure(figure, output_dir, "objective")

    for table_key, column, title, ylabel, stem in (
        ("recourse_scenarios", "n_ground_legs", "Final ground links per shipment",
         "Average ground links", "ground_links"),
        ("recourse_scenarios", "n_air_legs", "Final air links per shipment",
         "Average air links", "air_links"),
        ("first_stage_similarity", "jaccard_to_reference", "First-stage route similarity",
         "Jaccard similarity to largest K in each case", "first_stage_jaccard"),
        ("final_route_similarity", "jaccard_to_reference", "Final route similarity (keep + reassign)",
         "Jaccard similarity to largest K in each case", "final_route_jaccard"),
    ):
        data = tables[table_key]
        if data.empty:
            continue
        figure = plt.figure(figsize=(6.5, 4.5))
        axis = figure.add_subplot()
        line_style = (
            {"style": "case", "dashes": {"Cost/weight": "", "Distance only": (4, 2)}}
            if table_key != "first_stage_similarity" else {}
        )
        sns.lineplot(
            data=data, x="K", y=column, hue="case", palette=palette,
            estimator="mean", errorbar=("pi", 100), err_style="band",
            marker="o", ax=axis, **line_style,
        )
        if table_key in ("first_stage_similarity", "final_route_similarity"):
            references = data.groupby("case")["reference_K"].first()
            ylabel = "Jaccard similarity"
            title += "\n" + "; ".join(f"{case}: reference K={k}" for case, k in references.items())
            axis.set_ylim(-0.03, 1.03)
        else:
            axis.set_ylim(bottom=0)
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        axis.legend(title="Case", frameon=False)
        figure.tight_layout()
        _save_figure(figure, output_dir, stem)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output/k_sensitivity",
        help="Directory containing detailed matched results or legacy k_*.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output/figures/k_sensitivity",
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
        f"Processed {len(results)} case/K results and wrote figures to "
        f"{args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
