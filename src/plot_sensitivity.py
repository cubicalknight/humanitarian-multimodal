"""Plot primary cost sensitivity as heatmaps faceted by incompatibility penalty.

Run from the project root, for example::

    poetry run python src/plot_sensitivity.py --input-dir run_outputs/sensitivity/results

Final and reassigned links are distinct active links, averaged equally over all
shipment--scenario pairs. Sparse omitted scenarios contribute zero; runs without
an incumbent and missing parameter combinations remain unavailable.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PARAMETERS = ("cost_ground", "cost_flight", "cost_penalty_incompatibility")
LINK_METRICS = (
    "n_ground_legs",
    "n_air_legs",
    "n_reassigned_ground_legs",
    "n_reassigned_air_legs",
)
METRICS = {
    "objective_value": ("Objective", "Objective value"),
    "n_ground_legs": ("Final ground links", "Mean links per shipment–scenario"),
    "n_air_legs": ("Final air links", "Mean links per shipment–scenario"),
    "n_reassigned_ground_legs": (
        "Reassigned ground links", "Mean links per shipment–scenario"
    ),
    "n_reassigned_air_legs": (
        "Reassigned air links", "Mean links per shipment–scenario"
    ),
}


def load_results(input_dir: Path) -> list[dict[str, Any]]:
    """Read detailed run files, ignoring any summary file."""
    results = []
    for path in sorted(input_dir.glob("run_*.json")):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            _validate_result(result)
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f"Invalid result {path}: {error}") from error
        results.append(result)
    if not results:
        raise FileNotFoundError(f"No run_*.json result files found in {input_dir}.")
    return results


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number.") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number.")
    return number


def _validate_result(result: Mapping[str, Any]) -> None:
    if not isinstance(result, Mapping):
        raise ValueError("Result must be an object.")
    required = {
        "parameters", "objective_value", "status", "num_scenarios",
        "decision_variables_by_shipment",
    }
    missing = required - result.keys()
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")
    parameters = result["parameters"]
    if not isinstance(parameters, Mapping):
        raise ValueError("parameters must be an object.")
    for field in PARAMETERS:
        if field not in parameters:
            raise ValueError(f"Missing parameter {field}.")
        if _number(parameters[field], field) < 0:
            raise ValueError(f"{field} must be nonnegative.")
    count = result["num_scenarios"]
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("num_scenarios must be a positive integer.")
    shipments = result["decision_variables_by_shipment"]
    if not isinstance(shipments, Mapping) or not shipments:
        raise ValueError("decision_variables_by_shipment must be a nonempty object.")
    if result["objective_value"] is not None:
        _number(result["objective_value"], "objective_value")
    scenario_ids = {str(i) for i in range(count)}
    for shipment, decisions in shipments.items():
        if not isinstance(decisions, Mapping):
            raise ValueError(f"Shipment {shipment} decisions must be an object.")
        if not isinstance(decisions.get("recourse"), Mapping):
            raise ValueError(f"Shipment {shipment} must contain a recourse object.")
        for scenario, legs in decisions["recourse"].items():
            if scenario not in scenario_ids:
                raise ValueError(f"Shipment {shipment}: invalid scenario {scenario}.")
            if not isinstance(legs, list):
                raise ValueError(f"Shipment {shipment}: scenario legs must be a list.")
            for leg in legs:
                required_leg_fields = {"origin", "destination", "mode", "keep", "reassign"}
                if not isinstance(leg, Mapping) or not required_leg_fields <= leg.keys():
                    raise ValueError(f"Shipment {shipment}: malformed recourse leg.")
                if leg["mode"] not in ("ground", "air"):
                    raise ValueError(f"Shipment {shipment}: unknown mode {leg['mode']}.")
                for field in ("keep", "reassign"):
                    _number(leg[field], field)


def _count_links(
    legs: Iterable[Mapping[str, Any]], active_threshold: float
) -> dict[str, int]:
    """Count distinct final and reassigned links in one shipment's scenario.

    A fractional positive flow still uses one link. A link can be both final
    and reassigned, but repeated entries for the same link count only once.
    """
    final_links = set()
    reassigned_links = set()
    for leg in legs:
        link = (str(leg["origin"]), str(leg["destination"]), leg["mode"])
        kept_flow = float(leg["keep"])
        reassigned_flow = float(leg["reassign"])
        if kept_flow + reassigned_flow > active_threshold:
            final_links.add(link)
        if reassigned_flow > active_threshold:
            reassigned_links.add(link)

    return {
        "n_ground_legs": sum(mode == "ground" for _, _, mode in final_links),
        "n_air_legs": sum(mode == "air" for _, _, mode in final_links),
        "n_reassigned_ground_legs": sum(
            mode == "ground" for _, _, mode in reassigned_links
        ),
        "n_reassigned_air_legs": sum(
            mode == "air" for _, _, mode in reassigned_links
        ),
    }


def build_tables(
    results: Iterable[Mapping[str, Any]], *, active_threshold: float = 1e-6
) -> dict[str, pd.DataFrame]:
    """Return per-run means and the underlying shipment--scenario counts."""
    threshold = _number(active_threshold, "active_threshold")
    if threshold < 0:
        raise ValueError("active_threshold must be nonnegative.")
    run_rows = []
    scenario_rows = []
    seen_combinations = set()
    for result in results:
        _validate_result(result)
        parameters = {field: float(result["parameters"][field]) for field in PARAMETERS}
        combination = tuple(parameters.values())
        if combination in seen_combinations:
            raise ValueError(f"Duplicate parameter combination: {parameters}")
        seen_combinations.add(combination)
        objective = result["objective_value"]
        row = {
            **parameters,
            "run_index": result.get("run_index"),
            "status": result["status"],
            "objective_value": float(objective) if objective is not None else float("nan"),
        }
        # No incumbent means unavailable metrics, not zero-link routes.
        row.update(dict.fromkeys(LINK_METRICS, float("nan")))
        if objective is not None:
            totals = dict.fromkeys(LINK_METRICS, 0)
            pair_count = 0
            for shipment, decisions in result["decision_variables_by_shipment"].items():
                for scenario in range(result["num_scenarios"]):
                    # The JSON omits empty scenarios; include their zeros in the mean.
                    legs = decisions["recourse"].get(str(scenario), [])
                    counts = _count_links(legs, threshold)
                    scenario_rows.append({
                        **parameters,
                        "shipment_id": str(shipment),
                        "scenario": scenario,
                        **counts,
                    })
                    pair_count += 1
                    for column in LINK_METRICS:
                        totals[column] += counts[column]
            for column in LINK_METRICS:
                row[column] = totals[column] / pair_count
        run_rows.append(row)
    if not run_rows:
        raise ValueError("At least one sensitivity result is required.")
    return {
        "runs": pd.DataFrame(run_rows).sort_values(list(PARAMETERS)).reset_index(drop=True),
        "recourse_scenarios": pd.DataFrame(
            scenario_rows,
            columns=[*PARAMETERS, "shipment_id", "scenario", *LINK_METRICS],
        ),
    }


def _draw_heatmap(
    axis: plt.Axes,
    matrix: pd.DataFrame,
    *,
    penalty: float,
    metric: str,
    color_limits: tuple[float, float],
    color_axis: plt.Axes | None,
    color_label: str,
) -> None:
    """Draw one penalty panel; gray N/A cells distinguish missing data from zero."""
    lower, upper = color_limits
    axis.set_facecolor("#e6e6e6")
    sns.heatmap(
        matrix,
        mask=matrix.isna(),
        annot=True,
        annot_kws={"fontsize": 8 if metric == "objective_value" else 10},
        fmt=".4g" if metric == "objective_value" else ".2f",
        cmap="viridis",
        vmin=lower,
        vmax=upper,
        linewidths=0.5,
        ax=axis,
        cbar=color_axis is not None,
        cbar_ax=color_axis,
        cbar_kws={"label": color_label},
    )
    for row in range(len(matrix.index)):
        for column in range(len(matrix.columns)):
            if pd.isna(matrix.iloc[row, column]):
                axis.text(
                    column + 0.5, row + 0.5, "N/A",
                    ha="center", va="center", color="#555555",
                )
    axis.set(
        title=f"Incompatibility penalty = {penalty:g}",
        xlabel="Ground cost",
        ylabel="Air cost",
    )
    axis.set_xticklabels(
        [f"{value:.6g}" for value in matrix.columns], rotation=45, ha="right"
    )
    axis.set_yticklabels([f"{value:.6g}" for value in matrix.index], rotation=0)


def plot_tables(tables: Mapping[str, pd.DataFrame], output_dir: Path) -> None:
    """Write five annotated PDFs with comparable colors across penalty panels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white", context="notebook")
    runs = tables["runs"]
    ground_costs = sorted(runs["cost_ground"].unique())
    air_costs = sorted(runs["cost_flight"].unique())
    penalties = sorted(runs["cost_penalty_incompatibility"].unique())
    rows = min(3, len(penalties))
    columns = math.ceil(len(penalties) / rows)
    for metric, (title, label) in METRICS.items():
        figure, axes = plt.subplots(
            rows, columns, figsize=(4.5 * columns + 1, 4 * rows), squeeze=False
        )
        figure.subplots_adjust(
            left=0.1, right=0.85, bottom=0.23 if rows == 1 else 0.18,
            top=0.87, wspace=0.45, hspace=0.65,
        )
        color_axis = figure.add_axes((0.89, 0.22, 0.018, 0.55))

        # Use the same scale for every penalty panel of this metric.
        valid = runs[metric].dropna()
        if valid.empty:
            lower, upper = 0.0, 1.0
        else:
            lower, upper = float(valid.min()), float(valid.max())
        if lower == upper:
            upper = lower + max(abs(lower) * 0.01, 1e-6)

        for index, (axis, penalty) in enumerate(zip(axes.flat, penalties)):
            penalty_runs = runs.loc[runs["cost_penalty_incompatibility"] == penalty]
            matrix = penalty_runs.pivot(
                index="cost_flight", columns="cost_ground", values=metric
            )
            matrix = matrix.reindex(index=air_costs, columns=ground_costs)
            _draw_heatmap(
                axis, matrix, penalty=penalty, metric=metric,
                color_limits=(lower, upper),
                color_axis=color_axis if index == 0 else None,
                color_label=label,
            )
        for axis in list(axes.flat)[len(penalties):]:
            axis.set_visible(False)
        figure.suptitle(title)
        figure.savefig(output_dir / f"sensitivity_{metric}.pdf", bbox_inches="tight")
        plt.close(figure)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=Path(__file__).parent / "output" / "sensitivity" / "results_500",
        help="Directory containing detailed run_*.json files.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent.parent / "output" / "figures" / "sensitivity" / "results_500",
        help="Directory for the five PDF heatmaps.",
    )
    parser.add_argument(
        "--active-threshold", type=float, default=1e-6,
        help="Minimum keep + reassign flow for final links, or reassign flow for reassigned links.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    results = load_results(args.input_dir)
    tables = build_tables(results, active_threshold=args.active_threshold)
    plot_tables(tables, args.output_dir)
    print(f"Processed {len(results)} runs and wrote five PDF figures to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
