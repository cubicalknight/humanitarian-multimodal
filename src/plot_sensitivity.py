"""Plot cost sensitivity grids and penalty-response figures.

Run from the project root, for example::

    poetry run python src/plot_sensitivity.py --input-dir run_outputs/sensitivity/results

Final and reassigned links are distinct active links, averaged equally over all
shipment--scenario pairs. Sparse omitted scenarios contribute zero; runs without
an incumbent and missing parameter combinations remain unavailable.

The penalty-response figure instead sums reassignment variable values, matching
the expected-leg formula even for fractional recourse. Scenarios in these runs
are equally likely (p_omega = 1 / num_scenarios). The range is not a confidence
interval. Zero-penalty label changes alone do not establish route changes.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
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
EXPECTED_METRICS = (
    "expected_reassigned_air_legs", "expected_reassigned_ground_legs",
)
PENALTY_RESPONSE_METRICS = ("objective_value", *EXPECTED_METRICS)
BASELINE_AIR_COST = 0.00077
BASELINE_GROUND_COST = 0.000135
FIGURE_WIDTH_INCHES = 6.5
MIN_FIGURE_FONT_SIZE = 10
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
        print(f"Processing run {parameters} with {result['num_scenarios']} scenarios...")
        combination = tuple(parameters.values())
        print(f"Combination: {combination}")
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
        row.update(dict.fromkeys((*LINK_METRICS, *EXPECTED_METRICS), float("nan")))
        if objective is not None:
            totals = dict.fromkeys((*LINK_METRICS, *EXPECTED_METRICS), 0.0)
            pair_count = 0
            for shipment, decisions in result["decision_variables_by_shipment"].items():
                for scenario in range(result["num_scenarios"]):
                    # The JSON omits empty scenarios; include their zeros in the mean.
                    legs = decisions["recourse"].get(str(scenario), [])
                    counts = _count_links(legs, threshold)
                    # Each edge variable occurs once in the mathematical sum.
                    # Ignore duplicate serialized entries, as the grids do.
                    reassigned = {
                        (str(leg["origin"]), str(leg["destination"]), leg["mode"]):
                        float(leg["reassign"]) for leg in legs
                    }
                    for mode in ("air", "ground"):
                        counts[f"expected_reassigned_{mode}_legs"] = sum(
                            value for (_, _, leg_mode), value in reassigned.items()
                            if leg_mode == mode
                        )
                    scenario_rows.append({
                        **parameters,
                        "shipment_id": str(shipment),
                        "scenario": scenario,
                        **counts,
                    })
                    pair_count += 1
                    for column in totals:
                        totals[column] += counts[column]
            for column in totals:
                row[column] = totals[column] / pair_count
        run_rows.append(row)
    if not run_rows:
        raise ValueError("At least one sensitivity result is required.")
    return {
        "runs": pd.DataFrame(run_rows).sort_values(list(PARAMETERS)).reset_index(drop=True),
        "recourse_scenarios": pd.DataFrame(
            scenario_rows,
            columns=[*PARAMETERS, "shipment_id", "scenario", *LINK_METRICS, *EXPECTED_METRICS],
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


def build_penalty_response(
    runs: pd.DataFrame, *, baseline_air_cost: float = BASELINE_AIR_COST,
    baseline_ground_cost: float = BASELINE_GROUND_COST,
) -> pd.DataFrame:
    """Summarize available runs at each penalty; never substitute a baseline.

    Missing incumbents are excluded from bounds, and missing baseline runs stay
    NaN so lines break rather than interpolate across unavailable observations.
    Counts expose incomplete transportation-cost coverage in the exported CSV.
    """
    for name, value in (("baseline_air_cost", baseline_air_cost),
                        ("baseline_ground_cost", baseline_ground_cost)):
        if _number(value, name) < 0:
            raise ValueError(f"{name} must be nonnegative.")
    penalty = "cost_penalty_incompatibility"
    baseline = runs.loc[
        runs.cost_flight.map(lambda x: math.isclose(x, baseline_air_cost, rel_tol=1e-9, abs_tol=0))
        & runs.cost_ground.map(lambda x: math.isclose(x, baseline_ground_cost, rel_tol=1e-9, abs_tol=0))
    ].set_index(penalty)
    summary = pd.DataFrame(index=sorted(runs[penalty].unique()))
    summary.index.name = penalty
    for metric in PENALTY_RESPONSE_METRICS:
        grouped = runs.groupby(penalty)[metric]
        summary[f"{metric}_baseline"] = baseline[metric].reindex(summary.index)
        summary[f"{metric}_min"] = grouped.min()
        summary[f"{metric}_max"] = grouped.max()
        summary[f"{metric}_available_runs"] = grouped.count()
    return summary


def plot_penalty_response(
    runs: pd.DataFrame, output_dir: Path, *,
    baseline_air_cost: float = BASELINE_AIR_COST,
    baseline_ground_cost: float = BASELINE_GROUND_COST,
) -> None:
    """Write reassignment and objective figures plus their auditable data."""
    summary = build_penalty_response(
        runs, baseline_air_cost=baseline_air_cost, baseline_ground_cost=baseline_ground_cost,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "sensitivity_penalty_response.csv")
    # Include the reference even if that setting was not tested; no data marker
    # or interpolated value is manufactured at this position.
    settings = sorted(set(summary.index) | {100.0})
    display = summary.reindex(settings)
    positions = list(range(len(settings)))
    tick_labels = [
        f"{p:g}" if p in summary.index else f"{p:g}\n(reference only)" for p in settings
    ]
    font_settings = {
        "font.size": MIN_FIGURE_FONT_SIZE,
        "axes.titlesize": MIN_FIGURE_FONT_SIZE + 1,
        "axes.labelsize": MIN_FIGURE_FONT_SIZE,
        "xtick.labelsize": MIN_FIGURE_FONT_SIZE,
        "ytick.labelsize": MIN_FIGURE_FONT_SIZE,
        "legend.fontsize": MIN_FIGURE_FONT_SIZE,
    }
    with plt.rc_context(font_settings):
        figure, axes = plt.subplots(
            2, 1, sharex=True, figsize=(FIGURE_WIDTH_INCHES, 6.0)
        )
        for axis, mode, color in zip(
            axes, ("air", "ground"), ("#276DAD", "#B55D27")
        ):
            metric = f"expected_reassigned_{mode}_legs"
            if mode == "air":
                axis.fill_between(
                    positions, display[f"{metric}_min"].to_numpy(dtype=float),
                    display[f"{metric}_max"].to_numpy(dtype=float),
                    color=color, alpha=0.2,
                    label="Range of expected number of reassigned air legs across tested transportation costs",
                )
            axis.plot(
                positions, display[f"{metric}_baseline"], color=color, marker="o",
                linewidth=2, label=f"Expected number of reassigned {mode} legs given baseline transportation costs",
            )
            axis.axvline(
                settings.index(100.0), color="#666666", linestyle="--",
                linewidth=1.2, label=r"Baseline penalty $c_{\mathrm{pen}}=100$",
            )
            axis.set_title(f"{mode.capitalize()} Reassignment", loc="left")
            axis.set_ylabel(f"Expected Reassigned {mode.capitalize()} Legs\nper Shipment")
            axis.grid(which="major", axis="both", alpha=0.25)
            axis.margins(x=0.04, y=0.15)
            if summary[f"{metric}_baseline"].isna().any():
                warnings.warn(
                    f"Missing {mode} baseline results at some penalties; line has gaps.",
                    stacklevel=2,
                )
        axes[-1].set_xticks(positions, tick_labels)
        axes[-1].set_xlabel(r"Reassignment Penalty $c_{\mathrm{pen}}$")
        legend_lookup = {}
        for axis in axes:
            axis_handles, axis_labels = axis.get_legend_handles_labels()
            legend_lookup.update(zip(axis_labels, axis_handles))
        legend_labels = [
            "Range of expected number of reassigned air legs across tested transportation costs",
            "Expected number of reassigned air legs given baseline transportation costs",
            "Expected number of reassigned ground legs given baseline transportation costs",
            r"Baseline penalty $c_{\mathrm{pen}}=100$",
        ]
        legend_handles = [legend_lookup[label] for label in legend_labels]
        figure.legend(
            legend_handles, legend_labels, loc="lower center",
            bbox_to_anchor=(0.5, 0.01), ncol=1, frameon=True,
        )
        figure.tight_layout(rect=(0, 0.15, 1, 1))
        figure.savefig(output_dir / "sensitivity_penalty_response.pdf", bbox_inches="tight")
        plt.close(figure)

        metric = "objective_value"
        figure, axis = plt.subplots(figsize=(FIGURE_WIDTH_INCHES, 4.1))
        axis.fill_between(
            positions, display[f"{metric}_min"].to_numpy(dtype=float),
            display[f"{metric}_max"].to_numpy(dtype=float), color="#4C78A8", alpha=0.2,
            label="Objective range across tested transportation costs",
        )
        axis.plot(
            positions, display[f"{metric}_baseline"], color="#4C78A8", marker="o",
            linewidth=2, label="Objective value",
        )
        axis.axvline(
            settings.index(100.0), color="#666666", linestyle="--",
            linewidth=1.2, label=r"Baseline penalty $c_{\mathrm{pen}}=100$",
        )
        axis.set(
            xlabel=r"Reassignment Penalty $c_{\mathrm{pen}}$",
            ylabel="Objective Value",
        )
        axis.set_xticks(positions, tick_labels)
        axis.grid(which="major", axis="both", alpha=0.25)
        axis.margins(x=0.04, y=0.15)
        legend_handles, legend_labels = axis.get_legend_handles_labels()
        figure.legend(
            legend_handles, legend_labels, loc="lower center",
            bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=True,
        )
        if summary[f"{metric}_baseline"].isna().any():
            warnings.warn(
                "Missing objective baseline results at some penalties; line has gaps.",
                stacklevel=2,
            )
        figure.tight_layout(rect=(0, 0.19, 1, 1))
        figure.savefig(
            output_dir / "sensitivity_objective_penalty_response.pdf",
            bbox_inches="tight",
        )
        plt.close(figure)


def plot_tables(
    tables: Mapping[str, pd.DataFrame], output_dir: Path, *,
    baseline_air_cost: float = BASELINE_AIR_COST,
    baseline_ground_cost: float = BASELINE_GROUND_COST,
) -> None:
    """Write five heatmaps, two penalty-response PDFs, and a summary CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white", context="notebook")
    runs = tables["runs"]
    ground_costs = sorted(runs["cost_ground"].unique())
    # Heatmap rows render from top to bottom, so descending air costs put the
    # greatest value at the top. Ground costs remain ascending left to right.
    air_costs = sorted(runs["cost_flight"].unique(), reverse=True)
    penalties = sorted(runs["cost_penalty_incompatibility"].unique())
    rows = min(3, len(penalties))
    columns = math.ceil(len(penalties) / rows)
    for metric, (_, label) in METRICS.items():
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
        figure.savefig(output_dir / f"sensitivity_{metric}.pdf", bbox_inches="tight")
        plt.close(figure)

    plot_penalty_response(
        runs, output_dir, baseline_air_cost=baseline_air_cost,
        baseline_ground_cost=baseline_ground_cost,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=Path(__file__).parent / "output" / "sensitivity" / "results_500",
        help="Directory containing detailed run_*.json files.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent.parent / "output" / "figures" / "sensitivity" / "results_500",
        help="Directory for five PDF heatmaps, two penalty-response PDFs, and CSV.",
    )
    parser.add_argument(
        "--baseline-air-cost", type=float, default=BASELINE_AIR_COST,
        help="Air cost for the baseline response line (default: %(default)g).",
    )
    parser.add_argument(
        "--baseline-ground-cost", type=float, default=BASELINE_GROUND_COST,
        help="Ground cost for the baseline response line (default: %(default)g).",
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
    plot_tables(tables, args.output_dir, baseline_air_cost=args.baseline_air_cost,
                baseline_ground_cost=args.baseline_ground_cost)
    print(f"Processed {len(results)} runs and wrote seven PDF figures and a summary CSV to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
