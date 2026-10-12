"""Plot four baseline objectives and first-stage/recourse ground and air links.

Run: poetry run python src/plot_base_case_comparison.py
First-stage and recourse link counts are shown as four stacked horizontal
frequency plots with shared shipment-count axes. Each recourse subplot includes
kept and reassigned links over a wider first-stage frequency bar, with each
shipment repeated across its scenarios. Explicit paths resolve from the working
directory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator, MultipleLocator

CASES = (
    "integrated_restricted", "integrated_unrestricted",
    "myopic_restricted", "myopic_unrestricted",
)
LABELS = ["Integrated\nRestricted", "Integrated\nUnrestricted",
          "Myopic\nRestricted", "Myopic\nUnrestricted"]
COLORS = ["#1f77b4", "#79aed2", "#d62728", "#e99494"]
PLOT_WIDTH = 6.5
PLOT_HEIGHT = 7


def load_results(input_dir: Path) -> list[dict]:
    results = []
    for case in CASES:
        path = input_dir / f"{case}.json"
        with path.open(encoding="utf-8") as stream:
            result = json.load(stream)
        if result.get("case", case) != case:
            raise ValueError(f"{path}: case does not match filename")
        if "decision_variables_by_shipment" not in result:
            raise ValueError(f"{path}: missing shipment decisions")
        results.append(result | {"case": case})
    return results


def build_tables(results: list[dict], *, active_threshold: float = 1e-6) -> dict[str, pd.DataFrame]:
    if not math.isfinite(active_threshold) or active_threshold < 0:
        raise ValueError("active_threshold must be finite and non-negative")
    runs, first, final = [], [], []
    for result in results:
        case = result["case"]
        total = result.get("total_expected_cost", result.get("objective_value"))
        runs.append({"case": case, "objective": total, "status": result.get("status")})
        first_stage = result.get("stages", {}).get("first_stage", {})
        has_first = total is not None or first_stage.get("solution_count", 0) > 0
        for shipment_id, decisions in result["decision_variables_by_shipment"].items():
            if has_first:
                selected = {(leg["origin"], leg["destination"], leg["mode"])
                            for leg in decisions.get("first_stage", [])
                            if float(leg.get("value", 0)) > 0.5}
                for mode in ("ground", "air"):
                    first.append({"case": case, "shipment": shipment_id, "mode": mode,
                                  "count": sum(r[2] == mode for r in selected)})
            if total is None:
                continue
            scenarios = decisions.get("recourse", {})
            scenario_ids = (range(int(result["num_scenarios"])) if result.get("num_scenarios")
                            else sorted(map(int, scenarios)))
            for scenario in scenario_ids:
                legs = scenarios.get(str(scenario), [])
                for mode in ("ground", "air"):
                    for metric in ("Kept", "Reassigned", "Final"):
                        active = set()
                        for leg in legs:
                            keep = float(leg.get("keep", 0))
                            reassign = float(leg.get("reassign", 0))
                            value = {"Kept": keep, "Reassigned": reassign,
                                     "Final": keep + reassign}[metric]
                            if leg["mode"] == mode and value > active_threshold:
                                active.add((leg["origin"], leg["destination"], mode))
                        final.append({"case": case, "shipment": shipment_id,
                                      "scenario": scenario, "mode": mode,
                                      "metric": metric, "count": len(active)})
    return {"runs": pd.DataFrame(runs),
            "first": pd.DataFrame(first, columns=["case", "shipment", "mode", "count"]),
            "final": pd.DataFrame(final, columns=["case", "shipment", "scenario", "mode", "metric", "count"])}


def _save(fig, output_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _set_link_count_ticks(axis) -> None:
    """Mark every integer link count while labeling a readable subset."""
    axis.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    axis.yaxis.set_minor_locator(MultipleLocator(1))
    axis.tick_params(axis="y", which="major", left=True, direction="out", length=5)
    axis.tick_params(axis="y", which="minor", left=True, direction="out", length=3)


def plot_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook", rc={
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 12,
    })
    fig, axis = plt.subplots(figsize=(PLOT_WIDTH, 4.8))
    runs = tables["runs"].set_index("case")
    for i, case in enumerate(CASES):
        value = runs.loc[case, "objective"]
        if pd.isna(value):
            axis.text(i, 0.04, "No incumbent", transform=axis.get_xaxis_transform(),
                      ha="center", fontsize=10)
        else:
            axis.bar(i, value, color=COLORS[i], width=0.65)
            axis.annotate(f"{value:,.0f}", (i, value), xytext=(0, 5),
                          textcoords="offset points", ha="center", fontsize=10)
    axis.set_xticks(range(4), LABELS)
    axis.set(title="Base-case total expected cost", ylabel="Objective value", xlim=(-0.6, 3.6))
    axis.margins(y=0.18)
    _save(fig, output_dir, "objective")

    for mode in ("ground", "air"):
        data = tables["first"]
        data = data[data["mode"] == mode]
        max_links = int(data["count"].max()) if not data.empty else 0
        levels = np.arange(max_links + 1)
        fig, axes = plt.subplots(4, 1, figsize=(PLOT_WIDTH, PLOT_HEIGHT), sharex=True)
        for i, (axis, case) in enumerate(zip(axes, CASES, strict=True)):
            values = data.loc[data["case"] == case, "count"]
            frequencies = values.value_counts().reindex(levels, fill_value=0)
            axis.barh(levels, frequencies, height=0.68, color=COLORS[i])
            axis.set_title(LABELS[i].replace("\n", " "), loc="left", fontsize=10)
            axis.set_ylim(-0.55, max_links + 0.55)
            _set_link_count_ticks(axis)
            axis.grid(axis="y", visible=False)
            if values.empty:
                axis.text(0.5, 0.5, "N/A", transform=axis.transAxes,
                          ha="center", va="center", fontsize=10)
        fig.supxlabel("Number of Shipments")
        axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.supylabel(f"Active {mode.capitalize()} Links per Shipment")
        # fig.suptitle(f"First-stage {mode} links\nShipment frequency by active-link count")
        _save(fig, output_dir, f"first_stage_{mode}_links")

    for mode in ("ground", "air"):
        data = tables["final"]
        data = data[data["mode"] == mode]
        recourse = data[data["metric"].isin(("Kept", "Reassigned"))]
        # Repeat each first-stage count once per shipment/scenario, matching
        # the observation population used by the recourse distributions.
        pairs = data[["case", "shipment", "scenario", "mode"]].drop_duplicates()
        first_repeated = pairs.merge(tables["first"], on=["case", "shipment", "mode"],
                                     how="inner", validate="many_to_one")
        counts = pd.concat([recourse["count"], first_repeated["count"]])
        max_links = int(counts.max()) if not counts.empty else 0
        # Only the first-stage reference may show a zero-count bin.
        min_links = 0 if first_repeated["count"].eq(0).any() else 1
        levels = np.arange(min_links, max(max_links, 1) + 1)
        recourse_levels = levels[levels > 0]
        fig, axes = plt.subplots(4, 1, figsize=(PLOT_WIDTH, PLOT_HEIGHT), sharex=True)
        for axis, case in zip(axes, CASES, strict=True):
            case_data = recourse[recourse["case"] == case]
            first_counts = first_repeated.loc[first_repeated["case"] == case, "count"]
            first_frequencies = first_counts.value_counts().reindex(levels, fill_value=0)
            axis.barh(levels, first_frequencies, height=0.86, color="#cbd0d6",
                      edgecolor="#8a9199", linewidth=0.7,
                      label="First stage × scenarios", zorder=2)
            for offset, metric, color in zip(
                (-0.18, 0.18), ("Kept", "Reassigned"),
                ("#59a14f", "#f28e2b"), strict=True,
            ):
                values = case_data.loc[case_data["metric"] == metric, "count"]
                frequencies = values.value_counts().reindex(recourse_levels, fill_value=0)
                axis.barh(recourse_levels + offset, frequencies, height=0.32,
                          color=color, label=metric, zorder=3)
            axis.set_title(LABELS[CASES.index(case)].replace("\n", " "),
                           loc="left", fontsize=10)
            axis.set_ylim(min_links - 0.55, max(max_links, 1) + 0.55)
            _set_link_count_ticks(axis)
            axis.grid(axis="y", visible=False)
            if case_data.empty:
                axis.text(0.5, 0.5, "N/A", transform=axis.transAxes,
                          ha="center", va="center", fontsize=10)
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(color=color, label=label) for color, label in
                            (("#cbd0d6", "First Stage Assignments × Scenarios"),
                             ("#59a14f", "Kept"), ("#f28e2b", "Reassigned"))],
                   frameon=False, loc="lower center",
                   bbox_to_anchor=(0.5, -0.05), ncol=3)
        axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.supxlabel("Number of Shipment-Scenario Pairs")
        fig.supylabel(f"Active {mode.capitalize()} Links per Shipment-Scenario")
        _save(fig, output_dir, f"final_{mode}_links")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    src = Path(__file__).resolve().parent
    parser.add_argument("--input-dir", type=Path, default=src / "output/base_case_comparison")
    parser.add_argument("--output-dir", type=Path, default=src / "output/figures/base_case_comparison")
    parser.add_argument("--active-threshold", type=float, default=1e-6,
                        help="Minimum flow for kept, reassigned, or combined active links.")
    args = parser.parse_args(argv)
    tables = build_tables(load_results(args.input_dir), active_threshold=args.active_threshold)
    plot_tables(tables, args.output_dir)
    print(f"Wrote five comparison PDFs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
