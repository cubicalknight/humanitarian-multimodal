"""Map one shipment with a large route change in a cost-sensitivity sweep.

The output is three independent PDFs showing an initial assignment, a genuine
recourse reroute for the same shipment and scenario, and that scenario's route
under a different parameter setting.

Run from the repository root, for example::

    poetry run python src/plot_sensitivity_maps.py
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PARAMETERS = ("cost_ground", "cost_flight", "cost_penalty_incompatibility")
LegKey = tuple[str, str, str]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "output" / "sensitivity" / "results_500"
DEFAULT_OUTPUT_DIR = (
    SCRIPT_DIR.parent
    / "output"
    / "figures"
    / "sensitivity"
    / "results_500"
)


@dataclass(frozen=True, slots=True)
class ShipmentSelection:
    shipment_id: str
    scenario: int
    baseline_change: float
    varied_change: float

    @property
    def score(self) -> float:
        return self.baseline_change + self.varied_change


@dataclass(frozen=True, slots=True)
class ComparisonSelection:
    reference: Mapping[str, Any]
    varied: Mapping[str, Any]
    shipments: tuple[ShipmentSelection, ...]


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number.") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number.")
    return result


def _parameter_key(result: Mapping[str, Any]) -> tuple[float, ...]:
    parameters = result.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("Every result must contain a parameters object.")
    return tuple(_finite_number(parameters.get(field), field) for field in PARAMETERS)


def _validate_result(result: Mapping[str, Any], source: str = "result") -> None:
    required = {"parameters", "objective_value", "num_scenarios", "decision_variables_by_shipment"}
    missing = required - result.keys()
    if missing:
        raise ValueError(f"{source} is missing required fields: {sorted(missing)}")
    _parameter_key(result)
    count = result["num_scenarios"]
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError(f"{source} num_scenarios must be a positive integer.")
    shipments = result["decision_variables_by_shipment"]
    if not isinstance(shipments, Mapping) or not shipments:
        raise ValueError(f"{source} must contain shipment decisions.")
    for shipment_id, decisions in shipments.items():
        if not isinstance(decisions, Mapping):
            raise ValueError(f"{source} shipment {shipment_id!r} decisions must be an object.")
        if not isinstance(decisions.get("first_stage", []), list):
            raise ValueError(f"{source} shipment {shipment_id!r} first_stage must be a list.")
        recourse = decisions.get("recourse")
        if not isinstance(recourse, Mapping):
            raise ValueError(f"{source} shipment {shipment_id!r} recourse must be an object.")
        for scenario, legs in recourse.items():
            try:
                scenario_index = int(scenario)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{source} shipment {shipment_id!r} has invalid scenario {scenario!r}."
                ) from error
            if str(scenario_index) != str(scenario) or not 0 <= scenario_index < count:
                raise ValueError(
                    f"{source} shipment {shipment_id!r} has invalid scenario {scenario!r}."
                )
            if not isinstance(legs, list):
                raise ValueError(
                    f"{source} shipment {shipment_id!r} scenario {scenario} must be a list."
                )


def load_results(input_dir: Path) -> list[dict[str, Any]]:
    """Load detailed sensitivity results, excluding summaries and configuration."""
    results: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("run_*.json")):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(result, dict):
                raise ValueError("top-level value must be an object")
            _validate_result(result, str(path))
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"Invalid sensitivity result {path}: {error}") from error
        results.append(result)
    if not results:
        raise FileNotFoundError(f"No run_*.json result files found in {input_dir}.")
    return results


def leg_key(leg: Mapping[str, Any]) -> LegKey:
    try:
        mode = str(leg["mode"])
        key = str(leg["origin"]), str(leg["destination"]), mode
    except KeyError as error:
        raise ValueError(f"Malformed route leg; missing {error.args[0]!r}.") from error
    if mode not in {"air", "ground"}:
        raise ValueError(f"Unknown route mode {mode!r}.")
    return key


def jaccard_distance(left: set[LegKey], right: set[LegKey]) -> float:
    """Return normalized route-set change; two empty routes have distance zero."""
    union = left | right
    return 1.0 - len(left & right) / len(union) if union else 0.0


def first_stage_route(decisions: Mapping[str, Any], threshold: float) -> set[LegKey]:
    return {
        leg_key(leg)
        for leg in decisions.get("first_stage", [])
        if _finite_number(leg.get("value", 0.0), "first-stage value") > threshold
    }


def recourse_routes(
    decisions: Mapping[str, Any], scenario: int, threshold: float
) -> tuple[set[LegKey], set[LegKey], set[LegKey]]:
    """Return final, kept, and reassigned route sets for one scenario."""
    final: set[LegKey] = set()
    kept: set[LegKey] = set()
    reassigned: set[LegKey] = set()
    for leg in decisions.get("recourse", {}).get(str(scenario), []):
        key = leg_key(leg)
        keep = _finite_number(leg.get("keep", 0.0), "keep value")
        reassign = _finite_number(leg.get("reassign", 0.0), "reassign value")
        if keep > threshold:
            kept.add(key)
        if reassign > threshold:
            reassigned.add(key)
        if keep + reassign > threshold:
            final.add(key)
    return final, kept, reassigned


def _run_index(result: Mapping[str, Any]) -> int:
    value = result.get("run_index")
    return value if isinstance(value, int) and not isinstance(value, bool) else 2**63 - 1


def _mean_final_route_change(
    baseline: Mapping[str, Any], varied: Mapping[str, Any], threshold: float
) -> float:
    baseline_shipments = baseline["decision_variables_by_shipment"]
    varied_shipments = varied["decision_variables_by_shipment"]
    if set(baseline_shipments) != set(varied_shipments):
        raise ValueError(
            f"Run {_run_index(varied)} has a different shipment set from the baseline."
        )
    distances = []
    for shipment_id in sorted(baseline_shipments):
        for scenario in range(int(baseline["num_scenarios"])):
            baseline_route = recourse_routes(
                baseline_shipments[shipment_id], scenario, threshold
            )[0]
            varied_route = recourse_routes(
                varied_shipments[shipment_id], scenario, threshold
            )[0]
            distances.append(jaccard_distance(baseline_route, varied_route))
    return sum(distances) / len(distances)


def select_comparison(
    results: Iterable[Mapping[str, Any]],
    base_parameters: Mapping[str, Any],
    *,
    active_threshold: float = 1e-6,
) -> ComparisonSelection:
    """Select the strongest genuine reroute and a contrasting parameter run."""
    threshold = _finite_number(active_threshold, "active_threshold")
    if threshold < 0:
        raise ValueError("active_threshold must be non-negative.")
    result_list = list(results)
    if not result_list:
        raise ValueError("At least one sensitivity result is required.")
    for index, result in enumerate(result_list):
        _validate_result(result, f"result {index}")

    _ = base_parameters  # Kept for CLI/API compatibility with earlier versions.
    keys = [_parameter_key(result) for result in result_list]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate parameter combinations were found in the sensitivity results.")
    scenario_count = int(result_list[0]["num_scenarios"])
    for result in result_list:
        if int(result["num_scenarios"]) != scenario_count:
            raise ValueError(
                f"Run {_run_index(result)} has {result['num_scenarios']} scenarios; "
                f"expected {scenario_count}."
            )

    reroutes: list[tuple[float, int, int, Mapping[str, Any], str, int]] = []
    for result in result_list:
        if result["objective_value"] is None:
            continue
        for shipment_id, decisions in result["decision_variables_by_shipment"].items():
            initial = first_stage_route(decisions, threshold)
            for scenario in range(scenario_count):
                final, _, reassigned = recourse_routes(decisions, scenario, threshold)
                change = jaccard_distance(initial, final)
                if reassigned and change > 0:
                    reroutes.append(
                        (change, len(initial ^ final), len(reassigned), result,
                         str(shipment_id), scenario)
                    )
    if not reroutes:
        raise ValueError("No genuine reassignment route change was found in the sensitivity results.")
    _, _, _, reference, shipment_id, scenario = sorted(
        reroutes,
        key=lambda item: (
            -item[0], -item[1], -item[2], _run_index(item[3]), item[4], item[5]
        ),
    )[0]
    reference_decisions = reference["decision_variables_by_shipment"][shipment_id]
    initial = first_stage_route(reference_decisions, threshold)
    reference_final = recourse_routes(reference_decisions, scenario, threshold)[0]

    varied_candidates: list[tuple[float, Mapping[str, Any]]] = []
    for result in result_list:
        if result is reference or result["objective_value"] is None:
            continue
        decisions = result["decision_variables_by_shipment"].get(shipment_id)
        if decisions is None:
            continue
        varied_final = recourse_routes(decisions, scenario, threshold)[0]
        varied_candidates.append((jaccard_distance(reference_final, varied_final), result))
    if not varied_candidates:
        raise ValueError("No contrasting parameter run is available for the selected reroute.")
    varied_change, varied = sorted(
        varied_candidates,
        key=lambda item: (-item[0], _run_index(item[1]), _parameter_key(item[1])),
    )[0]
    selection = ShipmentSelection(
        shipment_id=shipment_id,
        scenario=scenario,
        baseline_change=jaccard_distance(initial, reference_final),
        varied_change=varied_change,
    )
    return ComparisonSelection(
        reference=reference,
        varied=varied,
        shipments=(selection,),
    )


def changed_parameter_label(
    baseline: Mapping[str, Any], varied: Mapping[str, Any]
) -> str:
    changes = []
    for field in PARAMETERS:
        before = float(baseline["parameters"][field])
        after = float(varied["parameters"][field])
        if before != after:
            changes.append(f"{field}: {before:g} -> {after:g}")
    return ", ".join(changes) or "No parameter change"


def load_node_coordinates(config_path: Path) -> dict[str, tuple[float, float]]:
    """Rebuild only deterministic network metadata and return lon/lat pairs."""
    from main import ProblemPreparer, SensitivityConfig

    network = ProblemPreparer(SensitivityConfig(config_path)).prepare_network()
    return {
        node_id: (float(node.longitude), float(node.latitude))
        for node_id, node in network.nodes.items()
    }


def _add_background(axis: plt.Axes) -> None:
    # Pin Natural Earth to the bundled coarse scale. Adaptive features try to
    # download 10m data as soon as a map is zoomed to one shipment.
    axis.add_feature(
        cfeature.LAND.with_scale("110m"),
        facecolor="lightgray",
        edgecolor="black",
        linewidth=0.35,
    )
    axis.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#e6f7ff")
    axis.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.4)
    axis.add_feature(
        cfeature.BORDERS.with_scale("110m"),
        linestyle=":",
        linewidth=0.35,
        alpha=0.6,
    )
    axis.set_global()


def _draw_routes(
    axis: plt.Axes,
    routes: Iterable[tuple[LegKey, str]],
    coordinates: Mapping[str, tuple[float, float]],
) -> None:
    mode_colors = {"air": "#1769aa", "ground": "#d97706"}
    route_list = list(routes)
    node_ids = sorted({node for route, _ in route_list for node in route[:2]})
    missing = sorted(set(node_ids) - coordinates.keys())
    if missing:
        raise ValueError(f"Route nodes are missing coordinates: {missing}")
    for (origin, destination, mode), status in route_list:
        origin_lon, origin_lat = coordinates[origin]
        destination_lon, destination_lat = coordinates[destination]
        is_context = status == "context"
        axis.plot(
            [origin_lon, destination_lon],
            [origin_lat, destination_lat],
            color="#7c8794" if is_context else mode_colors[mode],
            linewidth=1.5 if is_context else 2.8,
            linestyle="--" if status == "reassigned" else "-",
            alpha=0.5 if is_context else 0.95,
            transform=ccrs.Geodetic(),
            zorder={"context": 2, "kept": 3, "reassigned": 4}.get(status, 3),
        )
    if not node_ids:
        return
    axis.scatter(
        [coordinates[node][0] for node in node_ids],
        [coordinates[node][1] for node in node_ids],
        color="darkred",
        s=22,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )
    for index, node in enumerate(node_ids):
        longitude, latitude = coordinates[node]
        axis.annotate(
            node,
            xy=(longitude, latitude),
            xycoords=ccrs.PlateCarree()._as_mpl_transform(axis),
            xytext=(3, 3 + 7 * (index % 3)),
            textcoords="offset points",
            fontsize=6.5,
            color="darkred",
            ha="left",
            va="bottom",
            zorder=6,
        )


def _set_route_extent(
    axis: plt.Axes,
    routes: Iterable[LegKey],
    coordinates: Mapping[str, tuple[float, float]],
) -> None:
    """Zoom an axis around a row's routes, falling back to a global view."""
    node_ids = {node for route in routes for node in route[:2]}
    if not node_ids or node_ids - coordinates.keys():
        axis.set_global()
        return
    longitudes = [coordinates[node][0] for node in node_ids]
    latitudes = [coordinates[node][1] for node in node_ids]
    longitude_span = max(longitudes) - min(longitudes)
    latitude_span = max(latitudes) - min(latitudes)
    # A simple bounding box is misleading for routes that cross the date line.
    if longitude_span > 180:
        axis.set_global()
        return
    longitude_padding = max(5.0, longitude_span * 0.08)
    # Great-circle arcs between North America and Europe bow well north of
    # their endpoints, so endpoint-only padding clips the middle of the arc.
    latitude_padding = max(25.0, latitude_span * 1.5)
    extent = (
        max(-180.0, min(longitudes) - longitude_padding),
        min(180.0, max(longitudes) + longitude_padding),
        max(-85.0, min(latitudes) - latitude_padding),
        min(85.0, max(latitudes) + latitude_padding),
    )
    axis.set_extent(extent, crs=ccrs.PlateCarree())


def _legend_handles(*, include_context: bool) -> list[Line2D]:
    handles = [
        Line2D([0], [0], color="#1769aa", lw=3, label="Air"),
        Line2D([0], [0], color="#d97706", lw=3, label="Ground"),
    ]
    if include_context:
        handles.extend(
            [
                Line2D(
                    [0], [0], color="#7c8794", lw=1.5, alpha=0.5,
                    label="Prior route",
                ),
                Line2D(
                    [0], [0], color="#4b5563", lw=2.8, linestyle="-",
                    label="Kept",
                ),
                Line2D(
                    [0], [0], color="#4b5563", lw=2.8, linestyle="--",
                    label="Reassigned",
                ),
            ]
        )
    handles.append(
        Line2D(
            [0], [0], color="darkred", marker="o", linestyle="None", label="Node"
        )
    )
    return handles


def plot_comparison(
    selection: ComparisonSelection,
    coordinates: Mapping[str, tuple[float, float]],
    output_dir: Path,
    *,
    active_threshold: float = 1e-6,
) -> tuple[Path, Path, Path]:
    """Render three independent maps for the highest-ranked shipment."""
    projection = ccrs.Robinson(central_longitude=-20)
    reference_shipments = selection.reference["decision_variables_by_shipment"]
    varied_shipments = selection.varied["decision_variables_by_shipment"]
    chosen = selection.shipments[0]
    initial = first_stage_route(reference_shipments[chosen.shipment_id], active_threshold)
    _, baseline_kept, baseline_reassigned = recourse_routes(
        reference_shipments[chosen.shipment_id], chosen.scenario, active_threshold
    )
    _, varied_kept, varied_reassigned = recourse_routes(
        varied_shipments[chosen.shipment_id], chosen.scenario, active_threshold
    )
    baseline_final = baseline_kept | baseline_reassigned
    varied_final = varied_kept | varied_reassigned
    all_routes = initial | baseline_final | varied_final
    parameter_changes = changed_parameter_label(selection.reference, selection.varied)
    panels = (
        (
            "initial_assignment",
            "Initial assignment",
            [(route, "kept") for route in sorted(initial)],
            False,
            f"Reference run {_run_index(selection.reference)} first-stage route",
        ),
        (
            "baseline_reassignment",
            "Reference reassignment",
            [(route, "context") for route in sorted(initial)]
            + [(route, "kept") for route in sorted(baseline_kept)]
            + [(route, "reassigned") for route in sorted(baseline_reassigned)],
            True,
            f"{len(baseline_reassigned)} reassigned legs; "
            f"geometry change = {chosen.baseline_change:.2f}",
        ),
        (
            "varied_parameter",
            "Varied-parameter outcome",
            [(route, "context") for route in sorted(baseline_final)]
            + [(route, "kept") for route in sorted(varied_kept)]
            + [(route, "reassigned") for route in sorted(varied_reassigned)],
            True,
            f"route change = {chosen.varied_change:.2f}\n"
            + textwrap.fill(parameter_changes, width=95),
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for stem, title, routes, include_context, subtitle in panels:
        figure = plt.figure(figsize=(10, 7.2))
        axis = figure.add_subplot(1, 1, 1, projection=projection)
        _add_background(axis)
        _set_route_extent(axis, all_routes, coordinates)
        _draw_routes(axis, routes, coordinates)
        axis.set_title(
            f"Shipment {chosen.shipment_id}, scenario {chosen.scenario}\n{title}\n{subtitle}",
            fontsize=12,
            pad=10,
        )
        handles = _legend_handles(include_context=include_context)
        figure.legend(
            handles=handles,
            loc="lower center",
            ncol=len(handles),
            frameon=True,
        )
        figure.subplots_adjust(top=0.78, bottom=0.15, left=0.04, right=0.96)
        output_path = output_dir / f"shipment_{chosen.shipment_id}_{stem}.png"
        figure.savefig(output_path, bbox_inches="tight")
        plt.close(figure)
        output_paths.append(output_path)
    return output_paths[0], output_paths[1], output_paths[2]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
        help="Directory containing detailed run_*.json sensitivity results.",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Sensitivity configuration; defaults to <input-dir>/config.json.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory for the three independent PDF maps.",
    )
    parser.add_argument(
        "--active-threshold", type=float, default=1e-6,
        help="Minimum active first-stage, keep, or reassign flow.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    config_path = args.config or args.input_dir / "config.json"
    try:
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read sensitivity configuration {config_path}: {error}") from error
    if not isinstance(config_data.get("base_parameters"), Mapping):
        raise ValueError(f"{config_path} has no base_parameters object.")
    results = load_results(args.input_dir)
    selection = select_comparison(
        results,
        config_data["base_parameters"],
        active_threshold=args.active_threshold,
    )
    coordinates = load_node_coordinates(config_path)
    output_paths = plot_comparison(
        selection,
        coordinates,
        args.output_dir,
        active_threshold=args.active_threshold,
    )
    print(f"Selected varied run {_run_index(selection.varied)}: "
          f"{changed_parameter_label(selection.reference, selection.varied)}")
    for chosen in selection.shipments:
        print(f"Selected shipment {chosen.shipment_id}, scenario {chosen.scenario}, "
              f"combined route-change score {chosen.score:.3f}")
    for output_path in output_paths:
        print(f"Wrote shipment route-change map to {output_path.resolve()}")


if __name__ == "__main__":
    main()
