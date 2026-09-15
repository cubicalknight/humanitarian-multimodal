"""
Two-stage stochastic optimization model for humanitarian airlift assignment.

This module implements a two-stage stochastic program to model the matching process
under operational uncertainty. The first stage assigns shipments to routes and airlines,
minimizing upfront costs and expected recourse costs. The second stage handles recourse
actions when carriers reject shipments or aircraft are incompatible.

Mathematical formulation:
    Stage 1: min x { sum(c_flight * x) + E[Q(x, ξ)] }
    Stage 2: min y,z { recourse costs | acceptance and compatibility realizations }

Where ξ = {A, B} represents random carrier acceptance and aircraft compatibility.
"""

from __future__ import annotations

import math
import os
import random
import signal
import tracemalloc
from collections.abc import Sequence
from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import count
from math import cos, radians, sin
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import gurobipy as gp
import matplotlib.pyplot as plt
import psutil
from gurobipy import GRB
from matplotlib.lines import Line2D
from tqdm import tqdm

_proc = psutil.Process(os.getpid())

def mem(tag: str) -> None:
    print(f"[mem] {tag}: RSS={_proc.memory_info().rss/2**30:.2f} GiB", flush=True)

# ============================================================================
# Data Structures
# ============================================================================
# Global flag to handle interruption
# interrupted = False
# def signal_handler(sig, frame):
#     global interrupted

#     interrupted = True
#     print("\nInterrupt received! Stopping optimization...")
# # Register the signal handler for KeyboardInterrupt (Ctrl+C)
# signal.signal(signal.SIGINT, signal_handler)

# times = []
# gaps = []
# def my_callback(model, where):
#     global interrupted

#     if where == GRB.Callback.MIP:
#         time = model.cbGet(GRB.Callback.RUNTIME)
#         # time = model.cbGet(GRB.Callback.MIP_NODCNT)
#         gap = abs(model.cbGet(GRB.Callback.MIP_OBJBST) - model.cbGet(GRB.Callback.MIP_OBJBND))/abs(model.cbGet(GRB.Callback.MIP_OBJBST)) * 100

#         if gap != float('inf'):
#             print(f"Time: {time:.2f} seconds, Gap: {gap:.2f}%")
#             times.append(time)
#             gaps.append(gap)

#     if where == GRB.Callback.MIPNODE and interrupted:
#         model.terminate()  # Stop optimization safely


@dataclass(frozen=True, slots=True)
class Node:
    """Represents a node in the network (airport or non-airport)."""
    node_id: str
    latitude: float
    longitude: float
    connection_type: str | None = None  # e.g., "airport", "gnd", etc.

@dataclass(frozen=True, slots=True)
class LegOption:
    """Represents a single leg option (route + mode)."""
    route_id: str
    # airline_id: str
    origin: Node
    destination: Node
    distance_miles: float
    mode: str

    mu_slack: float | None = None
    sigma_slack: float | None = None
    u_vec: tuple[float, ...] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.mode == "air":
            object.__setattr__(self, "u_vec", self._to_unit_cartesian())

    def _to_unit_cartesian(self) -> tuple[float, ...]:
        origin_lat = radians(self.origin.latitude)
        origin_lon = radians(self.origin.longitude)
        destination_lat = radians(self.destination.latitude)
        destination_lon = radians(self.destination.longitude)

        return (
            cos(origin_lon) * cos(origin_lat),
            sin(origin_lon) * cos(origin_lat),
            sin(origin_lat),
            cos(destination_lon) * cos(destination_lat),
            sin(destination_lon) * cos(destination_lat),
            sin(destination_lat),
            self.distance_miles / 20_000,
        )

# TODO change origin dest to Node objects
@dataclass(frozen=True, slots=True)
class Shipment:
    """Represents a single shipment to be assigned."""
    shipment_id: str
    weight: float
    origin: Node
    destination: Node
    pallets: float = 0.0
    equivalent_cost: float = 0.0
    commodity: str | None = None


RouteKey = tuple[str, str, str]
RoutePath = tuple[RouteKey, ...]


def _shortest_route_path(
    origin: str,
    destination: str,
    outgoing: dict[str, tuple[RouteKey, ...]],
    legs: dict[RouteKey, LegOption],
    banned_nodes: frozenset[str] = frozenset(),
    banned_routes: frozenset[RouteKey] = frozenset(),
) -> RoutePath | None:
    """Return one deterministic shortest simple path under Yen exclusions."""
    if origin in banned_nodes or destination in banned_nodes:
        return None
    if origin == destination:
        return ()

    sequence = count()
    queue: list[tuple[float, int, str, RoutePath, frozenset[str]]] = [
        (0.0, next(sequence), origin, (), frozenset((origin,)))
    ]
    best_distance = {origin: 0.0}

    while queue:
        distance, _, node, path, path_nodes = heappop(queue)
        if distance > best_distance.get(node, math.inf):
            continue
        if node == destination:
            return path

        for route in outgoing.get(node, ()):
            next_node = route[1]
            if (
                route in banned_routes
                or next_node in banned_nodes
                or next_node in path_nodes
            ):
                continue
            next_distance = distance + legs[route].distance_miles
            if next_distance >= best_distance.get(next_node, math.inf):
                continue
            best_distance[next_node] = next_distance
            heappush(
                queue,
                (
                    next_distance,
                    next(sequence),
                    next_node,
                    path + (route,),
                    path_nodes | {next_node},
                ),
            )

    return None


def k_shortest_route_paths(
    legs: dict[RouteKey, LegOption],
    origin: str,
    destination: str,
    max_paths: int = 10,
) -> tuple[RoutePath, ...]:
    """Return up to ``max_paths`` shortest loopless directed edge paths.

    This is Yen's algorithm over route keys rather than node pairs, so parallel
    air and ground legs between the same nodes remain distinct alternatives.
    """
    if max_paths <= 0:
        raise ValueError("max_paths must be positive")

    outgoing_lists: dict[str, list[RouteKey]] = {}
    for route, leg in legs.items():
        if not math.isfinite(leg.distance_miles) or leg.distance_miles < 0:
            raise ValueError(f"Route {route!r} has an invalid distance")
        outgoing_lists.setdefault(route[0], []).append(route)
    outgoing = {
        node: tuple(sorted(routes, key=lambda route: (route[1], route[2], route[0])))
        for node, routes in outgoing_lists.items()
    }

    first_path = _shortest_route_path(origin, destination, outgoing, legs)
    if first_path is None:
        return ()

    accepted: list[RoutePath] = [first_path]
    accepted_set = {first_path}
    candidates: list[tuple[float, RoutePath]] = []
    candidate_set: set[RoutePath] = set()

    while len(accepted) < max_paths:
        previous_path = accepted[-1]
        previous_nodes = (origin,) + tuple(route[1] for route in previous_path)

        for spur_index in range(len(previous_path)):
            root_path = previous_path[:spur_index]
            spur_node = previous_nodes[spur_index]
            removed_routes = frozenset(
                path[spur_index]
                for path in accepted
                if len(path) > spur_index and path[:spur_index] == root_path
            )
            removed_nodes = frozenset(previous_nodes[:spur_index])
            spur_path = _shortest_route_path(
                spur_node,
                destination,
                outgoing,
                legs,
                banned_nodes=removed_nodes,
                banned_routes=removed_routes,
            )
            if spur_path is None:
                continue

            candidate = root_path + spur_path
            if candidate in accepted_set or candidate in candidate_set:
                continue
            candidate_cost = sum(legs[route].distance_miles for route in candidate)
            heappush(candidates, (candidate_cost, candidate))
            candidate_set.add(candidate)

        if not candidates:
            break
        _, next_path = heappop(candidates)
        candidate_set.remove(next_path)
        accepted.append(next_path)
        accepted_set.add(next_path)

    return tuple(accepted)


def build_feasible_routes_by_shipment(
    shipments: dict[str, Shipment],
    legs: dict[RouteKey, LegOption],
    max_paths: int = 10,
) -> dict[str, tuple[RouteKey, ...]]:
    """Build each shipment's ordered union of routes from its K shortest paths."""
    paths_by_od: dict[tuple[str, str], tuple[RoutePath, ...]] = {}
    feasible: dict[str, tuple[RouteKey, ...]] = {}

    for shipment_id, shipment in shipments.items():
        od = (shipment.origin.node_id, shipment.destination.node_id)
        paths = paths_by_od.get(od)
        if paths is None:
            paths = k_shortest_route_paths(legs, *od, max_paths=max_paths)
            paths_by_od[od] = paths
        if not paths or not any(paths):
            raise ValueError(
                f"No directed route path for shipment {shipment_id!r} from "
                f"{od[0]!r} to {od[1]!r}"
            )

        seen: set[RouteKey] = set()
        ordered_routes: list[RouteKey] = []
        for path in paths:
            for route in path:
                if route not in seen:
                    seen.add(route)
                    ordered_routes.append(route)
        feasible[shipment_id] = tuple(ordered_routes)

    return feasible


@dataclass(slots=True)
class UncertaintyRealization:
    """
    Represents a realization of uncertainty ξ = {U}.
    """
    leg: LegOption
    num_scenarions: int
    scenario_realize: list[float]


@dataclass(slots=True)
class StochasticOptimizationParameters:
    """Parameters for the two-stage stochastic program."""
    
    cost_flight: float
    cost_ground: float

    cost_penalty_incompatibility: float  # Penalty for flight incompatibility
    
    # These can be overridden per route/airline
    cost_reassignment: float | None = None  # Difference (c_flight' - c_flight)

    def get_reassignment_cost(
        self,
        original_cost: float,
        new_cost: float | None = None,
    ) -> float:
        """Calculate reassignment cost as difference between new and original flights."""
        if new_cost is not None:
            return max(0.0, new_cost - original_cost)
        if self.cost_reassignment is not None:
            return self.cost_reassignment
        raise ValueError("Reassignment cost is not defined and no new cost provided.")


@dataclass(slots=True)
class FirstStageSolution:
    """Solution from the first stage optimization."""
    assignments: dict[tuple[str, str], float]  # (shipment_id, flight_id) -> probability
    objective_value: float
    status: str
    

@dataclass(slots=True)
class SecondStageSolution:
    """Solution from the second stage optimization."""
    keep_assignments: dict[tuple[str, str], float]  # (shipment_id, flight_id) -> keep indicator
    reassignments: dict[tuple[str, str], float]  # (shipment_id, flight_id) -> reassign indicator
    objective_value: float
    status: str


@dataclass(slots=True)
class TwoStageSolution:
    """Complete solution from both stages."""
    first_stage: FirstStageSolution
    second_stage: SecondStageSolution
    expected_total_cost: float = 0.0
    
    def total_cost(self) -> float:
        """Returns total expected cost: first stage + expected second stage."""
        cost = self.first_stage.objective_value
        if self.second_stage is not None:
            cost += self.second_stage.objective_value
        return cost


class TwoStageSolver:
    """
    Solves the complete two-stage stochastic program using:
    1. Myopic approach (greedy, no foresight)
    2. Sample average approximation (SAA) with integrated optimization
    """
    
    def __init__(
        self,
        shipments: dict[str, Shipment],
        legs: dict[RouteKey, LegOption],
        # nodes: dict[str, Node],
        params: StochasticOptimizationParameters,
        solver_quiet: bool = False,
        feasible_routes_by_shipment: dict[str, Sequence[RouteKey]] | None = None,
        recourse_path_limit: int = 20,
    ):
        self.shipments = shipments
        self.legs = legs
        self.params = params

        # Materialize these once.  They are traversed many times while the model is
        # built, so repeatedly filtering ``self.R`` in the inner loops is costly.
        self.S = tuple(shipments)  # [s.shipment_id for s in self.shipments]

        self.R = list(self.legs.keys())
        self.air_legs = [r for r in self.R if r[2] == "air"]
        self.gnd_legs = [r for r in self.R if r[2] == "ground"]
        
        self.nodes = {i for (i, _, _) in self.R} | {j for (_, j, _) in self.R}
        self.outgoing = {
            node: tuple(route for route in self.R if route[0] == node)
            for node in self.nodes
        }
        self.incoming = {
            node: tuple(route for route in self.R if route[1] == node)
            for node in self.nodes
        }
        if feasible_routes_by_shipment is None:
            feasible_routes_by_shipment = build_feasible_routes_by_shipment(
                shipments,
                legs,
                max_paths=recourse_path_limit,
            )

        route_set = set(self.R)
        self.feasible_by_shipment: dict[str, tuple[RouteKey, ...]] = {}
        self.feasible_route_sets: dict[str, frozenset[RouteKey]] = {}
        self.feasible_nodes: dict[str, tuple[str, ...]] = {}
        self.feasible_outgoing: dict[str, dict[str, tuple[RouteKey, ...]]] = {}
        self.feasible_incoming: dict[str, dict[str, tuple[RouteKey, ...]]] = {}
        self.feasible_air_outgoing: dict[str, dict[str, tuple[RouteKey, ...]]] = {}
        self.feasible_air_incoming: dict[str, dict[str, tuple[RouteKey, ...]]] = {}
        self.feasible_ground: dict[str, tuple[RouteKey, ...]] = {}

        for shipment_id in self.S:
            if shipment_id not in feasible_routes_by_shipment:
                raise ValueError(f"Missing feasible routes for shipment {shipment_id!r}")
            routes = tuple(dict.fromkeys(feasible_routes_by_shipment[shipment_id]))
            unknown_routes = set(routes) - route_set
            if unknown_routes:
                raise ValueError(
                    f"Shipment {shipment_id!r} has unknown feasible routes: "
                    f"{sorted(unknown_routes)!r}"
                )
            if not routes:
                raise ValueError(f"Shipment {shipment_id!r} has no feasible routes")

            nodes = tuple(dict.fromkeys(
                node
                for route in routes
                for node in (route[0], route[1])
            ))
            outgoing_lists: dict[str, list[RouteKey]] = {}
            incoming_lists: dict[str, list[RouteKey]] = {}
            for route in routes:
                outgoing_lists.setdefault(route[0], []).append(route)
                incoming_lists.setdefault(route[1], []).append(route)
            outgoing = {
                node: tuple(node_routes)
                for node, node_routes in outgoing_lists.items()
            }
            incoming = {
                node: tuple(node_routes)
                for node, node_routes in incoming_lists.items()
            }
            self.feasible_by_shipment[shipment_id] = routes
            self.feasible_route_sets[shipment_id] = frozenset(routes)
            self.feasible_nodes[shipment_id] = nodes
            self.feasible_outgoing[shipment_id] = outgoing
            self.feasible_incoming[shipment_id] = incoming
            self.feasible_air_outgoing[shipment_id] = {
                node: air_routes
                for node, routes_from_node in outgoing.items()
                if (air_routes := tuple(
                    route for route in routes_from_node if route[2] == "air"
                ))
            }
            self.feasible_air_incoming[shipment_id] = {
                node: air_routes
                for node, routes_to_node in incoming.items()
                if (air_routes := tuple(
                    route for route in routes_to_node if route[2] == "air"
                ))
            }
            self.feasible_ground[shipment_id] = tuple(
                route for route in routes if route[2] == "ground"
            )

        self.route_reassignment_cost = {
            route: (
                self.params.get_reassignment_cost(
                    self.params.cost_flight if route[2] == "air" else self.params.cost_ground
                )
                + self.params.cost_penalty_incompatibility
            )
            * self.legs[route].distance_miles
            for route in self.R
        }
        # self.nodes = nodes

        self.solver_quiet = solver_quiet


    def stage_one_setup(self, model):
        print("Setting up first stage optimization...")

        if self.solver_quiet:
            model.Params.OutputFlag = 0
            model.Params.LogToConsole = 0

        model.Params.Threads = 8

        x = model.addVars(self.S, self.R, vtype=GRB.BINARY, name="x")

        for shipment in self.S:
            origin = self.shipments[shipment].origin.node_id
            destination = self.shipments[shipment].destination.node_id

            for l in self.nodes:
                flow_out = gp.quicksum(x[shipment, *route] for route in self.outgoing[l])
                flow_in = gp.quicksum(x[shipment, *route] for route in self.incoming[l])

                if l == origin:
                    rhs = 1
                elif l == destination:
                    rhs = -1
                else:
                    rhs = 0

                model.addConstr(flow_out - flow_in == rhs, name=f"flow_conservation_{shipment}_{l}")

            model.addConstrs(x[shipment, o, i, 'ground'] == gp.quicksum(x[shipment, k, j, 'air'] for (k, j, _) in self.air_legs if k == i) for (o, i, _) in self.gnd_legs if o == origin)
            model.addConstrs(x[shipment, j, d, 'ground'] == gp.quicksum(x[shipment, i, k, 'air'] for (i, k, _) in self.air_legs if k == j) for (j, d, _) in self.gnd_legs if d == destination)

        cost = gp.quicksum(
            self.params.cost_flight * x[s, i, j, m] * self.shipments[s].weight * self.legs[(i, j, m)].distance_miles 
            if m == 'air'
            else self.params.cost_ground * x[s, i, j, m] * self.shipments[s].weight * self.legs[(i, j, m)].distance_miles
            for s in self.S
            for (i, j, m) in self.R
        )

        return model, x, cost
    

    def stage_two_setup(self, model, x, Omega, scenarios):
        """Build recourse variables only on each shipment's feasible route set."""
        print("Setting up second stage optimization...")

        if self.solver_quiet:
            model.Params.OutputFlag = 0
            model.Params.LogToConsole = 0

        model.Params.Threads = 8
        omega = tuple(Omega)
        required_air_routes = {
            route
            for routes in self.feasible_by_shipment.values()
            for route in routes
            if route[2] == "air"
        }
        missing_scenarios = required_air_routes - scenarios.keys()
        if missing_scenarios:
            raise ValueError(
                "Missing uncertainty realizations for feasible air routes: "
                f"{sorted(missing_scenarios)!r}"
            )
        if omega:
            max_scenario = max(omega)
            too_short = {
                route: len(scenarios[route].scenario_realize)
                for route in required_air_routes
                if len(scenarios[route].scenario_realize) <= max_scenario
            }
            if too_short:
                raise ValueError(
                    "Insufficient uncertainty realizations for scenario index "
                    f"{max_scenario}: {too_short!r}"
                )

        sparse_pairs = sum(len(self.feasible_by_shipment[s]) for s in self.S)
        dense_pairs = len(self.S) * len(self.R)
        reduction = 1.0 - sparse_pairs / dense_pairs if dense_pairs else 0.0
        print(
            "Second-stage shipment-route pairs: "
            f"{sparse_pairs:,} sparse vs {dense_pairs:,} dense "
            f"({reduction:.1%} reduction); "
            f"{2 * sparse_pairs * len(omega):,} recourse variables."
        )

        def second_stage_indices():
            return (
                (s, *route, om)
                for s in self.S
                for route in self.feasible_by_shipment[s]
                for om in omega
            )

        still_avail = model.addVars(
            second_stage_indices(),
            vtype=GRB.BINARY,
            name="keep",
        )
        print("Keep variables added.")
        mem("After adding keep variables")

        reassign = model.addVars(
            second_stage_indices(),
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            name="reassign",
        )
        print("Reassign variables added.")
        mem("After adding reassign variables")

        model.addConstrs(
            (
                still_avail[s, *route, om]
                == x[s, *route]
                * (
                    scenarios[route].scenario_realize[om]
                    >= self.shipments[s].weight
                )
                if route[2] == "air"
                else still_avail[s, *route, om] <= x[s, *route]
            )
            for s in self.S
            for route in self.feasible_by_shipment[s]
            for om in omega
        )

        model.addConstrs(
            
                reassign[s, *route, om] + still_avail[s, *route, om] <= 1
                for s in self.S
                for route in self.feasible_by_shipment[s]
                for om in omega
            
        )

        # Retain the first-mile/last-mile linking constraints, but build their
        # sums from pre-indexed shipment-specific arcs.
        model.addConstrs(
            
                still_avail[s, *route, om] + reassign[s, *route, om]
                == gp.quicksum(
                    still_avail[s, *air_route, om]
                    + reassign[s, *air_route, om]
                    for air_route in self.feasible_air_outgoing[s].get(route[1], ())
                )
                for s in self.S
                for route in self.feasible_ground[s]
                if route[0] == self.shipments[s].origin.node_id
                for om in omega
            
        )
        model.addConstrs(
            
                still_avail[s, *route, om] + reassign[s, *route, om]
                == gp.quicksum(
                    still_avail[s, *air_route, om]
                    + reassign[s, *air_route, om]
                    for air_route in self.feasible_air_incoming[s].get(route[0], ())
                )
                for s in self.S
                for route in self.feasible_ground[s]
                if route[1] == self.shipments[s].destination.node_id
                for om in omega
            
        )

        for om in tqdm(
            omega,
            desc="Processing scenarios",
            unit="scenario",
            disable=self.solver_quiet,
        ):
            for s in self.S:
                orig_s = self.shipments[s].origin.node_id
                dest_s = self.shipments[s].destination.node_id
                for node in self.feasible_nodes[s]:
                    flow_out = gp.quicksum(
                        still_avail[s, *route, om] + reassign[s, *route, om]
                        for route in self.feasible_outgoing[s].get(node, ())
                    )
                    flow_in = gp.quicksum(
                        still_avail[s, *route, om] + reassign[s, *route, om]
                        for route in self.feasible_incoming[s].get(node, ())
                    )
                    rhs = 1 if node == orig_s else -1 if node == dest_s else 0
                    model.addConstr(
                        flow_out - flow_in == rhs,
                        name=f"recourse_flow_{s}_{node}_om{om}",
                    )

        cost = gp.quicksum(
            reassign[s, *route, om]
            * (
                self.params.get_reassignment_cost(
                    self.params.cost_flight
                    if route[2] == "air"
                    else self.params.cost_ground
                )
                + self.params.cost_penalty_incompatibility
            )
            for s in self.S
            for route in self.feasible_by_shipment[s]
            for om in omega
        )

        return model, still_avail, reassign, cost

    def _stage_two_setup_dense_legacy(self, model, x, Omega, scenarios):
        print("Setting up second stage optimization...")

        if self.solver_quiet:
            model.Params.OutputFlag = 0
            model.Params.LogToConsole = 0

        model.Params.Threads = 8

        air_legs = [r for r in self.R if r[2] == "air"]
        print(f"Air legs: {len(air_legs)} out of {len(self.R)} total legs.")
        gnd_legs = [r for r in self.R if r[2] == "ground"]

        # for (i,j,m) in air_legs:
        #     pass
        still_avail = model.addVars(self.S, self.R, Omega, vtype=GRB.BINARY, name="keep")
        print("Keep variables added.")

        mem("After adding keep variables")

        # indices = [(s, i, j, m, om) for s in self.S for (i, j, m) in self.R for om in Omega]

        reassign = model.addVars(self.S, self.R, Omega, vtype=GRB.CONTINUOUS, lb=0.0, name="reassign")
        print("Reassign variables added.")

        mem("After adding reassign variables")

        # cost = gp.LinExpr()
        
        # precompute whether there is sufficient slack for a shipment of weight w
        # u[(shipment, leg, scenario)] = True if slack >= weight else False
        # u = [scenarios[i,j,m].scenario_realize[om] >= self.shipments[s].weight for s in self.S for (i, j, m) in air_legs for om in Omega]
        # breakpoint()
        # u = {}
        # for s in self.S:
        #     weight_s = self.shipments[s].weight
        #     for (i, j, m) in air_legs:
        #         realizations = scenarios[i, j, m].scenario_realize
        #         for om in Omega:
        #             u[(s, i, j, m, om)] = realizations[om] >= weight_s

        # mem("After precomputing u")

        # breakpoint()

        # model.addConstrs(still_avail[s, i, j, m, om] == x[s, i, j, m] * (scenarios[i, j, 'air'].scenario_realize[om] >= self.shipments[s].weight) for s in self.S for (i, j, m) in air_legs for om in Omega)
        # model.addConstrs(still_avail[s, i, j, m, om] == x[s, i, j, m] for s in self.S for (i, j, m) in gnd_legs for om in Omega)

        model.addConstrs((still_avail[s, i, j, m, om] == x[s, i, j, m] * (scenarios[i, j, m].scenario_realize[om] >= self.shipments[s].weight) if m == "air" else 
                         still_avail[s, i, j, m, om] <= x[s, i, j, m])
                         for s in self.S for (i, j, m) in self.R for om in Omega)

        # model.addConstrs(reassign[s, i, j, m, om] + x[s, i, j, m]*u[(s, i, j, m, om)] <= 1 for s in self.S for (i, j, m) in self.R for om in Omega)
        model.addConstrs(reassign[s, i, j, m, om] + still_avail[s, i, j, m, om] <= 1 for s in self.S for (i, j, m) in self.R for om in Omega)

        # origin restriction
        model.addConstrs(still_avail[s, o, i, 'ground', om] + reassign[s, o, i, 'ground', om] 
                         == gp.quicksum(still_avail[s, k, j, 'air', om] + reassign[s, k, j, 'air', om] for (k, j, _) in air_legs if k == i) 
                         for s in self.S for (o, i, _) in gnd_legs if o == self.shipments[s].origin.node_id for om in Omega)

        # destination restriction
        model.addConstrs(still_avail[s, i, d, 'ground', om] + reassign[s, i, d, 'ground', om] 
                         == gp.quicksum(still_avail[s, j, k, 'air', om] + reassign[s, j, k, 'air', om] for (j, k, _) in air_legs if k == d) 
                         for s in self.S for (i, d, _) in gnd_legs if d == self.shipments[s].destination.node_id for om in Omega)

        for om in tqdm(Omega, desc="Processing scenarios", unit="scenario", disable=self.solver_quiet):

            for s in tqdm(
                self.S,
                desc=f"Processing shipments for scenario {om}",
                unit="shipment",
                disable=self.solver_quiet,
            ):
                # weight_s = self.shipments[s].weight
                orig_s = self.shipments[s].origin.node_id
                dest_s = self.shipments[s].destination.node_id

                # for (i, j, m) in air_legs:
                #     if slack[(i, j, m)] < weight_s:
                #         keep[s, i, j, m, om].UB = 0.0
                    # else:
                    #     model.addConstr(
                    #         keep[s, i, j, m, om] <= x[s, i, j, m],
                    #         name=f"keep_constraint_{s}_{i}_{j}_{m}_om{om}",
                    #     )

                # for (i, j, m) in self.R:
                    # model.addConstr(
                    #     x[s, i, j, m] * u[(s, i, j, m, om)] + reassign[s, i, j, m, om] <= 1,
                    #     name=f"reassign_constraint_{s}_{i}_{j}_{m}_om{om}",
                    # )

                    # if m == "air":
                    #     unit_cost = (
                    #         self.params.get_reassignment_cost(self.params.cost_flight)
                    #         + self.params.cost_penalty_rejection
                    #     )
                    # else:
                    #     unit_cost = (
                    #         self.params.get_reassignment_cost(self.params.cost_ground)
                    #         + self.params.cost_penalty_rejection
                    #     )

                    # cost += (
                    #     unit_cost
                    #     * weight_s
                    #     * self.legs[(i, j, m)].distance_miles
                    #     * reassign[s, i, j, m, om]
                    # )

                for l in self.nodes:
                    flow_out = gp.quicksum(
                        still_avail[s, i, j, m, om] + reassign[s, i, j, m, om]
                        for (i, j, m) in self.R
                        if i == l
                    )
                    flow_in = gp.quicksum(
                        still_avail[s, i, j, m, om] + reassign[s, i, j, m, om]
                        for (i, j, m) in self.R
                        if j == l
                    )

                    if l == orig_s:
                        rhs = 1
                    elif l == dest_s:
                        rhs = -1
                    else:
                        rhs = 0

                    model.addConstr(flow_out - flow_in == rhs, name=f"flow_constraint_{l}")

        cost = gp.quicksum(
            reassign[s, i, j, m, om] * (self.params.get_reassignment_cost(self.params.cost_flight) + self.params.cost_penalty_incompatibility)
            if m == "air"
            else reassign[s, i, j, m, om] * (self.params.get_reassignment_cost(self.params.cost_ground) + self.params.cost_penalty_incompatibility)
            for s in self.S
            for i, j, m in self.R
            for om in Omega
        )

        return model, still_avail, reassign, cost


    def solve_sample_average(
        self,
        scenarios: list[list[UncertaintyRealization]],
    ) -> tuple[TwoStageSolution, gp.Model]:
        """
        Solve using sample average approximation (SAA) with integrated two-stage optimization.
        
        Builds a single optimization problem where first-stage x variables are shared across
        all scenarios, and second-stage recourse decisions are scenario-specific.
        
        Objective: min_x,y,z { sum c_flight * x + (1/N) * sum_n Q_n(y_n, z_n | x, xi_n) }
        
        The first-stage decision X is optimized considering actual recourse costs across
        all scenarios, not just expected penalties.
        
        Args:
            scenarios: List of scenario realizations
        
        Returns:
            TwoStageSolution with integrated first and second stage solution
        """
        # Initialize integrated model
        # model = gp.Model("SAA_TwoStage")
        # First stage setup
        Omega = list(range(len(scenarios)))

        # Build feasible arc sets R_s by OD compatibility.
        # NOTE also a temporary band aid fix
        self.feasible_by_shipment: dict[str, list[str]] = {}
        for s in self.S:
            shipment = self.shipments_by_id[s]
            if shipment.origin is None or shipment.destination is None:
                self.feasible_by_shipment[s] = self.F.copy()
                continue

            feasible = [
                f for f in self.F
                if self.flights_by_id[f].origin == shipment.origin and self.flights_by_id[f].destination == shipment.destination
            ]
            if not feasible:
                raise ValueError(
                    f"No feasible arcs for shipment {s} with OD ({shipment.origin}, {shipment.destination})."
                )
            self.feasible_by_shipment[s] = feasible

        model, x, first_stage_cost = self.stage_one_setup(gp.Model("SAA_TwoStage"))
        
        # ========== First-stage variables (shared across scenarios) ==========
        # x = model.addVars(self.S, self.F, vtype=GRB.BINARY, name="x")
        
        # ========== Second-stage variables (scenario-specific) ==========
        # keep = model.addVars(self.S, self.F, Omega, vtype=GRB.BINARY, name="keep")
        # reassign = model.addVars(self.S, self.F, Omega, vtype=GRB.BINARY, name="reassign")
        
        # ========== Objective: First-stage + Average recourse ==========
        # first_stage_cost = gp.quicksum(
        #     self.flights_by_id[f].cost_flight * x[s, f]
        #     for s in self.S
        #     for f in self.F
        # )
        
        # Second stage setup
        model, keep, reassign, recourse_costs = self.stage_two_setup(model, x, Omega, scenarios)

        # Scenario-specific recourse costs
        # recourse_costs = gp.quicksum(
        #     self.params.get_reassignment_cost(self.flights_by_id[f].cost_flight) * reassign[s, f, om]
        #     for s in self.S
        #     for f in self.F
        #     for om in Omega
        # )
        
        # Add incompatibility penalties for all scenarios
        for om in Omega:
            scenario = scenarios[om]
            uncertainty_dict = {
                (ur.shipment_id, ur.flight_id): ur
                for ur in scenario
            }
            
            for s in self.S:
                for f in self.F:
                    ur = uncertainty_dict.get((s, f))
                    if ur is not None and not ur.compatibility:
                        recourse_costs += (
                            (keep[s, f, om] + reassign[s, f, om]) 
                            * self.params.cost_penalty_incompatibility
                        )
        
        # Average recourse cost across scenarios
        avg_recourse = recourse_costs / len(scenarios)
        
        model.setObjective(first_stage_cost + avg_recourse, GRB.MINIMIZE)
        
        # ========== First-stage constraints ==========
        # model.addConstrs(
        #     (gp.quicksum(x[s, f] for f in self.feasible_by_shipment[s]) == 1 for s in self.S),
        #     name="first_stage_assign"
        # )

        # NOTE these are bandaid fixes, will properly write in flow conservation constrs later
        # Disallow ineligible arcs outside R_s.
        # model.addConstrs(
        #     (x[s, f] == 0 for s in self.S for f in self.F if f not in self.feasible_by_shipment[s]),
        #     name="first_stage_ineligible_arc"
        # )
        
        '''========== Second-stage constraints (scenario-specific) ==========
        for om in Omega:
            scenario = scenarios[om]
            uncertainty_dict = {
                (ur.shipment_id, ur.flight_id): ur
                for ur in scenario
            }
            
            # Exactly one recourse option per shipment per scenario
            model.addConstrs(
                (
                    gp.quicksum(keep[s, f, om] + reassign[s, f, om] for f in self.feasible_by_shipment[s]) == 1
                    for s in self.S
                ),
                name=f"one_option_om{om}"
            )

            # NOTE once again a band aid fix
            # Disallow recourse on ineligible arcs outside R_s.
            model.addConstrs(
                (keep[s, f, om] == 0 for s in self.S for f in self.F if f not in self.feasible_by_shipment[s]),
                name=f"keep_ineligible_om{om}"
            )
            model.addConstrs(
                (reassign[s, f, om] == 0 for s in self.S for f in self.F if f not in self.feasible_by_shipment[s]),
                name=f"reassign_ineligible_om{om}"
            )
            
            # Can only keep if originally assigned AND accepted in this scenario
            for s in self.S:
                for f in self.F:
                    ur = uncertainty_dict.get((s, f))
                    acceptance_val = 1.0 if ur and ur.acceptance else 0.0
                    
                    model.addConstr(
                        keep[s, f, om] == x[s, f] * acceptance_val,
                        name=f"keep_constraint_{s}_{f}_om{om}"
                    )
            
            # Reassignment constraint
            # for s in S:
            #     # for f in F:
            #         model.addConstr(
            #             gp.quicksum(reassign[s, f, om] for f in F) >= gp.quicksum(x[s, f] for f in F) - gp.quicksum(keep[s, f, om] for f in F),
            #             name=f"reassign_constraint_{s}_{f}_om{om}"
            #         )

            model.addConstrs((reassign[s, f, om] + x[s,f] <= 1 for s in self.S for f in self.F), 
                name=f"reassign_diff_arc_om{om}"
            )'''
        
        # Optimize
        model.optimize()
        
        # ========== Extract solution ==========
        # First-stage assignments
        first_stage_assignments = {}
        if model.Status == GRB.OPTIMAL:
            for s in self.S:
                for f in self.F:
                    if x[s, f].X > 1e-6:
                        first_stage_assignments[(s, f)] = float(x[s, f].X)
        
        # Second-stage aggregate solution (averaged across scenarios)
        keep_assignments = {}
        reassignments = {}
        if model.Status == GRB.OPTIMAL:
            for s in self.S:
                for f in self.F:
                    keep_val = sum(keep[s, f, om].X for om in Omega) / len(Omega)
                    reassign_val = sum(reassign[s, f, om].X for om in Omega) / len(Omega)
                    if keep_val > 1e-6:
                        keep_assignments[(s, f)] = keep_val
                    if reassign_val > 1e-6:
                        reassignments[(s, f)] = reassign_val
        
        status_map = {GRB.OPTIMAL: "Optimal", GRB.SUBOPTIMAL: "Suboptimal", GRB.INFEASIBLE: "Infeasible"}
        status_str = status_map.get(model.Status, f"Status {model.Status}")
        
        return TwoStageSolution(
            first_stage=FirstStageSolution(
                assignments=first_stage_assignments,
                objective_value=float(model.ObjVal - avg_recourse.getValue()) if model.Status == GRB.OPTIMAL else float('inf'),
                status=status_str,
            ),
            second_stage=SecondStageSolution(
                keep_assignments=keep_assignments,
                reassignments=reassignments,
                objective_value=float(avg_recourse.getValue()) if model.Status == GRB.OPTIMAL else float('inf'),
                status=status_str,
            ),
            expected_total_cost=float(model.ObjVal) if model.Status == GRB.OPTIMAL else float('inf'),
        ), model


    def solve_myopic(self,
        scenarios: list[list[UncertaintyRealization]],
    ) -> tuple[TwoStageSolution, gp.Model]:
        # Build feasible arc sets R_s by OD compatibility.
        Omega = list(range(len(scenarios)))
        self.feasible_by_shipment = {}
        for s in self.S:
            shipment = self.shipments_by_id[s]
            if shipment.origin is None or shipment.destination is None:
                self.feasible_by_shipment[s] = self.F.copy()
                continue

            feasible = [
                f for f in self.F
                if self.flights_by_id[f].origin == shipment.origin and self.flights_by_id[f].destination == shipment.destination
            ]
            if not feasible:
                raise ValueError(
                    f"No feasible arcs for shipment {s} with OD ({shipment.origin}, {shipment.destination})."
                )
            self.feasible_by_shipment[s] = feasible

        # Stage 1: myopic first-stage objective only.
        model1, x, first_stage_cost = self.stage_one_setup(gp.Model("Myopic_Stage1"))
        model1.setObjective(first_stage_cost, GRB.MINIMIZE)
        model1.optimize()

        # Stage 2: fix x from stage 1 and optimize recourse.
        model2 = gp.Model("Myopic_Stage2")
        x2 = model2.addVars(self.S, self.F, vtype=GRB.BINARY, name="x")
        model2.addConstrs((x2[s, f] == x[s, f].X for s in self.S for f in self.F), name="fix_x_from_stage1")

        model2, keep, reassign, non_recourse_costs = self.stage_two_setup(model2, x2, Omega, scenarios)
        recourse_costs = non_recourse_costs / len(scenarios)

        for om in Omega:
            scenario = scenarios[om]
            uncertainty_dict = {
                (ur.shipment_id, ur.flight_id): ur
                for ur in scenario
            }

            for s in self.S:
                for f in self.F:
                    ur = uncertainty_dict.get((s, f))
                    if ur is not None and not ur.compatibility:
                        recourse_costs += (
                            (keep[s, f, om] + reassign[s, f, om])
                            * self.params.cost_penalty_incompatibility
                        )

        model2.setObjective(recourse_costs, GRB.MINIMIZE)
        model2.optimize()

        # ========== Extract solution ==========
        first_stage_assignments = {}
        if model1.Status == GRB.OPTIMAL:
            for s in self.S:
                for f in self.F:
                    if x[s, f].X > 1e-6:
                        first_stage_assignments[(s, f)] = float(x[s, f].X)

        keep_assignments = {}
        reassignments = {}
        if model2.Status == GRB.OPTIMAL:
            for s in self.S:
                for f in self.F:
                    keep_val = sum(keep[s, f, om].X for om in Omega) / len(Omega)
                    reassign_val = sum(reassign[s, f, om].X for om in Omega) / len(Omega)
                    if keep_val > 1e-6:
                        keep_assignments[(s, f)] = keep_val
                    if reassign_val > 1e-6:
                        reassignments[(s, f)] = reassign_val

        status_map = {GRB.OPTIMAL: "Optimal", GRB.SUBOPTIMAL: "Suboptimal", GRB.INFEASIBLE: "Infeasible"}
        status_str = status_map.get(model1.Status, f"Status {model1.Status}")

        return TwoStageSolution(
            first_stage=FirstStageSolution(
                assignments=first_stage_assignments,
                objective_value=float(model1.ObjVal) if model1.Status == GRB.OPTIMAL else float('inf'),
                status=status_str,
            ),
            second_stage=SecondStageSolution(
                keep_assignments=keep_assignments,
                reassignments=reassignments,
                objective_value=float(model2.ObjVal) if model2.Status == GRB.OPTIMAL else float('inf'),
                status=status_str,
            ),
            expected_total_cost=float(model1.ObjVal + model2.ObjVal) if model2.Status == GRB.OPTIMAL else float('inf'),
        ), model2


def plot_shipment_assignment(
    shipment_id: str,
    assignments: dict[tuple[str, str, str, str], float],
    legs: dict[tuple[str, str, str], LegOption],
    output_path: str | Path,
) -> None:
    """Plot the assigned legs for one shipment on a Robinson projection."""
    selected_legs = [
        (legs[(origin, destination, mode)], value)
        for (assigned_shipment, origin, destination, mode), value in assignments.items()
        if assigned_shipment == shipment_id and value > 1e-6
    ]
    if not selected_legs:
        raise ValueError(f"No assigned legs found for shipment {shipment_id!r}.")

    mode_colors = {"air": "#1769aa", "ground": "#d97706"}
    points = {
        endpoint.node_id: endpoint
        for leg, _ in selected_legs
        for endpoint in (leg.origin, leg.destination)
    }

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=-20))
    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="black", linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="#e6f7ff")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.6)

    gridlines = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {"size": 9}
    gridlines.ylabel_style = {"size": 9}

    for leg, value in selected_legs:
        ax.plot(
            [leg.origin.longitude, leg.destination.longitude],
            [leg.origin.latitude, leg.destination.latitude],
            color=mode_colors.get(leg.mode, "#4b5563"),
            linewidth=2.0 + value,
            alpha=0.85,
            transform=ccrs.Geodetic(),
            zorder=3,
        )

    ax.scatter(
        [node.longitude for node in points.values()],
        [node.latitude for node in points.values()],
        color="darkred",
        s=35,
        alpha=0.9,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )
    for node in points.values():
        ax.text(
            node.longitude,
            node.latitude,
            node.node_id,
            transform=ccrs.PlateCarree(),
            fontsize=8,
            color="darkred",
            ha="left",
            va="bottom",
            zorder=6,
        )

    ax.set_global()
    ax.set_title(f"Assigned legs for shipment {shipment_id}", fontsize=16, pad=16)
    ax.legend(
        handles=[
            Line2D([0], [0], color=mode_colors["air"], lw=3, label="Air"),
            Line2D([0], [0], color=mode_colors["ground"], lw=3, label="Ground"),
            Line2D([0], [0], color="darkred", marker="o", linestyle="None", markersize=7, label="Node"),
        ],
        loc="lower left",
        frameon=True,
    )
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "FirstStageSolution",
    "LegOption",
    "Node",
    "SecondStageSolution",
    "Shipment",
    "StochasticOptimizationParameters",
    "TwoStageSolution",
    "TwoStageSolver",
    "UncertaintyRealization",
]

if __name__ == "__main__":
    from data_processing import *
    # pull real shipment data and legs from the data processing module
    t100_processor = T100DataProcessing()
    t100_data = t100_processor.filter_data()
    t100_data = t100_processor._geolocate_nodes(t100_data)
    t100_data = t100_processor._calculate_distance(t100_data)

    shipping_processor = DataProcessing()
    shipping_processor.align_from(t100_processor)
    ship_data = shipping_processor.load_shipping_data()
    ship_data = shipping_processor._geolocate_nodes(ship_data)
    ship_data = shipping_processor._calculate_distance(ship_data)

    ship_overlap = shipping_processor.get_od_overlap(ship_data, t100_data)
    ship_overlap = ship_overlap.with_columns([
        pl.col("AW (lbs)").cast(pl.Float64, strict=False).alias("AW (lbs)"),
        pl.col("Commercial Cost for First Mile").cast(pl.Float64, strict=False).alias("Commercial Cost for First Mile"),
        pl.col("Commercial Cost for Last Mile").cast(pl.Float64, strict=False).alias("Commercial Cost for Last Mile"),
    ])

    import hashlib
    import json

    import geonamescache
    import requests
    from sklearn.neighbors import BallTree

    MILES_PER_METER = 0.000621371
    EARTH_RADIUS_MILES = 3958.8
    NEARBY_CITY_RADIUS_MILES = 100.0
    MAX_CITY_ATTEMPTS = 10
    MAX_GROUND_DISTANCE_MILES = 500.0

    legs: dict[tuple[str, str, str], LegOption] = {}
    nodes: dict[str, Node] = {}

    # Build the airport network first so all subsequent shipment endpoints can
    # refer to the canonical airport Node objects.
    for leg in t100_data.iter_rows(named=True):
        if leg["ORIGIN"] not in nodes:
            nodes[leg["ORIGIN"]] = Node(
                node_id=leg["ORIGIN"],
                latitude=leg["Origin_Lat"],
                longitude=leg["Origin_Lon"],
                connection_type="air"
            )

        if leg["DEST"] not in nodes:
            nodes[leg["DEST"]] = Node(
                node_id=leg["DEST"],
                latitude=leg["Destination_Lat"],
                longitude=leg["Destination_Lon"],
                connection_type="air"
            )
        
        legs[(leg['ORIGIN'], leg['DEST'], 'air')] = LegOption(
            route_id=f'{leg["ORIGIN"]}_{leg["DEST"]}',
            origin=nodes[leg["ORIGIN"]],
            destination=nodes[leg["DEST"]],
            distance_miles=leg["DISTANCE"],
            mode="air",
            mu_slack=leg["MU_SLACK"],
            sigma_slack=leg["SIGMA_SLACK"],
        )  

    airport_ids = sorted(nodes)
    airport_coordinates = {
        airport_id: (nodes[airport_id].latitude, nodes[airport_id].longitude)
        for airport_id in airport_ids
    }
    airport_tree = BallTree(
        np.radians([airport_coordinates[airport_id] for airport_id in airport_ids]),
        metric="haversine",
    )

    raw_cities = geonamescache.GeonamesCache().get_cities().values()
    cities = [
        {
            "lat": city["latitude"],
            "lon": city["longitude"],
            "countrycode": city["countrycode"],
        }
        for city in raw_cities
    ]
    city_tree = BallTree(
        np.radians([[city["lat"], city["lon"]] for city in cities]),
        metric="haversine",
    )
    airport_country_codes: dict[str, str] = {}

    def airport_country(airport_id: str) -> str:
        if airport_id not in airport_country_codes:
            lat, lon = airport_coordinates[airport_id]
            city_index = city_tree.query(np.radians([[lat, lon]]), k=1)[1][0][0]
            airport_country_codes[airport_id] = cities[city_index]["countrycode"]
        return airport_country_codes[airport_id]

    def nearby_city_indices(airport_id: str) -> list[int]:
        lat, lon = airport_coordinates[airport_id]
        indices, _distances = city_tree.query_radius(
            np.radians([[lat, lon]]),
            r=NEARBY_CITY_RADIUS_MILES / EARTH_RADIUS_MILES,
            return_distance=True,
            sort_results=True,
        )
        matching = [
            city_index
            for city_index in indices[0]
            if cities[city_index]["countrycode"] == airport_country(airport_id)
        ]
        if not matching:
            raise ValueError(
                f"No nearby cities found for airport {airport_id} within "
                f"{NEARBY_CITY_RADIUS_MILES:.0f} miles in the same country."
            )
        return matching

    cache_path = Path(__file__).parent / "cache/routing_cache.json"
    routing_cache: dict[str, dict] = (
        json.loads(cache_path.read_text()) if cache_path.exists() else {}
    )
    cache_dirty = [False]

    def route_cache_key(lat1: float, lon1: float, lat2: float, lon2: float) -> str:
        raw = f"{lat1:.5f},{lon1:.5f},{lat2:.5f},{lon2:.5f}"
        return hashlib.sha1(raw.encode()).hexdigest()

    def check_drivable(
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
        base: str = "https://router.project-osrm.org",
        timeout: int = 10,
    ) -> dict:
        """Return a cached OSRM driving-route feasibility result."""
        key = route_cache_key(lat1, lon1, lat2, lon2)
        if key in routing_cache:
            return routing_cache[key]

        url = f"{base}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            if data.get("code") == "Ok" and data.get("routes"):
                route = data["routes"][0]
                result = {
                    "feasible": True,
                    "distance_m": route["distance"],
                    "duration_s": route["duration"],
                }
            else:
                result = {"feasible": False, "distance_m": None, "duration_s": None}
        except requests.RequestException:
            result = {"feasible": False, "distance_m": None, "duration_s": None}

        routing_cache[key] = result
        cache_dirty[0] = True
        return result

    # Each group receives one synthetic endpoint.  The airport component makes
    # the grouping safe even if an NGO unexpectedly appears at multiple airports.
    endpoint_groups: dict[tuple[str, str, str], None] = {}
    shipment_rows = list(ship_overlap.iter_rows(named=True))
    for shipment in shipment_rows:
        ngo_id = str(shipment["NGO ID"])
        if shipment["Commercial Cost for First Mile"] is not None:
            endpoint_groups[(ngo_id, shipment["ORIGIN"], "O")] = None
        if shipment["Commercial Cost for Last Mile"] is not None:
            endpoint_groups[(ngo_id, shipment["DEST"], "D")] = None

    endpoint_nodes: dict[tuple[str, str, str], Node] = {}
    validated_endpoint_routes: dict[tuple[str, str, str], dict] = {}
    for endpoint_key in endpoint_groups:
        ngo_id, airport_id, side = endpoint_key
        airport_lat, airport_lon = airport_coordinates[airport_id]
        route_result = None
        selected_city = None
        for city_index in nearby_city_indices(airport_id)[:MAX_CITY_ATTEMPTS]:
            city = cities[city_index]
            result = check_drivable(city["lat"], city["lon"], airport_lat, airport_lon)
            if result["feasible"]:
                selected_city = city
                route_result = result
                break

        if selected_city is None or route_result is None:
            raise ValueError(
                f"No drivable GeoNames city found for endpoint {endpoint_key} after "
                f"{MAX_CITY_ATTEMPTS} attempts."
            )

        node_id = f"{ngo_id}_{airport_id}_{side}"
        endpoint_node = Node(
            node_id=node_id,
            latitude=selected_city["lat"],
            longitude=selected_city["lon"],
            connection_type="gnd",
        )
        nodes[node_id] = endpoint_node
        endpoint_nodes[endpoint_key] = endpoint_node
        validated_endpoint_routes[endpoint_key] = route_result

    shipments: dict[str, Shipment] = {}
    for shipment in shipment_rows:
        ngo_id = str(shipment["NGO ID"])
        origin = (
            endpoint_nodes[(ngo_id, shipment["ORIGIN"], "O")]
            if shipment["Commercial Cost for First Mile"] is not None
            else nodes[shipment["ORIGIN"]]
        )
        destination = (
            endpoint_nodes[(ngo_id, shipment["DEST"], "D")]
            if shipment["Commercial Cost for Last Mile"] is not None
            else nodes[shipment["DEST"]]
        )
        shipments[shipment["Shipment ID"]] = Shipment(
            shipment_id=shipment["Shipment ID"],
            weight=shipment["AW (lbs)"],
            origin=origin,
            destination=destination,
        )

    # BallTree removes airport pairs that cannot possibly meet the road-distance
    # threshold before we make an OSRM request.
    for endpoint_node in endpoint_nodes.values():
        airport_indices = airport_tree.query_radius(
            np.radians([[endpoint_node.latitude, endpoint_node.longitude]]),
            r=MAX_GROUND_DISTANCE_MILES / EARTH_RADIUS_MILES,
        )[0]
        for airport_index in airport_indices:
            airport_id = airport_ids[airport_index]
            airport = nodes[airport_id]
            result = check_drivable(
                endpoint_node.latitude,
                endpoint_node.longitude,
                airport.latitude,
                airport.longitude,
            )
            if not result["feasible"]:
                continue

            distance_miles = result["distance_m"] * MILES_PER_METER
            if distance_miles > MAX_GROUND_DISTANCE_MILES:
                continue
            legs[(endpoint_node.node_id, airport_id, "ground")] = LegOption(
                route_id=f"{endpoint_node.node_id}_{airport_id}_ground",
                origin=endpoint_node,
                destination=airport,
                distance_miles=distance_miles,
                mode="ground",
            )
            legs[(airport_id, endpoint_node.node_id, "ground")] = LegOption(
                route_id=f"{airport_id}_{endpoint_node.node_id}_ground",
                origin=airport,
                destination=endpoint_node,
                distance_miles=distance_miles,
                mode="ground",
            )

    if cache_dirty[0]:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(routing_cache))

    def plot_all_nodes_gnd():
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=-20))
        ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="black", linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, facecolor="#e6f7ff")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.6)

        gridlines = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)
        gridlines.top_labels = False
        gridlines.right_labels = False
        gridlines.xlabel_style = {"size": 9}
        gridlines.ylabel_style = {"size": 9}

        for node in nodes.values():
            if node.connection_type == "gnd":
                ax.scatter(
                    node.longitude,
                    node.latitude,
                    color="darkred",
                    s=35,
                    alpha=0.9,
                    transform=ccrs.PlateCarree(),
                    zorder=5,
                )
                ax.text(
                    node.longitude,
                    node.latitude,
                    node.node_id,
                    transform=ccrs.PlateCarree(),
                    fontsize=8,
                    color="darkred",
                    ha="left",
                    va="bottom",
                    zorder=6,
                )

        ax.set_global()
        ax.set_title("All Nodes (Airports and Non-Airport Nodes)", fontsize=16, pad=16)
        fig.tight_layout()
        output_path = Path(__file__).parent / Path("output/figures") / "all_nodes_map.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    plot_all_nodes_gnd()
    # breakpoint()
    # from itertools import islice
    # shipments = dict(islice(shipments.items(), 5))  # Limit to first 10 shipments for testing

    # TODO see if theres a more efficient way to handle the scenario generation, vectorize as needed
    """Load the theta values from a .npz file."""
    script_dir = Path(__file__).parent
    npz = np.load(script_dir / 'psi_bar.npz')
    psi_bar = npz['psi_bar']
    n_design = int(npz["n_design"])

    psi_alpha_bar, psi_beta_bar = psi_bar[:n_design], psi_bar[n_design:]

    rng = np.random.default_rng(42)

    def alpha_beta(design_matr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        log_alpha = (design_matr @ psi_alpha_bar) / np.sqrt(n_design)
        log_beta  = (design_matr @ psi_beta_bar)  / np.sqrt(n_design)
        return np.exp(log_alpha), np.exp(log_beta)

    def draw_down(design_matr: np.ndarray, mu_z: np.ndarray, num_samples: int) -> np.ndarray:
        scaling_factors = mu_z

        alpha_i, beta_i = alpha_beta(design_matr)
        return (rng.gamma(alpha_i, beta_i, num_samples) * scaling_factors).astype(np.float32)

    def generate_scenarios(num_scenarios: int) -> dict[tuple[str, str, str], UncertaintyRealization]:
        scenarios = dict[tuple[str, str, str], UncertaintyRealization]()
        for (i, j, m), leg in legs.items():
            if m != "air":
                continue
            slack_samples = t100_processor._truncated_slack_samples(np.array(leg.mu_slack), np.array(leg.sigma_slack), num_scenarios)
            epsilon = draw_down(leg.u_vec, leg.mu_slack, num_scenarios)

            slax = np.maximum((np.array(slack_samples).flatten() - epsilon), 0.0)
            scenarios[(i, j, m)] = UncertaintyRealization(leg=leg, num_scenarions=len(slax), scenario_realize=slax)

        return scenarios

    num_scenarios = 100
    scn = generate_scenarios(num_scenarios)
    # breakpoint()
    # Test first stage
    solver = TwoStageSolver(
        # # shipments={'s1':
        #     Shipment(shipment_id="S1", weight=100, origin="EWR", destination="MUC"),
        #     's2': Shipment(shipment_id="S2", weight=200, origin="DEN", destination="MUC"),
        # },
        shipments = shipments,
        legs=legs,
        params=StochasticOptimizationParameters(
            cost_flight=4.0,
            cost_ground=2.0,
            cost_penalty_incompatibility=5.0,
        ),
    )
    # breakpoint()
    model, x, cost = solver.stage_one_setup(gp.Model("Test_Stage1"))

    model, keep, reassign, recourse_costs = solver.stage_two_setup(model, x, Omega=range(num_scenarios), scenarios=scn)

    model.setObjective(cost + recourse_costs/num_scenarios, GRB.MINIMIZE)
    model.optimize()

    if model.Status == GRB.OPTIMAL:
        print("Optimal solution found.")
        selected_assignments = {}
        for s in solver.S:
            for (i, j, m) in solver.R:
                if x[s, i, j, m].X > 1e-6:
                    selected_assignments[(s, i, j, m)] = float(x[s, i, j, m].X)
                    print(f"Shipment {s} assigned to leg ({i}, {j}, {m}) with value {x[s, i, j, m].X}")
        if not selected_assignments:
            raise ValueError("No assignments found in the optimal solution.")
        # target_shipment = "23-0126-44"
        target_shipment = "23-0247"
        breakpoint()
        if target_shipment in shipments:
            map_path = Path(__file__).parent / Path("output/figures") / f"shipment_{target_shipment}_assignment_map.png"
            plot_shipment_assignment(target_shipment, selected_assignments, legs, map_path)
            print(f"Saved shipment map to {map_path}")

    elif model.status == GRB.INFEASIBLE:
        print("Model is infeasible.")
        model.computeIIS()
        print("IIS computed. The following constraints are in the IIS:")

        model.write("model_IIS.ilp")

