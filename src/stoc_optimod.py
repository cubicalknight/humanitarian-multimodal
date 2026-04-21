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

from dataclasses import dataclass
from typing import Sequence

import gurobipy as gp
from gurobipy import GRB


# ============================================================================
# Data Structures
# ============================================================================


@dataclass(frozen=True)
class FlightOption:
    """Represents a single flight option (route + airline)."""
    route_id: str
    airline_id: str
    origin: str
    destination: str
    distance_miles: float
    cost_flight: float  # c_flight: upfront flight cost


@dataclass(frozen=True)
class Shipment:
    """Represents a single shipment to be assigned."""
    shipment_id: str
    weight_kg: float
    origin: str | None = None
    destination: str | None = None
    pallets: float = 0.0
    equivalent_cost: float = 0.0
    commodity: str | None = None


@dataclass(frozen=True)
class UncertaintyRealization:
    """
    Represents a realization of uncertainty ξ = {A, B}.
    
    A_{s,(i,j),a}: binary indicator of carrier acceptance
    B_{s,(i,j),a}: binary indicator of aircraft compatibility (given acceptance)
    """
    shipment_id: str
    flight_id: str
    acceptance: bool
    compatibility: bool


@dataclass
class StochasticOptimizationParameters:
    """Parameters for the two-stage stochastic program."""
    
    # Cost parameters
    cost_penalty_rejection: float  # c_pt: penalty for carrier rejection
    cost_penalty_incompatibility: float  # Penalty for aircraft incompatibility
    
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
        return original_cost * 0.5  # Default: 50% markup on reassignment


@dataclass
class FirstStageSolution:
    """Solution from the first stage optimization."""
    assignments: dict[tuple[str, str], float]  # (shipment_id, flight_id) -> probability
    objective_value: float
    status: str
    

@dataclass
class SecondStageSolution:
    """Solution from the second stage optimization."""
    keep_assignments: dict[tuple[str, str], float]  # (shipment_id, flight_id) -> keep indicator
    reassignments: dict[tuple[str, str], float]  # (shipment_id, flight_id) -> reassign indicator
    objective_value: float
    status: str


@dataclass
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
        shipments: Sequence[Shipment],
        flights: Sequence[FlightOption],
        params: StochasticOptimizationParameters,
        solver_quiet: bool = False,
    ):
        self.shipments = shipments
        self.flights = flights
        self.params = params
        self.shipments_by_id: dict[str, Shipment]
        self.flights_by_id: dict[str, FlightOption]

        # Build keyed lookup maps from the sequence inputs provided by main.
        self.shipments_by_id = {s.shipment_id: s for s in self.shipments}
        self.S = [s.shipment_id for s in self.shipments]

        self.flights_by_id = {f"{f.route_id}_{f.airline_id}": f for f in self.flights}
        self.F = list(self.flights_by_id.keys())

        self.solver_quiet = solver_quiet


    def stage_one_setup(self, model):
        if self.solver_quiet:
            model.Params.OutputFlag = 0
            model.Params.LogToConsole = 0

        model.Params.Threads = 8

        x = model.addVars(self.S, self.F, vtype=GRB.BINARY, name="x")

        model.addConstrs(
            (gp.quicksum(x[s, f] for f in self.feasible_by_shipment[s]) == 1 for s in self.S),
            name="first_stage_assign"
        )

        # NOTE these are bandaid fixes, will properly write in flow conservation constrs later
        # Disallow ineligible arcs outside R_s.
        model.addConstrs(
            (x[s, f] == 0 for s in self.S for f in self.F if f not in self.feasible_by_shipment[s]),
            name="first_stage_ineligible_arc"
        )

        cost = gp.quicksum(
            self.flights_by_id[f].cost_flight * x[s, f]
            for s in self.S 
            for f in self.F)

        return model, x, cost
    

    def stage_two_setup(self, model, x, Omega, scenarios):
        if self.solver_quiet:
            model.Params.OutputFlag = 0
            model.Params.LogToConsole = 0

        model.Params.Threads = 8

        keep = model.addVars(self.S, self.F, Omega, vtype=GRB.BINARY, name="keep")
        reassign = model.addVars(self.S, self.F, Omega, vtype=GRB.BINARY, name="reassign")

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

            model.addConstrs((reassign[s, f, om] + x[s,f] <= 1 for s in self.S for f in self.F), 
                name=f"reassign_diff_arc_om{om}"
            )

        cost = gp.quicksum(
            self.params.get_reassignment_cost(self.flights_by_id[f].cost_flight) * reassign[s, f, om]
            for s in self.S
            for f in self.F
            for om in Omega
        )
        
        return model, keep, reassign, cost


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

        model2, keep, reassign, recourse_costs = self.stage_two_setup(model2, x2, Omega, scenarios)

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


__all__ = [
    "FlightOption",
    "Shipment",
    "UncertaintyRealization",
    "StochasticOptimizationParameters",
    "FirstStageSolution",
    "SecondStageSolution",
    "TwoStageSolution",
    "TwoStageSolver",
]

