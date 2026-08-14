"""
Multi-Commodity Network Flow Optimization for Humanitarian Logistics.
Implements a two-stage stochastic program using Sample Average Approximation (SAA).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import random

import gurobipy as gp
from gurobipy import GRB


# ============================================================================
# Data Structures
# ============================================================================

@dataclass(frozen=True)
class TransportEdge:
    """Represents a single directed edge (flight or ground leg) in the multigraph."""
    edge_id: str  # e.g., "W1_A1_Gnd" or "A1_A3_AirA"
    origin: str
    destination: str
    carrier_id: str
    distance_miles: float
    cost_leg: float  # Upfront transport cost per kg


@dataclass(frozen=True)
class Shipment:
    """Represents a commodity to be routed through the network."""
    shipment_id: str
    weight_kg: float
    origin: str
    destination: str


@dataclass(frozen=True)
class UncertaintyRealization:
    """
    Scalar realization of uncertainty U = I(z >= w_s).
    True if the latent slack on this edge is sufficient for this shipment.
    """
    shipment_id: str
    edge_id: str
    acceptance: bool


@dataclass
class StochasticOptimizationParameters:
    cost_penalty_rejection: float  # c_pt: penalty for carrier rejection
    cost_reassignment: float | None = None  

    def get_reassignment_cost(self, original_cost: float, new_cost: float | None = None) -> float:
        """Difference (c_leg' - c_leg)"""
        if new_cost is not None:
            return max(0.0, new_cost - original_cost)
        if self.cost_reassignment is not None:
            return self.cost_reassignment
        return original_cost * 0.5


# ============================================================================
# Solver (No logic changes, just variable renames for clarity)
# ============================================================================

class NetworkFlowTwoStageSolver:
    def __init__(
        self,
        shipments: Sequence[Shipment],
        edges: Sequence[TransportEdge],
        params: StochasticOptimizationParameters,
        solver_quiet: bool = True,
    ):
        self.shipments = shipments
        self.edges = edges
        self.params = params
        self.solver_quiet = solver_quiet
        
        # Lookups
        self.shipments_by_id = {s.shipment_id: s for s in shipments}
        self.edges_by_id = {e.edge_id: e for e in edges}
        
        self.S = list(self.shipments_by_id.keys())
        self.E = list(self.edges_by_id.keys())
        
        # Extract unique nodes from edges
        self.nodes = set()
        for e in self.edges:
            self.nodes.add(e.origin)
            self.nodes.add(e.destination)

    def _add_flow_conservation(self, model, flow_vars, s_id, scenario_suffix=""):
        """
        Adds strict MCNFP flow conservation (mass balance) for a given shipment.
        Sum(Out) - Sum(In) = 1 (Origin), -1 (Dest), 0 (Transit).
        """
        shipment = self.shipments_by_id[s_id]
        
        for node in self.nodes:
            # Sum of flow leaving the node
            out_flow = gp.quicksum(
                flow_vars[e_id] for e_id, e in self.edges_by_id.items() if e.origin == node
            )
            # Sum of flow entering the node
            in_flow = gp.quicksum(
                flow_vars[e_id] for e_id, e in self.edges_by_id.items() if e.destination == node
            )

            # Node Divergence
            if node == shipment.origin:
                model.addConstr(out_flow - in_flow == 1, name=f"flow_orig_{s_id}_{node}{scenario_suffix}")
            elif node == shipment.destination:
                model.addConstr(out_flow - in_flow == -1, name=f"flow_dest_{s_id}_{node}{scenario_suffix}")
            else:
                model.addConstr(out_flow - in_flow == 0, name=f"flow_trans_{s_id}_{node}{scenario_suffix}")


    def build_integrated_saa_model(self, scenarios: list[list[UncertaintyRealization]]):
        model = gp.Model("Humanitarian_Logistics_MCNFP")
        if self.solver_quiet:
            model.Params.OutputFlag = 0
        model.Params.Threads = 8

        Omega = list(range(len(scenarios)))

        # ================= STAGE 1 =================
        x = model.addVars(self.S, self.E, vtype=GRB.BINARY, name="x")

        # Nominal Flow Conservation
        for s in self.S:
            flow_vars = {e: x[s, e] for e in self.E}
            self._add_flow_conservation(model, flow_vars, s, scenario_suffix="_stage1")

        first_stage_cost = gp.quicksum(
            self.edges_by_id[e].cost_leg * self.shipments_by_id[s].weight_kg * x[s, e]
            for s in self.S for e in self.E
        )

        # ================= STAGE 2 =================
        keep = model.addVars(self.S, self.E, Omega, vtype=GRB.BINARY, name="keep")
        reassign = model.addVars(self.S, self.E, Omega, vtype=GRB.BINARY, name="reassign")
        
        recourse_cost = gp.LinExpr()

        for om in Omega:
            scenario = scenarios[om]
            uncertainty_dict = {(ur.shipment_id, ur.edge_id): ur.acceptance for ur in scenario}

            for s in self.S:
                # Recourse Flow Conservation (The kept + reassigned arcs must form a valid path)
                recourse_flow_vars = {e: keep[s, e, om] + reassign[s, e, om] for e in self.E}
                self._add_flow_conservation(model, recourse_flow_vars, s, scenario_suffix=f"_om{om}")

                for e in self.E:
                    # Look up physical feasibility realization U(omega)
                    # Note: We assume Ground ("Gnd") links always have 100% acceptance/slack in this toy sim
                    if "Gnd" in e:
                        u_val = 1.0
                    else:
                        u_val = 1.0 if uncertainty_dict.get((s, e), False) else 0.0

                    # 1. Keep Constraint: Must have chosen it in Stage 1 AND it must have slack
                    model.addConstr(keep[s, e, om] <= x[s, e])
                    model.addConstr(keep[s, e, om] <= u_val)
                    
                    # 2. Reassign Constraint: Cannot reassign to the exact same leg you were rejected from
                    model.addConstr(reassign[s, e, om] + x[s, e] <= 1)

                    # Tally recourse costs for this scenario
                    weight = self.shipments_by_id[s].weight_kg
                    c_diff = self.params.get_reassignment_cost(self.edges_by_id[e].cost_leg)
                    penalty = self.params.cost_penalty_rejection
                    
                    recourse_cost += (c_diff + penalty) * weight * reassign[s, e, om]

        # Objective: Min(Stage 1 Cost + Expected Recourse Cost)
        avg_recourse_cost = recourse_cost / len(Omega)
        model.setObjective(first_stage_cost + avg_recourse_cost, GRB.MINIMIZE)

        return model, x, keep, reassign


# ============================================================================
# Toy Problem Execution
# ============================================================================

def run():
    print("Setting up Multimodal Humanitarian Logistics Network...")
    
    # 1. Commodities
    shipments = [
        Shipment(shipment_id="S1", weight_kg=1500.0, origin="W1", destination="W2"),
        Shipment(shipment_id="S2", weight_kg=2200.0, origin="W1", destination="W2"),
    ]

    # 2. Network Edges (Multimodal: Ground and Air)
    edges = [
        # Outbound First-Mile (Warehouse to Departure Airport) -> GROUND
        TransportEdge("W1_A1_Gnd", "W1", "A1", "Gnd", 50, 0.2),
        TransportEdge("W1_A2_Gnd", "W1", "A2", "Gnd", 75, 0.3),
        
        # Air Transit links -> AIR
        TransportEdge("A1_A3_AirA", "A1", "A3", "AirA", 500, 1.3),
        TransportEdge("A1_A4_AirB", "A1", "A4", "AirB", 600, 1.0),
        TransportEdge("A2_A3_AirC", "A2", "A3", "AirC", 500, 1.5),
        TransportEdge("A2_A4_AirA", "A2", "A4", "AirA", 600, 0.8),
        
        # Inbound Last-Mile (Arrival Airport to Warehouse) -> GROUND
        TransportEdge("A3_W2_Gnd", "A3", "W2", "Gnd", 40, 0.15),
        TransportEdge("A4_W2_Gnd", "A4", "W2", "Gnd", 60, 0.25),
    ]

    params = StochasticOptimizationParameters(cost_penalty_rejection=500.0, cost_reassignment=1.5)

    # 3. Simulate SAA Scenarios (Latent Slack -> Binary Feasibility)
    N_SCENARIOS = 15
    scenarios = []

    random.seed(41)  # For reproducibility
    
    for _ in range(N_SCENARIOS):
        scenario = []
        for s in shipments:
            for e in edges:
                # We only simulate uncertainty for Air legs. Ground is assumed deterministic.
                if "Gnd" in e.edge_id:
                    acceptance = True
                else:
                    # Simulating true slack: z_i_true = z_T100 - epsilon
                    simulated_z_i = random.uniform(1000, 4000) 
                    
                    # Selection rule: U = I(w_s <= z_true)
                    acceptance = bool(simulated_z_i >= s.weight_kg)
                
                scenario.append(
                    UncertaintyRealization(shipment_id=s.shipment_id, edge_id=e.edge_id, acceptance=acceptance)
                )
        scenarios.append(scenario)

    # 4. Solve
    print("Building MCNFP model and optimizing...")
    solver = NetworkFlowTwoStageSolver(shipments, edges, params)
    model, x_vars, keep_vars, reassign_vars = solver.build_integrated_saa_model(scenarios)
    model.optimize()

    # 5. Extract and print results
    if model.Status == GRB.OPTIMAL:
        print("\n--- OPTIMIZATION SUCCESSFUL ---")
        print(f"Total Expected Cost: ${model.ObjVal:,.2f}")
        
        print("\nFirst Stage Assignments (Nominal Path):")
        for s in solver.S:
            path = [e for e in solver.E if x_vars[s, e].X > 0.5]
            print(f"  Shipment {s}: -> ".join(["Origin"] + path))
            
        print("\nExpected Reassignment Vulnerabilities:")
        for s in solver.S:
            for e in solver.E:
                # Average the reassignments across scenarios
                reassign_prob = sum(reassign_vars[s, e, om].X for om in range(N_SCENARIOS)) / N_SCENARIOS
                if reassign_prob > 0.05:
                    print(f"  Shipment {s} forced to re-route via {e} ({reassign_prob*100:.1f}% risk)")
    else:
        print(f"Optimization failed with status {model.Status}")

if __name__ == "__main__":
    run()