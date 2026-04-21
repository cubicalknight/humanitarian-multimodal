# %%
import numpy as np
import random
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import signal
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from tqdm import tqdm

# Global flag to handle interruption
interrupted = False

def signal_handler(sig, frame):
    global interrupted
    interrupted = True
    print("\nInterrupt received! Stopping optimization...")

# Register the signal handler for KeyboardInterrupt (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

def my_callback(model, where):
    global interrupted
    if where == GRB.Callback.MIPNODE:
        if interrupted:
            model.terminate()  # Stop optimization safely

@dataclass(frozen=True)
class Vehicle:
    vehicle_name: str
    is_truck: bool = False
    capacity: int = 0
    speed: int = 1
    range: int = 1
    cost: float = 1.0


# Maybe reincorporate later
# @dataclass(frozen=True)
# class AircraftAssignment:
#     vehicle: Vehcile = field(default_factory=Vehcile)
#     # If negative 1, then assume it is a truck
#     number: int = -1

@dataclass(frozen=True)
class Facility:
    name: str
    is_airport: bool = False
    supply: int = 0
    demand: int = 0

@dataclass
class Route:
    origin: Facility
    destination: Facility
    distance: int = 0
    aircraft_assigned: Dict[str, int] = field(default_factory=dict)  # Use vehicle name as key

    @property
    def is_air_route(self) -> bool:
        return self.origin.is_airport and self.destination.is_airport

    @property
    def endpoints(self) -> Tuple[str, str]:
        return (self.origin.name, self.destination.name)

@dataclass
class Scenario:
    name: str
    probability: float
    demand: float


@dataclass
class BuildData:
    facilities: List[Facility] = field(default_factory=list)
    
    scenarios: List[Scenario] = field(default_factory=list)
    aircraft_types: List[Vehicle] = field(default_factory=list)

    # Assume that the routes are subsets with size = 2 of the facilities
    routes: List[Route] = field(default_factory=list)
    
    truck_pool: int = 10000
    truck_capacity: int = 1000
    truck_cost: float = 10.0
    truck_speed: float = 10.0
    truck_range: float = 1000.0

gp.setParam('OutputFlag', 0)

class ReservationModel:
    def __init__(self, data: BuildData):
        self.data: BuildData = data

        # Higher alpha can be interpreted as valuaing the revenue loss more than the demand
        # Initially set to 0.5 to signify equal weighting of both objectives
        self.alpha: float = 0.5  # Weighting factor for the objective function
        self.cost_of_unmet_demand: float = 100.0  # Cost of unmet demand, can be adjusted based on scenario
        self.cost_lost_revenue: float = 100.0  # Cost of lost revenue, can be adjusted based on scenario

    def capacity_reservation_model(self):
        model = gp.Model("Supply-Demand Optimization")

        # Decision Variables
        # Capacity reserved for use in an aircraft at an airport node
        aircraft_by_route = [
            (route.endpoints[0], route.endpoints[1], aircraft_name, i)
            for route in self.data.routes if route.is_air_route
            for aircraft_name, count in route.aircraft_assigned.items()
            for i in range(count)
        ]
       
        k = model.addVars(aircraft_by_route, vtype=GRB.INTEGER, lb=0, name="reserved_capacity")        
        loss = model.addVars(k.keys(), vtype=GRB.CONTINUOUS, name="lost_value")

        # Objective Functions
        model.setObjective(loss.sum(), GRB.MINIMIZE)

        # Constraints
        aircraft_dict = {v.vehicle_name: v for v in self.data.aircraft_types}
        model.addConstrs((k[o, d, a, i] <= aircraft_dict[a].capacity for o, d, a, i in k.keys()), "Capacity_Constraint")

        # Unless lost revenue is scaled in a non-linear way, demand will be allocated to the first available aircraft (greedy approach)
        model.addConstrs(((1 - self.alpha) * loss[o, d, a, i] >= self.cost_lost_revenue * self.alpha * k[o, d, a, i] for o, d, a, i in k.keys()), "Lost_Revenue_Constraint")

        # This constraint estimates the lower bound on demand that must be met
        # Assume that it doesn't matter how much demand is met, as long as it is met
        # model.addConstr(lost_revenue >= sum(self.scenario_demand[scenario] for scenario in self.scenario_probability.keys()) - 
        #     gp.quicksum(k[(od, a)] for od in self.aircraft_at_airport.keys() for a in self.aircraft_at_airport[od]), "Demand_Constraint")
        model.addConstr(
            self.alpha * gp.quicksum(loss[o, d, a, i] for o, d, a, i in k.keys())
            >= self.cost_of_unmet_demand * (1 - self.alpha) * (sum(self.data.scenarios[scenario].demand * self.data.scenarios[scenario].probability for scenario in range(len(self.data.scenarios)))
            - gp.quicksum(k[o, d, a, i] for o, d, a, i in k.keys())),
            name="Demand_Constraint"
        )

        model.optimize(my_callback)

        if model.status == GRB.INFEASIBLE:
            model.computeIIS()
            model.write("model.ilp") 
            raise Exception("Model is infeasible. Check the model.ilp file for details.")
        elif model.status == GRB.OPTIMAL or model.status == GRB.INTERRUPTED:
            # print("Model solved successfully.")
            # print(f"Objective value: {model.ObjVal}")
            # for v in model.getVars():
            #     if v.X > 0:
            #         print(f"{v.VarName}: {v.X}")
            pass
        else:
            raise Exception("Model optimization failed.")
        
        return model, k, loss


# =====================================================
# 1.  Wrapper: run MILP forward with given parameters
# =====================================================
def run_milp_forward(model_obj, c_r, c_u, alpha):
    """Run MILP forward model and return total reserved capacity."""
    model_obj.cost_lost_revenue = c_r
    model_obj.cost_of_unmet_demand = c_u
    model_obj.alpha = alpha

    model, k, loss = model_obj.capacity_reservation_model()
    total_reserved = sum(v.X for v in k.values())
    return total_reserved


# =====================================================
# 2.  Log-prior function
# =====================================================
def log_prior(params):
    c_r, c_u, alpha = params
    # Invalid region
    if c_r <= 0 or c_u <= 0 or not (0 < alpha < 1):
        return -np.inf

    # Weak log-normal priors on costs + Beta(2,2)-like on alpha
    lp = -0.5 * ((np.log(c_r)**2 + np.log(c_u)**2) / (1.0**2))
    lp += np.log(alpha * (1 - alpha))  # encourages alpha near 0.5
    return lp


# =====================================================
# 3.  Log-likelihood function
# =====================================================
def log_likelihood(params, model_obj, observed_data, sigma=20.0):
    c_r, c_u, alpha = params
    if alpha <= 0 or alpha >= 1 or c_r <= 0 or c_u <= 0:
        return -np.inf

    logL = 0
    for obs in observed_data:
        # Replace model demand with this observed level
        for s in model_obj.data.scenarios:
            s.demand = obs["demand"]

        # Solve the MILP forward model
        try:
            R_pred = run_milp_forward(model_obj, c_r, c_u, alpha)
        except Exception as e:
            print("MILP solve failed:", e)
            return -np.inf

        R_obs = obs["reserved_obs"]
        logL += -0.5 * ((R_obs - R_pred) / sigma) ** 2
    return logL


# =====================================================
# 4.  Log-posterior (prior + likelihood)
# =====================================================
def log_posterior(params, model_obj, observed_data, sigma=20.0):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(params, model_obj, observed_data, sigma)
    return lp + ll


# =====================================================
# 5.  Simple Metropolis–Hastings MCMC (with Burn-in)
# =====================================================
def metropolis_mcmc(model_obj, observed_data, n_samples=5000, n_burn=1000, step_size=0.15):
    """
    Performs MCMC sampling.
    
    Args:
        model_obj: The ReservationModel instance.
        observed_data: The list of observed data points.
        n_samples (int): The number of samples to *retain* after burn-in.
        n_burn (int): The number of initial samples to *discard*.
        step_size (float): The scale of the proposal distribution.
    """
    samples = []
    theta = np.array([1.0, 1.0, 0.5])  # initial guess
    logpost = log_posterior(theta, model_obj, observed_data)

    total_iterations = n_samples + n_burn
    
    print(f"Running MCMC for {total_iterations} total iterations ({n_burn} burn-in + {n_samples} samples)...")

    for t in tqdm(range(total_iterations), desc="MCMC Sampling"):
        # propose new parameters
        proposal = theta + step_size * np.random.randn(3)
        proposal[2] = np.clip(proposal[2], 1e-3, 0.999)  # keep alpha in (0,1)

        logpost_prop = log_posterior(proposal, model_obj, observed_data)
        accept = np.log(random.random()) < (logpost_prop - logpost)

        if accept:
            theta, logpost = proposal, logpost_prop
        
        # Only save samples *after* the burn-in period
        if t >= n_burn:
            samples.append(theta.copy())

    return np.array(samples)


# =====================================================
# 6.  Example usage (mock data)
# =====================================================
if __name__ == "__main__":
    # Define facilities
    A = Facility("A", is_airport=True)
    B = Facility("B", is_airport=True)

    # One aircraft type
    plane = Vehicle(vehicle_name="A320", capacity=200)

    # One air route
    route = Route(origin=A, destination=B, distance=500, aircraft_assigned={"A320": 2})

    # Scenarios (will be updated in likelihood loop)
    scenarios = [Scenario(name="base", probability=1.0, demand=500)]

    # Bundle data
    data = BuildData(
        facilities=[A, B],
        aircraft_types=[plane],
        routes=[route],
        scenarios=scenarios
    )

    # Initialize model
    model_obj = ReservationModel(data)

    # Observed demand vs reserved capacity (fake data)
    observed_data = [
        {"demand": 250*1.1, "reserved_obs": 250},
        {"demand": 400*1.1, "reserved_obs": 400},
        {"demand": 550*1.1, "reserved_obs": 550},
    ]

    # Run MCMC
    samples = metropolis_mcmc(model_obj, observed_data, n_samples=50000)

    plt.figure(figsize=(10,4))
    # plt.plot(samples[:,0], label="cost_lost_revenue")
    # plt.plot(samples[:,1], label="cost_of_unmet_demand")
    plt.plot(samples[:,0]/samples[:,1], label="c_r / c_u")
    plt.plot(samples[:,2], label="alpha")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title("MCMC Trace")
    plt.show()

    print("\nPosterior means:")
    print(f"c_r   ≈ {np.mean(samples[:,0]):.2f}")
    print(f"c_u   ≈ {np.mean(samples[:,1]):.2f}")
    print(f"alpha ≈ {np.mean(samples[:,2]):.2f}")
# %%
