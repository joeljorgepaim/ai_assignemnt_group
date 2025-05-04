import random
import math
import matplotlib.pyplot as plt
from copy import deepcopy


class TSP:
    def __init__(self, towns, distances):
        
        self.towns = towns
        self.distances = distances

    def route_distance(self, route):
        
        total = 0
        for i in range(len(route)):
            town1 = route[i]
            town2 = route[(i + 1) % len(route)]
            total += self.distances[town1][town2]
        return total


class SimulatedAnnealingSolver:
    def __init__(self, tsp, initial_temp=10000, cooling_rate=0.003, iterations=10000):
        
        self.tsp = tsp
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.iterations = iterations

    def initial_route(self):
         
        route = self.tsp.towns.copy()
        # Start from Windhoek (first town)
        start_town = route.pop(0)
        random.shuffle(route)
        return [start_town] + route + [start_town]

    def generate_neighbor(self, route):
         
        new_route = route.copy()
        # Ensure we don't swap the first or last town (Windhoek)
        i, j = random.sample(range(1, len(new_route) - 1), 2)
        new_route[i], new_route[j] = new_route[j], new_route[i]
        return new_route

    def acceptance_probability(self, current_energy, new_energy, temp):
         
        if new_energy < current_energy:
            return 1.0
        return math.exp((current_energy - new_energy) / temp)

    def solve(self):
         
        current_route = self.initial_route()
        best_route = current_route.copy()

        current_energy = self.tsp.route_distance(current_route)
        best_energy = current_energy

        temp = self.initial_temp

        energy_history = [current_energy]
        temp_history = [temp]

        for i in range(self.iterations):
            if temp <= 0.1:
                break

            new_route = self.generate_neighbor(current_route)
            new_energy = self.tsp.route_distance(new_route)

            if self.acceptance_probability(current_energy, new_energy, temp) > random.random():
                current_route = new_route
                current_energy = new_energy

                if current_energy < best_energy:
                    best_route = current_route.copy()
                    best_energy = current_energy

            # Cool the temperature
            temp *= 1 - self.cooling_rate
            energy_history.append(current_energy)
            temp_history.append(temp)

        return best_route, best_energy, energy_history, temp_history


def plot_route(tsp, route, title):
    
    plt.figure(figsize=(10, 6))

    # Plot the route
    x = range(len(route))
    y = [tsp.route_distance(route[:i + 1]) for i in range(len(route))]

    plt.plot(x, y, 'b-', marker='o')
    plt.title(title)
    plt.xlabel("Town Index")
    plt.ylabel("Cumulative Distance (km)")

    for i, town in enumerate(route):
        plt.annotate(town, (x[i], y[i]))

    plt.grid()
    plt.show()


def plot_convergence(energy_history, temp_history):
     
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(energy_history)
    plt.title("Route Distance over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Distance (km)")

    plt.subplot(1, 2, 2)
    plt.plot(temp_history)
    plt.title("Temperature over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Temperature")

    plt.tight_layout()
    plt.show()


def create_distance_matrix():
    
    towns = ["Windhoek", "Swakopmund", "Walvis Bay", "Otjiwarongo", "Tsumeb",
             "Grootfontein", "Mariental", "Keetmanshoop", "Ondangwa", "Oshakati"]

    # Distance matrix from the assignment
    distance_matrix = [
        [0, 361, 395, 249, 433, 459, 268, 497, 678, 712],  # Windhoek
        [361, 0, 35.5, 379, 562, 589, 541, 859, 808, 779],  # Swakopmund
        [395, 35.5, 0, 413, 597, 623, 511, 732, 884, 855],  # Walvis Bay
        [249, 379, 413, 0, 260, 183, 519, 768, 514, 485],  # Otjiwarongo
        [433, 562, 597, 260, 0, 60, 682, 921, 254, 288],  # Tsumeb
        [459, 589, 623, 183, 60, 0, 708, 947, 308, 342],  # Grootfontein
        [268, 541, 511, 519, 682, 708, 0, 231, 909, 981],  # Mariental
        [497, 859, 732, 768, 921, 947, 231, 0, 1175, 1210],  # Keetmanshoop
        [678, 808, 884, 514, 254, 308, 909, 1175, 0, 30],  # Ondangwa
        [712, 779, 855, 485, 288, 342, 981, 1210, 30, 0]  # Oshakati
    ]

    # Create distance dictionary
    distances = {}
    for i, town1 in enumerate(towns):
        distances[town1] = {}
        for j, town2 in enumerate(towns):
            distances[town1][town2] = distance_matrix[i][j]

    return towns, distances


def main():
    # Create the TSP problem
    towns, distances = create_distance_matrix()
    tsp = TSP(towns, distances)

    # Initialize and run the solver
    solver = SimulatedAnnealingSolver(tsp, initial_temp=10000, cooling_rate=0.003, iterations=10000)

    print("Starting simulated annealing for TSP...")
    best_route, best_distance, energy_history, temp_history = solver.solve()

    # Display results
    initial_route = solver.initial_route()
    initial_distance = tsp.route_distance(initial_route)

    print("\n=== Results ===")
    print(f"Initial route distance: {initial_distance:.2f} km")
    print(f"Initial route: {initial_route}")
    print(f"\nOptimized route distance: {best_distance:.2f} km")
    print(f"Optimized route: {best_route}")

    # Plot results
    plot_convergence(energy_history, temp_history)
    plot_route(tsp, initial_route, "Initial Route")
    plot_route(tsp, best_route, "Optimized Route")

    # Compare with brute force for small subset (4 towns)
    print("\n=== Small-scale Validation ===")
    small_towns = towns[:4]
    small_distances = {town: {t: distances[town][t] for t in small_towns} for town in small_towns}
    small_tsp = TSP(small_towns, small_distances)

    # Brute force solution
    from itertools import permutations
    all_routes = permutations(small_towns[1:])  # All permutations excluding Windhoek
    all_routes = [[small_towns[0]] + list(route) + [small_towns[0]] for route in all_routes]

    brute_force_distances = [small_tsp.route_distance(route) for route in all_routes]
    optimal_distance = min(brute_force_distances)
    optimal_route = all_routes[brute_force_distances.index(optimal_distance)]

    print(f"Brute force optimal distance for {len(small_towns)} towns: {optimal_distance:.2f} km")
    print(f"Optimal route: {optimal_route}")

    # Solve with simulated annealing
    small_solver = SimulatedAnnealingSolver(small_tsp, initial_temp=1000, cooling_rate=0.003, iterations=1000)
    sa_route, sa_distance, _, _ = small_solver.solve()

    print(f"\nSimulated annealing distance: {sa_distance:.2f} km")
    print(f"SA route: {sa_route}")
    print(f"Difference from optimal: {sa_distance - optimal_distance:.2f} km")


if __name__ == "__main__":
    main()