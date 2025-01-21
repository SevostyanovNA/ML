import os
import math
import numpy as np
import random
from collections import defaultdict
from sklearn.model_selection import ParameterGrid

def read_vrp_file(file_path):
    """Чтение файла .vrp и извлечение данных в зависимости от формата"""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    data = {}
    coordinates = []
    demands = []
    depot = None
    optimal_value = None
    edge_weights = []
    dimension = 0
    edge_weight_format = None

    section = None
    for line in lines:
        line = line.strip()
        if line.startswith('NAME') or line.startswith('TYPE') or line.startswith('DIMENSION') or line.startswith('CAPACITY'):
            key, value = line.split(':')
            data[key.strip()] = value.strip()
        elif line.startswith('COMMENT'):
            if 'Optimal value' in line:
                optimal_value_str = line.split('Optimal value:')[1].strip()
                optimal_value = int(''.join(filter(str.isdigit, optimal_value_str)))
            elif 'Best value' in line:
                optimal_value_str = line.split('Best value:')[1].strip()
                optimal_value = int(''.join(filter(str.isdigit, optimal_value_str)))
        elif line.startswith('EDGE_WEIGHT_FORMAT'):
            edge_weight_format = line.split(':')[1].strip()
        elif line.startswith('NODE_COORD_SECTION'):
            section = 'NODE_COORD_SECTION'
        elif line.startswith('EDGE_WEIGHT_SECTION'):
            section = 'EDGE_WEIGHT_SECTION'
        elif line.startswith('DEMAND_SECTION'):
            section = 'DEMAND_SECTION'
        elif line.startswith('DEPOT_SECTION'):
            section = 'DEPOT_SECTION'
        elif line == 'EOF':
            break
        elif section == 'NODE_COORD_SECTION':
            _, x, y = map(float, line.split())
            coordinates.append((x, y))
        elif section == 'EDGE_WEIGHT_SECTION':
            edge_weights.extend(map(int, line.split()))
        elif section == 'DEMAND_SECTION':
            node, demand = map(int, line.split())
            demands.append(demand)
        elif section == 'DEPOT_SECTION':
            depot = int(line) if int(line) != -1 else depot

    dimension = int(data['DIMENSION'])

    if edge_weight_format == 'LOWER_ROW':
        dist_matrix = np.zeros((dimension, dimension))
        index = 0
        for i in range(1, dimension):
            for j in range(i):
                dist_matrix[i][j] = edge_weights[index]
                dist_matrix[j][i] = edge_weights[index]
                index += 1
    elif coordinates:
        dist_matrix = np.zeros((dimension, dimension))
        for i in range(dimension):
            for j in range(dimension):
                if i != j:
                    dist_matrix[i][j] = math.sqrt((coordinates[i][0] - coordinates[j][0])**2 +
                                                  (coordinates[i][1] - coordinates[j][1])**2)
                    if dist_matrix[i][j] == 0:
                        dist_matrix[i][j] = 1e-6
    else:
        raise ValueError("Не найдено данных для построения матрицы расстояний")

    return {
        'name': data['NAME'],
        'capacity': int(data['CAPACITY']),
        'dist_matrix': dist_matrix,
        'demands': demands,
        'depot': depot - 1,  # Convert to 0-based index
        'optimal_value': optimal_value
    }

def calculate_routes_cost(routes, dist_matrix):
    """Вычисление стоимости маршрутов"""
    cost = 0
    for route in routes:
        for i in range(len(route) - 1):
            cost += dist_matrix[route[i]][route[i + 1]]
    return cost

def solve_cvrp_with_aco(data, dist_matrix, num_ants=20, alpha=1.0, beta=5.0, evaporation_rate=0.7, num_iterations=500, stop_criteria=10):
    capacity = data['capacity']
    demands = data['demands']
    depot = data['depot']
    num_nodes = len(demands)

    pheromones = np.ones((num_nodes, num_nodes))
    best_routes = None
    best_cost = float('inf')
    last_best_cost = best_cost
    stagnant_iterations = 0

    for iteration in range(num_iterations):
        all_routes = []
        all_costs = []

        for ant in range(num_ants):
            visited = set()
            routes = []

            while len(visited) < num_nodes - 1:
                current_route = [depot]
                current_load = 0

                while True:
                    probabilities = []
                    current_node = current_route[-1]
                    for next_node in range(num_nodes):
                        if next_node not in visited and next_node != depot and current_load + demands[next_node] <= capacity:
                            pheromone = pheromones[current_node][next_node] ** alpha
                            distance = (1.0 / dist_matrix[current_node][next_node]) ** beta
                            probabilities.append((next_node, pheromone * distance))

                    if probabilities:
                        next_node = random.choices(
                            [node for node, _ in probabilities],
                            weights=[prob for _, prob in probabilities]
                        )[0]
                        current_route.append(next_node)
                        visited.add(next_node)
                        current_load += demands[next_node]
                    else:
                        current_route.append(depot)
                        break

                routes.append(current_route)

            route_cost = calculate_routes_cost(routes, dist_matrix)
            all_routes.append(routes)
            all_costs.append(route_cost)

            if route_cost < best_cost:
                best_cost = route_cost
                best_routes = routes

        pheromones *= (1 - evaporation_rate)
        for routes, cost in zip(all_routes, all_costs):
            for route in routes:
                for i in range(len(route) - 1):
                    pheromones[route[i]][route[i + 1]] += 1.0 / (cost + 1e-6)

        if abs(last_best_cost - best_cost) / best_cost < 1e-3:
            stagnant_iterations += 1
            if stagnant_iterations >= stop_criteria:
                print(f"Stopping at iteration {iteration}, as there was no significant improvement.")
                break
        else:
            stagnant_iterations = 0
        last_best_cost = best_cost

        print(f"Iteration {iteration}: Best Cost = {best_cost}")

    best_routes = [[node + 1 for node in route] for route in best_routes]

    return best_routes, best_cost

def tune_hyperparameters(data, dist_matrix):
    """Автоматический подбор гиперпараметров для ACO"""
    param_grid = {
        'num_ants': [10, 20, 30],
        'alpha': [0.5, 1.0, 2.0],
        'beta': [2.0, 5.0, 10.0],
        'evaporation_rate': [0.5, 0.7, 0.9],
        'num_iterations': [100, 300, 500]
    }

    best_params = None
    best_cost = float('inf')

    for params in ParameterGrid(param_grid):
        print(f"Testing parameters: {params}")
        _, cost = solve_cvrp_with_aco(
            data, dist_matrix,
            num_ants=params['num_ants'],
            alpha=params['alpha'],
            beta=params['beta'],
            evaporation_rate=params['evaporation_rate'],
            num_iterations=params['num_iterations']
        )

        if cost < best_cost:
            best_cost = cost
            best_params = params

    print(f"Best parameters: {best_params}")
    print(f"Best cost: {best_cost}")

    return best_params, best_cost

def main():
    directory = './vrp_files_b/'
    results = {}
    deviations = []

    for file_name in os.listdir(directory):
        if file_name.endswith('.vrp'):
            file_path = os.path.join(directory, file_name)
            data = read_vrp_file(file_path)
            dist_matrix = data['dist_matrix']

            print(f"Tuning hyperparameters for {file_name}...")
            best_params, best_cost = tune_hyperparameters(data, dist_matrix)

            optimal_value = data.get('optimal_value', None)
            if optimal_value:
                deviation = ((best_cost - optimal_value) / optimal_value) * 100
                deviations.append(deviation)

            results[file_name] = {
                'best_params': best_params,
                'best_cost': best_cost,
                'optimal_value': optimal_value,
                'deviation': deviation if optimal_value else None
            }

    avg_deviation = sum(deviations) / len(deviations) if deviations else None

    for file_name, result in results.items():
        print(f"File: {file_name}")
        print(f"Best Parameters: {result['best_params']}")
        print(f"Best Cost: {result['best_cost']}")
        print(f"Optimal Value: {result['optimal_value']}")
        print(f"Deviation: {result['deviation']:.2f}%" if result['deviation'] is not None else "Deviation: N/A")
        print()

    if avg_deviation is not None:
        print(f"Average Deviation: {avg_deviation:.2f}%")

if __name__ == '__main__':
    main()