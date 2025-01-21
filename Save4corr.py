import os
import math
import numpy as np
import random
from collections import defaultdict
from sklearn.model_selection import ParameterGrid
import json
import matplotlib.pyplot as plt



def read_vrp_file(file_path):
    """Чтение файла .врп и извлечение данных в зависимости от формата"""
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
                    dist_matrix[i][j] = round(dist_matrix[i][j])  # Округляем до целого
                    if dist_matrix[i][j] == 0:
                        dist_matrix[i][j] = 1
    else:
        raise ValueError("Не найдено данных для построения матрицы расстояний")

    return {
        'name': data['NAME'],
        'capacity': int(data['CAPACITY']),
        'dist_matrix': dist_matrix,
        'coordinates': coordinates,
        'demands': demands,
        'depot': depot - 1,  # Convert to 0-based index
        'optimal_value': optimal_value
    }

def calculate_routes_cost(routes, dist_matrix):
    """Вычисление общей стоимости для набора маршрутов"""
    total_cost = 0
    for route in routes:
        for i in range(len(route) - 1):
            cost = dist_matrix[route[i]][route[i + 1]]
            total_cost += cost
    return total_cost

def format_routes(routes):
    """Перевод маршрутов в полный формат"""
    formatted_routes = []
    for route in routes:
        if len(route) > 2:
            formatted_routes.append(route)  # Добавляем весь маршрут
    return formatted_routes

def two_opt(route, dist_matrix):
        best_route = route[:]
        best_cost_two_opt = calculate_routes_cost([route], dist_matrix)
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    if j - i == 1:  # Исключаем соседние узлы
                        continue
                    new_route = route[:i] + route[i:j][::-1] + route[j:]
                    new_cost = calculate_routes_cost([new_route], dist_matrix)
                    if new_cost < best_cost_two_opt:
                        best_route = new_route
                        best_cost_two_opt = new_cost
                        improved = True
        return best_route

def solve_cvrp_with_aco(data, dist_matrix, num_ants=20, alpha=1.0, beta=5.0, evaporation_rate=0.7, num_iterations=500, stop_criteria=10):
    capacity = data['capacity']
    demands = data['demands']
    depot = data['depot']
    num_nodes = len(demands)

    pheromones = np.ones((num_nodes, num_nodes))
    best_routes_overall = None
    best_cost_overall = float('inf')
    last_best_cost_overall = best_cost_overall
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

                optimized_route = two_opt(current_route, dist_matrix)
                routes.append(optimized_route)

            route_cost = calculate_routes_cost(routes, dist_matrix)
            all_routes.append(routes)
            all_costs.append(route_cost)

            if route_cost < best_cost_overall:
                best_cost_overall = route_cost
                best_routes_overall = [route[:] for route in routes]  # Сохраняем копию маршрутов

        pheromones *= (1 - evaporation_rate)
        for routes, cost in zip(all_routes, all_costs):
            for route in routes:
                for i in range(len(route) - 1):
                    pheromones[route[i]][route[i + 1]] += 1.0 / (cost + 1e-6)

        if abs(last_best_cost_overall - best_cost_overall) / best_cost_overall < 1e-3:
            stagnant_iterations += 1
            if stagnant_iterations >= stop_criteria:
                break
        else:
            stagnant_iterations = 0
        last_best_cost_overall = best_cost_overall


    best_routes_overall = format_routes(best_routes_overall)
    assert calculate_routes_cost(best_routes_overall, dist_matrix) == best_cost_overall, \
        "Ошибка: Маршруты не соответствуют минимальной стоимости"

    return best_routes_overall, best_cost_overall


def load_or_create_params(file_name, data, dist_matrix, param_dir, use_recommended):
    base_name = os.path.splitext(file_name)[0]
    params_file = os.path.join(param_dir, f"{base_name}_params.json")
    if os.path.exists(params_file):
        if use_recommended:
            with open(params_file, 'r') as f:
                return json.load(f)
        else:
            best_params, best_cost_tune = tune_hyperparameters(data, dist_matrix)
            params = {'best_params': best_params, 'best_cost_tune': best_cost_tune}
            with open(params_file, 'w') as f:
                json.dump(params, f)
            return params
    else:
        if use_recommended:
            raise FileNotFoundError(f"Recommended parameters not found for {file_name}")
        best_params, best_cost_tune = tune_hyperparameters(data, dist_matrix)
        params = {'best_params': best_params, 'best_cost_tune': best_cost_tune}
        os.makedirs(param_dir, exist_ok=True)
        with open(params_file, 'w') as f:
            json.dump(params, f)
        return params

def tune_hyperparameters(data, dist_matrix):
    param_grid = {
        'num_ants': [10, 20, 30],
        'alpha': [0.5, 1.0, 2.0],
        'beta': [2.0, 5.0, 10.0],
        'evaporation_rate': [0.5, 0.7, 0.9],
        'num_iterations': [100, 300, 500]
    }

    best_params = None
    best_cost_tune = float('inf')
    best_routes_tune = None

    for params in ParameterGrid(param_grid):
        routes, cost = solve_cvrp_with_aco(
            data, dist_matrix,
            num_ants=params['num_ants'],
            alpha=params['alpha'],
            beta=params['beta'],
            evaporation_rate=params['evaporation_rate'],
            num_iterations=params['num_iterations']
        )

        if cost < best_cost_tune:
            best_cost_tune = cost
            best_params = params
            best_routes_tune = routes

    return best_params, best_cost_tune

def main():
    directories = {
        'e': ('./vrp_files_e/', './e_params/'),
        'p': ('./vrp_files_p/', './p_params/'),
        'b': ('./vrp_files_b/', './b_params/')
    }

    print("Choose datasets to process (e.g., e, p, b or all):")
    selected_sets = input().strip().lower().split()
    if 'all' in selected_sets:
        selected_sets = ['e', 'p', 'b']

    print("Use recommended parameters or tune new ones? (recommended/tune):")
    use_recommended = input().strip().lower() == 'recommended'

    print("Choose output format: full, brief, or detailed (full/brief/detailed):")
    output_format = input().strip().lower()

    results = {}
    deviations = []

    for key, (data_dir, param_dir) in directories.items():
        if key not in selected_sets:
            continue

        for file_name in os.listdir(data_dir):
            if file_name.endswith('.vrp'):
                file_path = os.path.join(data_dir, file_name)
                data = read_vrp_file(file_path)
                dist_matrix = data['dist_matrix']

                try:
                    params = load_or_create_params(file_name, data, dist_matrix, param_dir, use_recommended)
                except FileNotFoundError as e:
                    print(e)
                    continue

                best_params = params['best_params']

                optimal_value = data.get('optimal_value', None)

                # Решение задачи CVRP с ACO
                routes, best_cost_overall = solve_cvrp_with_aco(data, dist_matrix, **best_params)

                # Вычисление отклонения
                deviation = ((best_cost_overall - optimal_value) / optimal_value) * 100 if optimal_value else None
                if deviation is not None:
                    deviations.append(deviation)

                # Вывод результатов
                if output_format == 'full':
                    print(f"File: {file_name}")
                    print(f"Best Parameters: {best_params}")
                    print(f"Best Cost (Calculated): {best_cost_overall}")
                    print(f"Optimal Value: {optimal_value}")
                    print(f"Deviation: {deviation:.2f}%\n" if deviation is not None else "Deviation: N/A\n")
                elif output_format == 'brief':
                    print(f"File: {file_name}")
                    print(f"Best Cost (Calculated): {best_cost_overall}")
                    print(f"Optimal Value: {optimal_value}")
                    print(f"Deviation: {deviation:.2f}%\n" if deviation is not None else "Deviation: N/A\n")
                elif output_format == 'detailed':
                    print(f"File: {file_name}")
                    for route in routes:
                        print(f"{route}")
                    print(f"Best Cost (Calculated): {best_cost_overall}")
                    print(f"Optimal Value: {optimal_value}")
                    print(f"Deviation: {deviation:.2f}%\n" if deviation is not None else "Deviation: N/A\n")

    if deviations:
        avg_deviation = sum(deviations) / len(deviations)
        print(f"Average Deviation: {avg_deviation:.2f}%")

if __name__ == '__main__':
    main()
