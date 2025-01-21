import os
import math
import numpy as np
import matplotlib.pyplot as plt

# Functions to process VRP and SOL files
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

def read_solution_file(sol_path):
    """Чтение файла .sol для загрузки эталонных маршрутов"""
    with open(sol_path, 'r') as f:
        lines = f.readlines()

    routes = []
    for line in lines:
        line = line.strip()
        if line.startswith('Route'):
            # Конвертация маршрута без изменения индексов (они уже начинаются с 0)
            route = list(map(int, line.split(':')[1].strip().split()))
            routes.append(route)

    return routes

def calculate_routes_cost(routes, dist_matrix):
    """Вычисление общей стоимости для набора маршрутов"""
    total_cost = 0
    for route in routes:
        extended_route = [0] + route + [0]  # Добавляем депо в начало и конец
        for i in range(len(extended_route) - 1):
            cost = dist_matrix[extended_route[i]][extended_route[i + 1]]
            total_cost += cost
    return total_cost

def validate_solution_with_vrp(vrp_path, sol_path):
    """Проверка стоимости маршрутов из .sol с данными из .врп"""
    vrp_data = read_vrp_file(vrp_path)
    dist_matrix = vrp_data['dist_matrix']
    optimal_value = vrp_data['optimal_value']
    routes = read_solution_file(sol_path)

    calculated_cost = calculate_routes_cost(routes, dist_matrix)

    if not math.isclose(calculated_cost, optimal_value, rel_tol=1e-6):
        print(f"{vrp_path}: Mismatch - Calculated cost = {calculated_cost}, Optimal value = {optimal_value}")
    else:
        print(f"{vrp_path}: Validation passed - Cost is correct")

def validate_all_vrp_files():
    """Проверка всех .врп и .sol файлов в директориях"""
    directories = {
        'b': ('./vrp_files_b/', './vrp_files_b/')
    }

    for key, (vrp_dir, sol_dir) in directories.items():
        print(f"Validating files in {vrp_dir}...")
        for file_name in os.listdir(vrp_dir):
            if file_name.endswith('.vrp'):
                vrp_path = os.path.join(vrp_dir, file_name)
                sol_path = os.path.join(sol_dir, file_name.replace('.vrp', '.sol'))

                if not os.path.exists(sol_path):
                    print(f"Solution file not found for {file_name}")
                    continue

                validate_solution_with_vrp(vrp_path, sol_path)

if __name__ == '__main__':
    validate_all_vrp_files()

