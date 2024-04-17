import numpy as np
import tsplib95


def parse_sol_file(file_path: str):
    if not file_path.endswith('.sol'):
        # file_path = file_path.split('.')[0]
        file_path += '.sol'

    tours = []
    cost = None
    run_time = None

    with open(file_path, 'r') as f:
        while True:
            line = f.readline()
            if line.startswith('Route'):
                tours.append([int(st) for st in line.split(':')[1].split(' ') if st.isnumeric()])
            if line.startswith('Cost'):
                cost = float(line.split(' ')[1])
            if line.startswith('Time'):
                run_time = float(line.split(' ')[1])
            if not line:
                break
    return tours, cost, run_time


def parse_vrp_file(file_path: str):
    if not file_path.endswith('.vrp'):
        # file_path = file_path.split('.')[0]
        file_path += '.vrp'
    problem = tsplib95.loaders.load(file_path)

    coords = np.array([v for v in problem.node_coords.values()])
    demands = np.array([v for v in problem.demands.values()])
    q = problem.capacity
    return coords, demands, q, problem


def parse_log_file(file_path: str):
    intermediate_costs = []
    with open(file_path, 'r') as f:
        while True:
            line = f.readline()
            if line.startswith('It'):
                intermediate_cost = float(line.split('|')[2].split(' ')[3])
                intermediate_costs.append(intermediate_cost)
            if not line: break
    return intermediate_costs
