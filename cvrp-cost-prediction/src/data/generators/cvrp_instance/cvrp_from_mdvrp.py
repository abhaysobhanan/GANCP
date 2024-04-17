import random
import numpy as np

from src.data.generators.x_instance.x_intance_generator import generate_x_instance

def generate_cvrp_instances(n_range):
    customer_per_depot = random.randint(n_range[0],n_range[1])
    n_depots = random.randint(2, 10)
    n_customers = customer_per_depot*n_depots
    # print(n_customers, " ", n_depots, "\n")

    x_data = generate_x_instance(n_customers)
    x_list = list(x_data['x_coordinates'])
    y_list = list(x_data['y_coordinates'])
    demand_list = list(x_data['demands'])
    Q = x_data['vehicle_capacity']

    depots = []
    depots.append([x_list.pop(0), y_list.pop(0)])
    demand_list.pop(0)

    for i in range(1,n_depots):
        depot = [random.randint(1, 1000), random.randint(1, 1000)]
        depots.append(depot)

    vehicle_capacities = []
    for i in range(0,n_depots):
        vehicle_capacities.append(Q)

    M = int(np.ceil((2+2*random.random())*sum(demand_list)/(Q*n_depots))) #number of vehicles per depot
        
    mdvrp_chromosome = np.random.randint(0, n_depots, size=n_customers)

    #convert chromosome to depot clusters
    depot_clusters = [[] for i in range(0,n_depots)]
    for node in range(0, n_customers):
        depot = mdvrp_chromosome[node]
        depot_clusters[depot].append(node)

    #create cvrp instances
    cvrp_lists = [{} for i in range(0,n_depots)]
    for depot in range(0, n_depots):
        cvrp_lists[depot]['x_coordinates'] = [depots[depot][0]]
        cvrp_lists[depot]['y_coordinates'] = [depots[depot][1]]
        cvrp_lists[depot]['demands'] = [0]
        cvrp_lists[depot]['service_times'] = np.zeros(len(depot_clusters[depot])+1)
        cvrp_lists[depot]['vehicle_capacity'] = Q
        cvrp_lists[depot]['num_vehicles'] = M
        cvrp_lists[depot]['depot'] = 0
        for depot_customer in depot_clusters[depot]:
            cvrp_lists[depot]['x_coordinates'].append(x_list[depot_customer])
            cvrp_lists[depot]['y_coordinates'].append(y_list[depot_customer])
            cvrp_lists[depot]['demands'].append(demand_list[depot_customer])

    return cvrp_lists
