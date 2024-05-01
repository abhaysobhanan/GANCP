include("utils.jl")
include("parameters.jl")
using PyCall, Hygese

py_dir = joinpath(@__DIR__, "..\\cvrp-cost-prediction\\")
# @show py_dir 

py"""
import os
import sys; sys.path.append($py_dir); 
import dgl
import torch
import numpy
from src.dgl.network.transformer import Transformer
from src.dgl.get_graph import get_knn_graph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Transformer(in_dim=4, latent_dim=128, n_layers=4).to(device)
state_dict = torch.load(os.path.join($py_dir, 'weights', 'model_step4.pt'),
                        map_location=torch.device(device))
model.load_state_dict(state_dict)
model.eval()

def batch_prediction(coords, demands, Qs, batch_size, mini_batch_len):
    gs = [None]*batch_size
    meta_data = [None]*batch_size
    cost_scalers = [None]*batch_size
    for i in range(batch_size):
        gs[i], meta_data[i] = get_knn_graph(coord=numpy.asarray(coords[i]), demand=numpy.asarray(demands[i]), q=Qs[i])
        cost_scalers[i] = meta_data[i]['cost_scaler']

    predictions = []

    for mini_batch_count in range(0, batch_size, mini_batch_len):  #larger instances ~ more memory 
        mini_batch_min = mini_batch_count
        mini_batch_max = min(batch_size, mini_batch_min+mini_batch_len)

        gs_mini_batch = dgl.batch(gs[mini_batch_min:mini_batch_max]).to(device)
        with torch.no_grad():
            preds = model(gs_mini_batch, gs_mini_batch.ndata['feat'])
        preds = preds.cpu().flatten().numpy()*cost_scalers[mini_batch_min:mini_batch_max]      #.cpu()

        predictions = numpy.concatenate((predictions, preds)) #torch.cat((predictions, preds), 0) #torch.cuda.empty_cache()
    return predictions   
"""

function batch_fitness_cost(instance::MDVRP_Instance, population::Vector{Vector{Int}})
    list_len = length(population)
    coords = Vector{Vector{Vector{Float64}}}(undef, list_len * instance.D)
    demands = Vector{Vector{Int64}}(undef, list_len * instance.D)
    Qs = repeat(vcat(instance.vehicle_loads...), list_len)
    for i in eachindex(population)
        clusters = chromosome_to_clusters(instance, population[i])
        depot = 0
        for cluster in clusters
            depot += 1
            coord = instance.customers[cluster]
            pushfirst!(coord, instance.depots[depot])
            coords[(i-1)*instance.D+depot] = coord
            demands[(i-1)*instance.D+depot] = vcat(instance.demands[cluster]...)
            pushfirst!(demands[(i-1)*instance.D+depot], 0)
        end
    end

    mini_batch_len = find_CUDA_batch_size(instance)  #hardcoded to manage CUDA memory limit (8 GB)
    all_costs = py"batch_prediction"(coords, demands, Qs, list_len*instance.D, mini_batch_len)
    costs = Vector{Float64}(undef, list_len)
    for i in eachindex(costs)
        costs[i] = sum(all_costs[(i-1)*instance.D+1:i*instance.D])
    end
    return costs
end

function diversity(population::Vector{Vector{Int}})
    pop_len = length(population)
    scores = Vector{Float64}(undef, pop_len)
    for i = 1:pop_len
        score = 0
        for j = 1:pop_len
            score += sum(population[i].!=population[j]) #hamming distance (1/instance.N)*
        end
        scores[i] = score
    end
    return scores./sum(scores)
end

function fitness_scores(costs::Vector{Float64}, diversities::Vector{Float64}, w1::Float64, w2::Float64, excess_demands::Vector{Int}; infeas_penalty = 0.1)
    costs = costs./sum(costs)
    # use normalized costs with diversities
    score = w1*costs - w2*diversities + infeas_penalty*costs.*excess_demands
    return score
end

function hgs_solutions(instance::MDVRP_Instance, top_solutions::Vector{Vector{Int}}, hgs_time_limit::Float64)
    ap = AlgorithmParameters(timeLimit=hgs_time_limit) 
    list_len = length(top_solutions)
    customers = Array(1:instance.N)
    costs = Vector{Float64}(undef, list_len)
    mdvrp_routes = Vector{Vector{Vector{Vector{Int64}}}}(undef, list_len)
    for i in eachindex(top_solutions)
        clusters = chromosome_to_clusters(instance, top_solutions[i])
        routes = Vector{Vector{Vector{Int64}}}(undef, length(clusters))
        cost = 0
        for index in eachindex(clusters)
            cluster = clusters[index]
            if length(cluster) > 1
                x, y = coords_to_xy(instance.customers[cluster])
                demand = instance.demands[cluster]
                # route_duration = instance.route_durations[index]
                # if route_duration == 0
                #     route_duration = Inf
                # end
                Q = instance.vehicle_loads[index][1]
                pushfirst!(x, instance.depots[index][1])
                pushfirst!(y, instance.depots[index][2])
                service_times = zeros(Int, length(x))
                pushfirst!(demand, 0)
                solution = solve_cvrp(x, y, demand, Q, n_vehicles=instance.M, ap; verbose=false, round=false) # Q, duration_limit=route_duration
                if ((solution.cost == 0) && (length(demand)>1))
                    cost = Inf
                else
                    cost += solution.cost
                end
                # Use the following to re-index customers from solver solution to obtain mdvrp routes
                allocated_customers = [0; customers[cluster]] #CVRP: first index is depot
                for j in eachindex(solution.routes)
                    solution.routes[j] = allocated_customers[solution.routes[j]]
                end
                routes[index] = solution.routes
            elseif length(cluster) == 1
                cost += 2*norm(instance.customers[cluster[1]]-instance.depots[index])
            end
        end
        costs[i] = cost
        mdvrp_routes[i] = routes
    end
    return costs, mdvrp_routes
end
