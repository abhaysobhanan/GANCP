function chromosome_to_clusters(instance::MDVRP_Instance, chromosome::Vector{Int})  # chromosome to CVRP clusters 
    clusters = Vector{Int}[Vector{}() for _ in 1:instance.D]
    for node = 1:instance.N
        depot = chromosome[node]
        push!(clusters[depot], node)
    end
    return clusters
end

function clusters_to_chromosome(instance::MDVRP_Instance, clusters::Vector{Vector{Int}})
    individual = Vector{Int}(undef, instance.N)
    for i in eachindex(clusters)
        individual[clusters[i]] .= i 
    end
    return individual
end

function coords_to_xy(coords::Vector{Vector{Float64}})
    list_len = length(coords)
    x = Vector{Int}(undef, list_len)
    y = Vector{Int}(undef, list_len)
    for i = 1:list_len
        x[i] = coords[i][1]
        y[i] = coords[i][2]
    end
    return x, y
end

function delete_duplicates(population::Vector{Vector{Int}}, costs::Vector{Float64})
    indices = unique(i -> population[i], 1:length(population))   #gives unique indices
    population_new = deepcopy(population[indices])
    costs_new = copy(costs[indices])
    return population_new, costs_new
end

function are_infeasibles(instance::MDVRP_Instance, population::Vector{Vector{Int}}) 
    infeasibilities = Vector{Bool}(undef, length(population))
    excess_demands = Vector{Int}(undef, length(population))  #approx
    for i in eachindex(population)
        clusters = chromosome_to_clusters(instance, population[i])
        infeasible = false
        excess_demand = 0
        for depot in eachindex(clusters)
            excess = sum(instance.demands[clusters[depot]]) - instance.M*instance.vehicle_loads[depot]
            if excess > 0
                infeasible = true
                excess_demand += excess
            end
        end
        infeasibilities[i] = infeasible
        excess_demands[i] = excess_demand
    end
    return infeasibilities, excess_demands
end

function repair_infeasibles(instance::MDVRP_Instance, population::Vector{Vector{Int}}, infeasibilities::Vector{Bool}, excess_demands::Vector{Int}, prob_repair::Float64)
    for index in eachindex(infeasibilities)
        if ((infeasibilities[index]==true) && (rand()<prob_repair))
            population[index] = improve_infeasible(instance, population[index])
            excess_demands[index] = 0
            infeasibilities[index]==false
        end
    end
    return population, infeasibilities, excess_demands
end

function is_route_feasible(instance::MDVRP_Instance, cluster::Vector{Int}, depot::Int) ## checks if CVRP capacity is feasible (heuristically)
    current_load = sum(instance.demands[cluster])
    return current_load, (current_load <= instance.M*instance.vehicle_loads[depot])
end

function improve_infeasible(instance::MDVRP_Instance, individual::Vector{Int})
    clusters = chromosome_to_clusters(instance, individual)
    depots = instance.D
    current_loads = Vector{Int}(undef, depots)
    feasibilities = Vector{Bool}(undef, depots)
    for depot = 1:depots
        current_loads[depot], feasibilities[depot] = is_route_feasible(instance, clusters[depot], depot)
    end
    while (sum(feasibilities) < depots)
        depot = argmin(feasibilities)
        customer = rand(clusters[depot])
        deleteat!(clusters[depot], findfirst(x->x==customer,clusters[depot]))
        current_loads[depot], feasibilities[depot] = is_route_feasible(instance, clusters[depot], depot)
        new_depots = findall(==(1), feasibilities)
        for new_depot in new_depots
            if current_loads[new_depot]+instance.demands[customer] <= instance.M*instance.vehicle_loads[new_depot]
                push!(clusters[new_depot], customer)
                current_loads[new_depot] += instance.demands[customer]
                break
            end
        end
    end
    return clusters_to_chromosome(instance, clusters)
end