using Clustering

function NDA_Clustering(instance::MDVRP_Instance)
    hgs_time = find_HGS_time(instance) 
    # hgs_time = 1.0
    # assign customer to nearest depot
    decomposition = Vector{Int}(undef, instance.N)
    customer_depot_distances = Array{Float64, 2}(undef, instance.N, instance.D)
    for i = 1:instance.N
        for j = 1:instance.D
            customer_depot_distances[i,j] = norm(instance.customers[i]-instance.depots[j])
        end
    end
    decomposition = argmin.(eachrow(customer_depot_distances)) #minimum dist from each customer to closest depot
    infeasibilities, _ = are_infeasibles(instance, [decomposition])
    if infeasibilities[1] == true
        hgs_costs = Inf
    else
        hgs_costs, _ = hgs_solutions(instance, [decomposition], hgs_time)
        hgs_costs = round(hgs_costs[1], digits=2)
    end
    return hgs_costs
end

function KMeans_Clustering(instance)
    hgs_time = find_HGS_time(instance)
    data = hcat(instance.customers...)
    result = kmeans(data, instance.D)
    assignments = result.assignments
    centroids = result.centers'

    # find unique allocation of each centroid to a depot
    centroid_assignments = Vector{Int}(undef, instance.D)
    unallocated_depots = Array(1:instance.D)
    for centroid_idx = 1:size(centroids)[1]
        min_dist = Inf
        for depot in unallocated_depots
            dist = norm(centroids[1,:] - instance.depots[depot])
            if dist < min_dist
                min_dist = dist
                centroid_assignments[centroid_idx] = depot
            end
        end
        index_to_remove = findfirst(x -> x == centroid_assignments[centroid_idx], unallocated_depots)
        deleteat!(unallocated_depots, index_to_remove)
    end
    for i = 1:instance.N
        assignments[i] = centroid_assignments[assignments[i]]
    end
    
    infeasibilities, _ = are_infeasibles(instance, [assignments])
    if infeasibilities[1] == true
        hgs_costs = Inf
    else
        hgs_costs, _ = hgs_solutions(instance, [assignments], hgs_time)
        hgs_costs = round(hgs_costs[1], digits=2)
    end
    # clust = chromosome_to_clusters(instance, assignments)
    return hgs_costs
end