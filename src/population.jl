include("utils.jl")

function generate_population(instance::MDVRP_Instance, max_pop_limit::Int)
    population = Vector{Vector{Int}}(undef, max_pop_limit)

    #parent1: assign customer to nearest depot
    parent1 = Vector{Int}(undef, instance.N)
    customer_depot_distances = Array{Float64, 2}(undef, instance.N, instance.D)
    for i = 1:instance.N
        for j = 1:instance.D
            customer_depot_distances[i,j] = norm(instance.customers[i]-instance.depots[j])
        end
    end
    parent1 = argmin.(eachrow(customer_depot_distances))
    population[1] = parent1
    
    #parent2: assign customer to nearest neighbor's nearest depot
    customer_customer_distances = Array{Float64, 2}(undef, instance.N, instance.N)
    for i = 1:instance.N
        for j = 1:instance.N
            if i == j
                customer_customer_distances[i,j] = Inf
            else
                customer_customer_distances[i,j] = norm(instance.customers[i]-instance.customers[j])
            end
        end
    end
    nearest_neighbor_list = argmin.(eachrow(customer_customer_distances))
    parent2 = Vector{Int}(undef, instance.N)
    for i = 1:instance.N
        parent2[i] = parent1[nearest_neighbor_list[i]]
    end
    population[2] = parent2

    #assign customer to 2nd nearest depot
    if instance.D >= 3
        parent3 = Vector{Int}(undef, instance.N) 
        for i = 1:instance.N
            parent3[i] = findfirst(==(sort(customer_depot_distances[i,:])[2]), customer_depot_distances[i,:])
        end
        population[3] = parent3
    else
        population[3] = rand(1:instance.D, instance.N)
    end
    
    # create true random parents
    for i = 4:max_pop_limit
        population[i] = rand(1:instance.D, instance.N)
    end
    
    return population
end

function generate_subpopulation(instance::MDVRP_Instance, population::Vector{Vector{Int}}, scores::Vector{Float64}, pop_limit::Int)
    subpopulation = Vector{Int}[]
    pop_length = length(scores)
    sorted_scores_indices = sortperm(scores)
    while length(subpopulation)<pop_limit
        parents = Vector{Vector{Int}}(undef, 2)
        #parent1 is selected with either high priority or binary tournament selection
        if rand() < 0.5
            #priority selection
            parents[1] = population[sorted_scores_indices[minimum(rand(1:pop_length, 2))]] 
        else
            #binary tournament selection
            parents[1] = population[sorted_scores_indices[minimum(rand(1:pop_length, 2))]]
        end
        #add parent2
        parents[2] = population[sorted_scores_indices[minimum(rand(1:pop_length, 2))]]
        
        while parents[1] == parents[2]
            parents[2] = population[sorted_scores_indices[minimum(rand(1:pop_length, 2))]]
        end
        child1 = Vector{Int}(undef, instance.N)
        child2 = Vector{Int}(undef, instance.N)
        for index = 1:instance.N
            child1[index] = parents[rand(1:2)][index]
            child2[index] = parents[rand(1:2)][index]
        end

        append!(subpopulation, [child1])
        append!(subpopulation, [child2])
    end
    # delete duplicates in subpopulation
    subpopulation = unique(subpopulation)
    return subpopulation
end