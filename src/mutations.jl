function swap_depots(instance::MDVRP_Instance, individual::Vector{Int})
    customer1 = rand(1:instance.N)
    customer2 = rand(1:instance.N)
    #check selected customer depots are not the same
    while (customer1 == customer2) || (individual[customer1] == individual[customer2]) 
        customer2 = rand(1:instance.N)
    end
    new_individual = copy(individual)
    new_individual[customer1] = individual[customer2]
    new_individual[customer2] = individual[customer1]
    return new_individual
end

# flips the depot of a random customer
function flip_depot(instance::MDVRP_Instance, individual::Vector{Int})
    new_individual = copy(individual)
    index = rand(1:instance.N)
    new_individual[index] = rand(setdiff(Set(1:instance.D), new_individual[index]))
    return new_individual
end

function mutations(instance::MDVRP_Instance, population::Vector{Vector{Int}}, sorted_indices::Vector{Int})
    new_population = Vector{Int}[]
    for i = 1:10
        index = rand(sorted_indices[1:10])
        individual = copy(population[index]) 
        for i = 1:floor(Int, 0.05*instance.N)
            if rand() < 0.5
                #exchange mutations
                new_individual = swap_depots(instance, individual)
            else
                #flip_mutation
                new_individual = flip_depot(instance, individual)
            end
            append!(new_population, [new_individual])
        end
    end
    return new_population
end

function targeted_mutation(instance::MDVRP_Instance, population::Vector{Vector{Int}}, sorted_indices::Vector{Int}, elite_parents::Vector{Vector{Int}})
    pop_length = length(population)
    new_population = Vector{Int}[]
    for i in 1:ceil(0.05*pop_length)
        elite_parent = elite_parents[rand(1:3)]
        individual = copy(population[sorted_indices[minimum(rand(1:pop_length, 2))]])
        customers = unique(rand(1:instance.N, floor(Int, 0.1*instance.N))) #for about 10% of customers
        individual[customers] = elite_parent[customers]
        append!(new_population, [individual])
    end
    return new_population
end