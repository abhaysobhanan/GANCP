include("population.jl")
include("utils.jl")
include("evaluate_sol.jl")
include("mutations.jl")

function mdvrp_solve(instance::MDVRP_Instance; weights=[1.0,0.4], pop_limit=[30,40], generations=5, hgs_time_limit=1.0, prob_repair=0.5)
    start_time = time()
    population = generate_population(instance, 10)

    infeasibilities, excess_demands = are_infeasibles(instance, population)  # boolean vector
    population, infeasibilities, excess_demands = repair_infeasibles(instance, population, infeasibilities, excess_demands, prob_repair)
    
    costs = batch_fitness_cost(instance, population)
    #fitness evaluation
    diversities = diversity(population)
    scores = fitness_scores(costs, diversities, weights[1], weights[2], excess_demands)

    sorted_indices_pop = sortperm(costs)
    top_solutions = population[sorted_indices_pop[1:2]]
    top_costs = costs[sorted_indices_pop[1:2]]
    elite_parents = deepcopy(population[sorted_indices_pop[1:3]])

    #subpopulation becomes main population
    for generation = 1:generations  
        subpopulation = generate_subpopulation(instance, population, scores, pop_limit[1])
        infeasibilities, excess_demands = are_infeasibles(instance, subpopulation)
        subpopulation, infeasibilities, excess_demands = repair_infeasibles(instance, subpopulation, infeasibilities, excess_demands, prob_repair)
        subcosts = batch_fitness_cost(instance, subpopulation)
        diversities = diversity(subpopulation)
        sub_scores = fitness_scores(subcosts, diversities, weights[1], weights[2], excess_demands)

        #targeted mutations  
        sorted_indices = sortperm(sub_scores)
        new_population_targM = targeted_mutation(instance, subpopulation, sorted_indices, elite_parents)
        #additional random mutations
        new_population_randM = mutations(instance, subpopulation, sorted_indices)
        new_population = vcat(new_population_targM, new_population_randM)
        new_costs = batch_fitness_cost(instance, new_population)
        new_infeas, new_excess_dem = are_infeasibles(instance, new_population)

        subpopulation = vcat(subpopulation, new_population)
        subcosts = vcat(subcosts, new_costs)
        infeasibilities = vcat(infeasibilities, new_infeas)
        excess_demands = vcat(excess_demands, new_excess_dem)

        #save the best solutions from the current generation
        sorted_indices = sortperm(subcosts)
        top_solutions = vcat(top_solutions, subpopulation[sorted_indices[1:2]])  
        top_costs = vcat(top_costs, subcosts[sorted_indices[1:2]])

        if generation < generations
            #preserve top 1% of original population
            sorted_indices_pop = sortperm(costs)
            append!(subpopulation, population[sorted_indices_pop[1:ceil(Int, 0.01*length(population))]])
            append!(subcosts, costs[sorted_indices_pop[1:ceil(Int, 0.01*length(population))]])

            #subpopulation becomes population
            population, costs = delete_duplicates(subpopulation, subcosts)
            diversities = diversity(population)
            infeasibilities, excess_demands = are_infeasibles(instance, population)
            scores = fitness_scores(costs, diversities, weights[1], weights[2], excess_demands)
            sorted_scores_indices = sortperm(scores)

            pop_size = minimum([length(population), pop_limit[2]])
            population = population[sorted_scores_indices[1:pop_size]] 
            costs = costs[sorted_scores_indices[1:pop_size]]

            scores = scores[sorted_scores_indices[1:pop_size]]
            sorted_indices_pop = sortperm(scores)
        end
        # print("*** Generation ", generation, " completed.\n")
    end
    nn_time = time()-start_time   ## NN heuristic time

    top_solutions, top_costs = delete_duplicates(top_solutions, top_costs)
    top_infeas, _ = are_infeasibles(instance, top_solutions)
    top_solutions = top_solutions[.!top_infeas]
    top_costs = top_costs[.!top_infeas]
    sorted_topcost_indices = sortperm(top_costs) 
    top_solutions = top_solutions[sorted_topcost_indices[1:minimum([5,length(top_costs)])]]  #5
    nn_costs = top_costs[sorted_topcost_indices[1:minimum([5,length(top_costs)])]]

    hgs_costs, mdvrp_routes = hgs_solutions(instance, top_solutions, hgs_time_limit)
    total_time = time() - start_time   ## GA+NN+HGS total time
    
    return nn_costs, nn_time, hgs_costs, total_time, mdvrp_routes
end