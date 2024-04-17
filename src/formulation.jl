using JuMP, Gurobi
include("read_mdvrp.jl")

"""
Attempt to solve using the exact formulation only for very small mdvrp instances.
"""

instance = read_mdvrp("../a_small_instance.txt")
coords = vcat(instance.customers, instance.depots)
distances = Array{Float64, 2}(undef, instance.N+instance.D, instance.N+instance.D)
for i = 1:instance.N+instance.D
    for j = 1:instance.N+instance.D
        distances[i,j] = norm(coords[i]-coords[j])
    end
    distances[i,i] = Inf
end

model = Model(Gurobi.Optimizer)
N = instance.N
D = instance.D
V = N + D
K = instance.M
K_total = K*D
@variable(model, x[1:V, 1:V, 1:K_total] >= 0, Bin)
@variable(model, u[1:N] >= 0)

@objective(model, Min, sum(x[i,j,k]*distances[i,j] for i=1:V, j=1:V, k=1:K_total))

for j = 1:N
    @constraint(model, sum(x[i, j, k] for i=1:V, k=1:K_total) == 1)
    # @constraint(model, sum(x[j, i, k] for i=1:V, k=1:K_total) == 1)
end
for j = 1:V
    for k = 1:K_total
        @constraint(model, sum(x[j, i, k] for i=1:V) == sum(x[i, j, k] for i=1:V))
    end
end

for k = 1:K_total
    @constraint(model, sum(instance.demands[i]*x[i, j, k] for i=1:N, j=1:V) <= instance.vehicle_loads[1])
    for i = 1:N
        for j = 1:N
            if i != j
                @constraint(model, (u[i] - u[j] + N*x[i,j,k]) <= N-1)
            end
        end
    end
end

for i = N+1:V
    @constraint(model, sum(x[i, j, k] for j=N+1:V, k=1:K_total) == 0)   ## no depot to depot transit
    for k = 1:K_total
        if k < (i-N-1)*K+1 || k > (i-N-1)*K+K
            # @constraint(model, sum(x[i, j, k] for j=1:N) == 0)
            @constraint(model, sum(x[j, i, k] for j=1:N) == 0)
        end
    end
    for k = (i-N-1)*K+1:(i-N-1)*K+K
        # @constraint(model, sum(x[i, j, k] for j=1:N) <= 1)
        @constraint(model, sum(x[j, i, k] for j=1:N) <= 1)
    end
end

set_time_limit_sec(model, 600.0) # seconds
optimize!(model)