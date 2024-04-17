using LinearAlgebra

struct MDVRP_Instance
    customers::Vector{Vector{Float64}}
    depots::Vector{Vector{Float64}}
    vehicle_loads::Vector{Int}
    route_durations::Vector{Int}
    demands::Vector{Int}
    service_times::Vector{Int}
    M::Int #no of vehicles
    N::Int
    D::Int
end

function read_mdvrp(file_name) 
    f = open(file_name)
    lines = readlines(f)

    line = split(lines[1], " ") #vehicles, customers, depots
    M, N, D = parse(Int64, line[2]), parse(Int64, line[3]), parse(Int64, line[4])

    vehicle_loads = Vector{Int64}(undef, D)  # at each depot
    route_durations = Vector{Int64}(undef, D) # for each depot
    for i = 2:D+1
        route_durations[i-1] = parse(Int64, split(lines[i], " ")[1])
        vehicle_loads[i-1]= parse(Int64, split(lines[i], " ")[2])
    end
    
    depots = [Vector{Float64}(undef,2) for _ in 1:D]
    for i = 1+D+N+1:1+D+N+D
        line = split(lines[i], " ")
        j = 2
        
        while isempty(line[j])
            j += 1
        end
        depots[i-N-D-1][1] = parse(Float64,line[j])
        j += 1
        
        while isempty(line[j])
            j += 1
        end
        depots[i-N-D-1][2] = parse(Float64,line[j])
    end

    cus_coords = [Vector{Float64}(undef,3) for _ in 1:N]
    demands = Vector{Int64}(undef, N) 
    service_times = Vector{Int64}(undef, N) 
    
    for i = 1:N
        line = split(lines[1+D+i], " ")
        j = 1
        while isempty(line[j])
            j += 1
        end

        customer = parse(Int64,line[j])
        j += 1
        while isempty(line[j])
            j += 1
        end

        x_coord = parse(Float64,line[j])
        j += 1
        while isempty(line[j])
            j += 1
        end

        y_coord = parse(Float64,line[j])
        j += 1
        while isempty(line[j])
            j += 1
        end
        
        service_time = parse(Float64,line[j])
        j += 1 
        while isempty(line[j])
            j += 1
        end

        dem = parse(Float64,line[j])

        cus_coords[customer] = [x_coord, y_coord]
        demands[customer] = dem
        service_times[customer] = service_time
    end
    return MDVRP_Instance(cus_coords, depots, vehicle_loads, route_durations, demands, service_times, M, N, D)
end