# Genetic Algorithms with Neural Cost Predictor

Genetic Algorithms with Neural Cost Predictor (GANCP) solve Hierarchical Vehicle Routing Problems (HVRPs) by employing a genetic algorithm to make good-quality higher-level decisions, with the aid of a graph neural network that predicts the cost of VRP subproblems.
GANCP, combined with the [HGS-CVRP solver](https://github.com/vidalt/HGS-CVRP) in the final stage, provides the actual routing solutions to the HVRPs, and is referred to as GANCP<sup>+</sup>. 

This repository contains the code for solving large-scale Multi-Depot Vehicle Routing Problems in `Julia 1.7` using `PyCall` package to perform CVRP cost predictions using GNN implemented in `Python 3.9`. 
If you use GANCP, please cite:
```bibtex
@article{sobhanan2023genetic,
  title={Genetic Algorithms with Neural Cost Predictor for Solving Hierarchical Vehicle Routing Problems},
  author={Sobhanan, Abhay and Park, Junyoung and Park, Jinkyoo and Kwon, Changhyun},
  journal={arXiv preprint arXiv:2310.14157},
  year={2023}
}
```

The MDVRP instances should follow the format of Cordeau MDVRP benchmark instances. Example usage of GANCP solver:

1. Load the functions and the MDVRP instance:
```julia
include("src/read_mdvrp.jl")
include("src/main.jl")
instance = read_mdvrp("sample_instance.txt")
```

2. Find the MDVRP solution: 
```julia
gancp, nn_time, gancp_plus, total_time, mdvrp_routes = mdvrp_solve(instance)
````
Returns the top-five solution candidates obtained using GANCP, along with the predicted costs, GANCP time, GANCP<sup>+</sup> costs, and the total time. 
```julia
julia> print(gancp, "\n", nn_time, "\n", gancp_plus, "\n", total_time)
[69100.37072506937, 69128.4473577427, 69361.76932506775, 69404.22165751646, 69419.3386310017]
55.58299994468689
[69160.87131304345, 69234.75648901977, 69264.09358395533, 69014.79261770715, 69820.23097507822]
105.66499996185303
```


**Parameter Modifications**

1. Modify the device-specific memory limit and HGS-CVRP solver time in `src/parameters.jl`. 

2. Additional GA parameters can be provided as an input as follows:
````julia
include("src/parameters.jl")
hgs_time = find_HGS_time(instance)
gancp, nn_time, gancp_plus, total_time, mdvrp_routes = mdvrp_solve(instance;  weights=[1.0,0.4], pop_limit=[40,50], generations=3, hgs_time_limit=hgs_time, prob_repair=0.8)
````

3. The pre-trained weights of the neural network for the three training phases discussed in the paper are available in `cvrp-cost-prediction/weights`. Note that for more accurate cost predictions, the test data should follow the distribution of the training data.
