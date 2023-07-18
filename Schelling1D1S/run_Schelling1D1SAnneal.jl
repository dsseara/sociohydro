using Pkg
Pkg.activate(".")
Pkg.instantiate()

include("src/Schelling1D1S.jl")
using .Schelling1D1S
using ArgParse
using PyPlot
using Printf

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--grid_size", "-g"
            help = "size of grid"
            default = 32
            arg_type = Int
        "--capacity", "-c"
            help = "maximum occupancy of each lattice site"
            arg_type = Int
            default = 50
        "--preferred_density", "-p"
            help = "preferred density for asymmetric utility function"
            arg_type = Float64
            default = 0.5
        "--n_sweeps", "-n"
            help = "number of sweeps to run for each temperature"
            arg_type = Int
            default = 100
        "--m", "-m"
            help = "utility of rho = 1 state"
            arg_type = Float64
            default = 0.5
        "--moveType"
            help = "whether to move randomly or locally"
            arg_type = String
            default = "random"
        "--alpha"
            help = "altruism parameter"
            arg_type = Float64
            default = 0.0
        "--fill"
            help = "fill fraction of total spots"
            arg_type = Float64
            default = 0.5
        "--utility_func"
            help = "either asymmetric_utility or quadratic_utility"
            arg_type = String
            default = "asymmetric_utility"
        "--T_init"
            help = "initial temperature used in Glauber rule"
            arg_type = Float64
            default = 0.5
        "--T_step"
            help = "size of temperature steps when annealing"
            arg_type = Float64
            default = 0.05
        "--T_final"
            help = "final temperature of anneal"
            arg_type = Float64
            default = 0.0
        "--snapshot"
            help = "how often to save output during each simulation"
            arg_type = Int64
            default = -1
        "--savepath"
            help = "where to save output"
            arg_type = String
            default = "."
    end

    return parse_args(s)
end



function main()
    parsed_args = parse_commandline()
    println(parsed_args)
    # replace string with function name
    if parsed_args["utility_func"] == "asymmetric_utility"
        parsed_args["utility_func"] = asymmetric_utility
    elseif parsed_args["utility_func"] == "quadratic_utility"
        parsed_args["utility_func"] = quadratic_utility
    end
    
    # initial state
    state = random_state(parsed_args)
    parsed_args["temperature"] = parsed_args["T_init"]

    while parsed_args["temperature"] >= parsed_args["T_final"]
        filename = @sprintf("temp%0.3f.hdf5", parsed_args["temperature"])
        run_simulation!(state, parsed_args)
        save(state, parsed_args, parsed_args["savepath"], filename)
        parsed_args["temperature"] -= parsed_args["T_step"]
    end
        
end

main()
