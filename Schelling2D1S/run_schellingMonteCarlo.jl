using Pkg
Pkg.activate(".")
Pkg.instantiate()

include("src/Schelling2D1S.jl")
using .Schelling2D1S
using ArgParse

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
            help = "number of sweeps to run"
            arg_type = Int
            default = 100
        "--temperature", "-T"
            help = "temperature used in Glauber rule"
            arg_type = Float64
            default = 0.01
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
        "--savepath"
            help = "where to save output"
            arg_type = String
            default = "./data.hdf5"
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
    
    state_init = random_state(parsed_args)
    state = run_simulation(state_init, parsed_args)
    # plot(state_init, state, parsed_args)
    save(state_init, state, parsed_args)
end

main()
