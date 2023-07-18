using Pkg
Pkg.activate(".")
Pkg.instantiate()

include("src/Schelling2D2S.jl")
using .Schelling2D2S
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
            help = "fill fraction of each species"
            arg_type = Float64
            nargs = 2
            default = [0.25, 0.25]
        "--utility_func"
            help = "either asymmetric_utility or quadratic_utility"
            arg_type = String
            default = "asymmetric_utility"
        "--savepath"
            help = "where to save output"
            arg_type = String
            default = "."
        "--delta"
            help = "non-reciprocity parameter"
            arg_type = Float64
            default = 0.0
        "--snapshot"
            help = "save state of system every snapshot sweeps. -1 means only save initial and final configs"
            arg_type = Int64
            default = -1
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
    
    state = random_state(parsed_args)
    run_simulation!(state, parsed_args)
    # plot(state_init, state, parsed_args)
    # save(state_init, state, parsed_args)
end

main()
