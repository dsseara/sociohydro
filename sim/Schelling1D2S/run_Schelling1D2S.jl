using Pkg
Pkg.activate(".")
Pkg.instantiate()

include("src/Schelling1D2S.jl")
using .Schelling1D2S
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
        "--n_sweeps", "-n"
            help = "number of sweeps to run"
            arg_type = Int
            default = 100
        "--temperature", "-T"
            help = "temperature used in Glauber rule"
            arg_type = Float64
            default = 0.1
        "--alpha"
            help = "altruism parameter"
            arg_type = Float64
            default = 0.0
        "--fill"
            help = "fill fraction of each species"
            arg_type = Float64
            nargs = 2
            default = [0.25, 0.25]
        "--snapshot"
            help = "save state of system every snapshot sweeps. -1 = save init and final configs"
            arg_type = Int64
            default = -1
        "--delta"
            help = "non-reciprocity parameter"
            arg_type = Float64
            default = 0.0
        "--kappa"
            help = "reciprocity parameter"
            arg_type = Float64
            default = 0.0
        "--dt"
            help = "size of time step for sweep"
            arg_type = Float64
            default = 0.01
        "--savepath"
            help = "where to save output"
            arg_type = String
            default = "."
    end

    return parse_args(s)
end


function main()
    println("here")
    parsed_args = parse_commandline()
    println(parsed_args)
    state = random_state(parsed_args)
    run_simulation!(state, parsed_args)
end

main()
