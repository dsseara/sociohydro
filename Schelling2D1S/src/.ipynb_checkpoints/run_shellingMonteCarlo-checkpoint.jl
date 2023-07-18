using Pkg
Pkg.activate(".")
include("shellingMonteCarlo.jl")
using .shelling
using ArgParse
using PyPlot

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
            default = "local"
        "--alpha"
            help = "altruism parameter"
            arg_type = Float64
            default = 0.0
        "--fill"
            help = "fill fraction of total spots"
            arg_type = Float64
            default = 0.5
    end

    return parse_args(s)
end

function plot(state_init, state_final, params)
    # unpack params
    capacity = params["capacity"]
    n_sweeps = params["n_sweeps"]
    
    fig, ax = plt.subplots(1, 3, dpi=150, figsize=(6, 3))

    a1 = ax[1].imshow(state_init ./ capacity,
                      cmap="Blues",
                      vmin=0,
                      vmax=1)
    c1 = fig.colorbar(a1, ax=ax[1], shrink=0.35)
    c1.ax.set(title=L"$\rho$")
    ax[1].set(title="sweep 0", xticks=[], yticks=[])

    a2 = ax[2].imshow(state_final ./ capacity,
                      cmap="Reds",
                      vmin=0,
                      vmax=1)
    c2 = fig.colorbar(a2, ax=ax[2], shrink=0.35)
    c2.ax.set(title=L"$\rho$")
    ax[2].set(title="sweep " * string(n_sweeps), xticks=[], yticks=[])

    ax[3].hist(state_init[:] ./ capacity,
               alpha=0.5,
               density=true,
               facecolor="C0",
               edgecolor="k",
               label="init")
    ax[3].hist(state_final[:] ./ capacity,
               alpha=0.5,
               density=true,
               facecolor="C3",
               edgecolor="k",
               label="final")
    ax[3].set(xlim=[0, 1],
              xticks=[0, 0.5, 1],
              xlabel="density",
              ylabel="pdf")
    
    ax[3].set_aspect((ax[3].set_xlim()[2] - ax[3].set_xlim()[1]) / (ax[3].set_ylim()[2] - ax[3].set_ylim()[1]))

    plt.tight_layout()
    plt.show()
end


function main()
    parsed_args = parse_commandline()
    state_init = random_state(parsed_args)
    state = run_simulation(state_init, parsed_args)
    plot(state_init, state, parsed_args)
end

main()