using Pkg
Pkg.activate(".")
Pkg.instantiate()

using SchellingMF
using ArgParse
using HDF5
# using Plots
using LaTeXStrings
using JSON

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
		"--Nx"
            help = "size of grid"
            arg_type = Int
            default = 128
        "--dx"
        	help = "spatial step size"
        	arg_type = Float64
        	default = 0.3125
        "--dt"
        	help = "temporal step size"
        	arg_type = Float64
        	default = 0.001
        "--t_burn"
            help = "amount of time to run before starting to save"
            arg_type = Float64
            default = 1000.0
        "--t_save"
            help = "amount of time to run after burn-in and save output"
            arg_type = Float64
            default = 1000.0
        "--temp"
            help = "temperature for diffusion constant"
            arg_type = Float64
            default = 0.1
        "--alpha"
            help = "altruism parameter"
            arg_type = Float64
            default = 0.0
        "--delta"
            help = "incompatibility parameter"
            arg_type = Float64
            default = 0.0
        "--kappa"
            help = "symmetric compatibility parameter"
            arg_type= Float64
            default = 0.0
        "--Gamma"
        	help = "strength of bilaplacian"
        	arg_type = Float64
        	default = 1.0
        "--fill"
            help = "fill fraction of each species"
            arg_type = Float64
            nargs = 2
            default = [0.25, 0.25]
        "--snapshot"
            help = "save state of system every snapshot time steps"
            arg_type = Int64
            default = 1
        "--viz_skip"
            help = "how many saved outputs to skip in visualization. If -1, don't make viz"
            arg_type = Int64
            default = 1
        "--savepath"
            help = "where to save output"
            arg_type = String
            default = "."
        "--filename"
            help = "name of save file"
            arg_type = String
            default = "data"

    end

    return parse_args(s)
end

# function load_data(savepath)
#     # load data
#     files = readdir(savepath, join=true)
#     params = JSON.parsefile(files[end])
#     N_saved = length(files) - 1

#     ϕA_array = Array{Float64}(undef, params["Nx"], N_saved)
#     ϕB_array = Array{Float64}(undef, params["Nx"], N_saved)
#     x_array = Array{Float64}(undef, params["Nx"])
#     t_array = Array{Float64}(undef, N_saved)
#     for (fidx, file) in enumerate(files[1:end-1])
#         h5open(file, "r") do d
#             ϕA_array[:, fidx] = read(d["data"], "phiA")
#             ϕB_array[:, fidx] = read(d["data"], "phiB")
#             t_array[fidx] = read(d["data"], "t")
#             if fidx == 1
#                 x_array[:] = read(d["data"], "x")
#             end
#         end
#     end
#     return ϕA_array, ϕB_array, x_array, t_array
# end


# function make_movie(ϕA::Array{Float64, 2}, ϕB::Array{Float64, 2},
#                     x::Array{Float64, 1}, t::Array{Float64, 1},
#                     savepath::String, skip::Int64)

#     # create animation
#     anim = @animate for ii in 1:skip:length(t)
#     plot(x, ϕA[:, ii], lw=2, label=L"A")
#     plot!(x, ϕB[:, ii], lw=2, label=L"B")
#     plot!(ylim=(-0.1, 1.1), xlims=(minimum(x), maximum(x)),
#           framestyle=:box, thickness_scaling=1.5,
#           xlabel=L"x", ylabel=L"\phi^a",
#           title=L"t = " * string(round.(t[ii], digits=2)),
#           leg=:outerright)
#     end

#     mp4(anim, savepath * "_movie.mp4", fps=20)
# end


# function make_kymo(ϕA::Array{Float64, 2}, ϕB::Array{Float64, 2},
#                    x::Array{Float64, 1}, t::Array{Float64, 1},
#                    savepath::String, skip::Int64)

#     hA = heatmap(x, t[1:skip:end], transpose(ϕA[:, 1:skip:end]),
#                  cmap=:Blues, clims=(0, 1),
#                  title=L"ϕ^A", xlabel=L"x", ylabel=L"t")
#     hB = heatmap(x, t[1:skip:end], transpose(ϕB[:, 1:skip:end]),
#                  cmap=:Reds, clims=(0, 1),
#                  title=L"ϕ^B", yformatter=_->"", xlabel=L"x")
#     hC = heatmap(x, t[1:skip:end], transpose(ϕA[:, 1:skip:end] - ϕB[:, 1:skip:end]),
#                  cmap=cgrad(:RdBu), clims=(-1, 1),
#                  title=L"ϕ^A - ϕ^B", yformatter=_->"", xlabel=L"x")
#     p = plot(hA, hB, hC, layout=(1, 3),
#              framestyle=:box, size=(900, 300), legend=true,
#              thickness_scaling=1.5)
#     savefig(p, savepath * "_kymo.pdf")

# end

function main()
    parsed_args = parse_commandline()
    # println(parsed_args)

    # calculate number of time steps for burn in and saving
    Nt_burn = Int(round(parsed_args["t_burn"] / parsed_args["dt"]))
    Nt_save = Int(round(parsed_args["t_save"] / parsed_args["dt"]))

    # simulation parameters
    dx = parsed_args["dx"]
    Nx = parsed_args["Nx"]
    dt = parsed_args["dt"]
    Nt_burn = Nt_burn
    Nt_save = Nt_save
    ϕA0, ϕB0 = parsed_args["fill"]
    α = parsed_args["alpha"]
    δ = parsed_args["delta"]
    κ = parsed_args["kappa"]
    temp = parsed_args["temp"]
    Γ = parsed_args["Gamma"]
    δϕA0 = 0.05 * ϕA0 * (1 - ϕA0 - ϕB0)
    δϕB0 = 0.05 * ϕB0 * (1 - ϕA0 - ϕB0)
    # saving params
    snapshot = parsed_args["snapshot"]
    viz_skip = parsed_args["viz_skip"]
    savepath = string(rstrip(parsed_args["savepath"], ['/']))
    filename = string(parsed_args["filename"])

    # run burn-in
    println("Running burn-in...")
    ϕA, ϕB, t = run_simulation(dx, Nx, dt, Nt_burn, Nt_burn, savepath, filename,
                               ϕA0=ϕA0, ϕB0=ϕB0, δϕA0=δϕA0, δϕB0=δϕB0,
                               α=α, δ=δ, κ=κ, temp=temp, Γ=Γ);

    # run sim after burn-in
    println("Running sim...")
    _, _, _ = run_simulation(ϕA, ϕB, t, dx, Nx, dt, Nt_save,
                             snapshot, savepath, filename,
                             ϕA0=ϕA0, ϕB0=ϕB0, δϕA0=δϕA0, δϕB0=δϕB0,
                             α=α, δ=δ, κ=κ, temp=temp, Γ=Γ);

    # load data
    # ϕA, ϕB, x, t, params = load_data(savepath)

    # if viz_skip > 0
    #     # make visualizations
    #     println("Making movie...")
    #     make_movie(ϕA, ϕB, x, t, savepath, viz_skip)
    #     println("Making kymograph...")
    #     make_kymo(ϕA, ϕB, x, t, savepath, viz_skip)
    # end

    println("Done.")
end

main()

