module SchellingMF

using LinearAlgebra
using Distributions
using StatsBase
using Printf
using HDF5
using JSON
using ProgressMeter

export run_simulation, update!, load_data

function calc_grad(field::Array{T, 2}, dx::T, periodic::Bool;
                   enforce_bcs::Bool=true, bcval::T=0.0,
                   order::Int64=4) where T<:AbstractFloat
    ∇xfield = similar(field)
    ∇yfield = similar(field)
    Ny, Nx = size(field)

    # create stencils
    if order==4
        central_stencil = [1, -8, 0, 8, -1] / 12
        forward_stencil = [-25, 48, -36, 16, -3] / 12
    elseif order==2
        central_stencil = [-1, 0, 1] / 2
        forward_stencil = [-3, 2, -1] / 2
    end

    backward_stencil = -reverse(forward_stencil)

    cs_halfwidth = length(central_stencil) ÷ 2
    fs_width = length(forward_stencil)
    bs_width = fs_width

    # central differences in center region
    for x in 1+cs_halfwidth:Nx-cs_halfwidth
        for y in 1+cs_halfwidth:Ny-cs_halfwidth
            indx = x-cs_halfwidth:x+cs_halfwidth
            indy = y-cs_halfwidth:y+cs_halfwidth
            ∇xfield[y, x] = sum(field[y, indx] .* central_stencil)
            ∇yfield[y, x] = sum(field[indy, x] .* central_stencil)
        end
    end

    if periodic # do central differences at edges
        # vertical edges, include corners
        for x in [1:cs_halfwidth Nx-cs_halfwidth:Nx]
            for y in 1:Ny
                indx = mod1.(x-cs_halfwidth:x+cs_halfwidth, Nx)
                indy = mod1.(y-cs_halfwidth:y+cs_halfwidth, Ny)
                ∇xfield[y, x] = sum(field[y, indx] .* central_stencil)
                ∇yfield[y, x] = sum(field[indy, x] .* central_stencil)
            end
        end
        # horizontal edges, exclude corners
        for y in [1:cs_halfwidth Ny-cs_halfwidth:Ny]
            for x in 1+cs_halfwidth:Nx-cs_halfwidth
                indx = mod1.(x-cs_halfwidth:x+cs_halfwidth, Nx)
                indy = mod1.(y-cs_halfwidth:y+cs_halfwidth, Ny)
                ∇xfield[y, x] = sum(field[y, indx] .* central_stencil)
                ∇yfield[y, x] = sum(field[indy, x] .* central_stencil)
            end
        end

    else # do forward/backward differences at ends
        ### edges ###
        # left edge, forward difference on x, central on y
        for x in 1:cs_halfwidth
            for y in 1+cs_halfwidth:Ny-cs_halfwidth
                indx = x:x+fs_width
                indy = y-cs_halfwidth:y+cs_halfwidth
                ∇xfield[y, x] = sum(field[y, indx] .* forward_stencil)
                ∇yfield[y, x] = sum(field[indy, x] .* central_stencil)
            end
        end
        # right edge, backward difference on x, central on y
        for x in Nx-cs_halfwidth+1:Nx
            for y in 1+cs_halfwidth:Ny-cs_halfwidth
                indx = x-bs_width:x
                indy = y-cs_halfwidth:y+cs_halfwidth
                ∇xfield[y, x] = sum(field[y, indx] .* backward_stencil)
                ∇yfield[y, x] = sum(field[indy, x] .* central_stencil)
            end
        end
        # bottom edge, forward difference on y, central on x
        for y in 1:cs_halfwidth
            for x in 1+cs_halfwidth:Nx-cs_halfwidth
                indx = x-cs_halfwidth:x+cs_halfwidth
                indy = y:y+fs_width
                ∇xfield[y, x] = sum(field[y, indx] .* central_stencil)
                ∇yfield[y, x] = sum(field[indy, x] .* forward_stencil)
            end
        end
        # top edge, backward difference on y, central on x
        for y in Ny-cs_halfwidth+1:Ny
            for x in 1+cs_halfwidth:Nx-cs_halfwidth
                indx = x-cs_halfwidth:x+cs_halfwidth
                indy = y-bs_width:y
                ∇xfield[y, x] = sum(field[y, indx] .* central_stencil)
                ∇yfield[y, x] = sum(field[indy, x] .* backward_stencil)
            end
        end

        ### corners ###
        # bottom left
        # forward difference for x and y
        for x in 1:cs_halfwidth
            for y in 1:cs_halfwidth
                indx = x:x+fs_width
                indy = y:y+fs_width
                ∇xfield[y, x] = sum(field[y, indx] .* forward_stencil)
                ∇yfield[y, x] = sum(field[indy, x] .* forward_stencil)
            end
        end

        # bottom right
        # backward difference for x, forward difference for y
        for x in Nx-cs_halfwidth+1:Nx
            for y in 1:cs_halfwidth
                indx = x-bs_width:x
                indy = y:y+fs_width
                ∇xfield[y, x] = sum(field[y, indx] .* backward_stencil)
                ∇yfield[y, x] = sum(field[indy, x] .* forward_stencil)
            end
        end

        # top left
        # forward difference for x, backward difference for y
        for x in 1:cs_halfwidth
            for y in Ny-cs_halfwidth+1:Ny
                indx = x:x+fs_width
                indy = y-bs_width:y
                ∇xfield[y, x] = sum(field[y, indx] .* forward_stencil)
                ∇yfield[y, x] = sum(field[indy, x] .* backward_stencil)
            end
        end

        # top right
        # backward difference for x and y
        for x in Nx-cs_halfwidth+1:Nx
            for y in Ny-cs_halfwidth+1:Ny
                indx = x-bs_width:x
                indy = y-bs_width:y
                ∇xfield[y, x] = sum(field[y, indx] .* backward_stencil)
                ∇yfield[y, x] = sum(field[indy, x] .* backward_stencil)
            end
        end

        # enforce boundary conditions
        if enforce_bcs
            # left
            ∇xfield[1:end, 1] .= bcval
            # right
            ∇xfield[1:end, end] .= bcval
            # bottom
            ∇yfield[1, 1:end] .= bcval
            # top
            ∇yfield[end, 1:end] .= bcval
        end
    end

    return ∇xfield ./ dx, ∇yfield ./ dx
end

function calc_lap(field::Array{T, 2}, dx::T, periodic::Bool) where T<:AbstractFloat
    ∇²field = similar(field)
    Ny, Nx = size(field)

    # central differences in center region
    # Do order: up, right, center, down, left
    for x in 2:Nx-1
        for y in 2:Ny-1
            ∇²field[y, x] = field[y+1, x] + field[y, x+1] - 4 * field[y, x] + field[y-1, x] + field[y, x-1]
        end
    end

    if periodic # do central differences at edges
        ### corners ###
        # bottom left
        ∇²field[1, 1] = field[2, 1] + field[1, 2] - 4 * field[1, 1] + field[Ny, 1] + field[1, Nx]

        # bottom right
        ∇²field[1, Nx] = field[2, Nx] + field[1, 1] - 4 * field[1, Nx] + field[Ny, Nx] + field[1, Nx-1]

        # top left
        ∇²field[Ny, 1] = field[1, 1] + field[Ny, 2] - 4 * field[Ny, 1] + field[Ny-1, 1] + field[Ny, Nx]

        # top right
        ∇²field[Ny, Nx] = field[1, Nx] + field[Ny, 1] - 4 * field[Ny, Nx] + field[Ny-1, Nx] + field[Ny, Nx-1]

        ### edges ###
        # vertical
        for y in 2:Ny-1
            # left
            ∇²field[y, 1] = field[y+1, 1] + field[y, 2] - 4 * field[y, 1] + field[y-1, 1] + field[y, Nx]
            # right
            ∇²field[y, Nx] = field[y+1, Nx] + field[y, 1] - 4 * field[y, Nx] + field[y-1, Nx] + field[y, Nx-1]
        end

        # horizontal
        for x in 2:Nx-1
            # bottom
            ∇²field[1, x] = field[2, x] + field[1, x+1] - 4 * field[1, x] + field[Ny, x] + field[1, x-1]
            # top
            ∇²field[Ny, x] = field[1, x] + field[Ny, x+1] - 4 * field[Ny, x] + field[Ny-1, x] + field[Ny, x-1]
        end

    else # do forward/backward differences at edges
        ### corners ###
        # bottom left
        ∇²field[1, 1] = field[3, 1] - 2 * field[2, 1] + 2 * field[1, 1] - 2 * field[1, 2] + field[1, 3]

        # bottom right
        ∇²field[1, Nx] = field[1, Nx-2] - 2 * field[1, Nx-1] + 2 * field[1, Nx] - 2 * field[2, Nx] + field[3, Nx]

        # top left
        ∇²field[Ny, 1] = field[Ny-2, 1] - 2 * field[Ny-1, 1] + 2 * field[Ny, 1] - 2 * field[Ny, 2] + field[Ny, 3]

        # top right
        ∇²field[Ny, Nx] = field[Ny, Nx-2] - 2 * field[Ny, Nx-1] + 2 * field[Ny, Nx] - 2 * field[Ny-1, Nx] + field[Ny-2, Nx]

        ### edges ###
        # vertical
        for y in 2:Ny-1
            # left
            ∇²field[y, 1] = field[y+1, 1] - field[y, 1] + field[y-1, 1] - 2 * field[y, 2] + field[y, 3]
            # right
            ∇²field[y, Nx] = field[y+1, Nx] - field[y, Nx] + field[y-1, Nx] - 2 * field[y, Nx-1] + field[y, Nx-2]
        end

        # horizontal
        for x in 2:Nx-1
            # bottom
            ∇²field[1, x] = field[1, x-1] - field[1, x] + field[1, x+1] - 2 * field[2, x] + field[3, x]
            # top
            ∇²field[Ny, x] = field[Ny, x-1] - field[Ny, x] + field[Ny, x+1] - 2 * field[Ny-1, x] + field[Ny-2, x]
        end
    end

    return ∇²field ./ dx^2
end


function calc_grad3(field::Array{T, 2}, dx::T, periodic::Bool;
                    enforce_bcs::Bool=true, bcval::T=0.0) where T<:AbstractFloat
    ∇²field = calc_lap(field, dx, periodic)
    ∇³xfield, ∇³yfield = calc_grad(∇²field, dx, periodic;
                                   enforce_bcs=enforce_bcs,
                                   bcval=bcval)
    return ∇³xfield, ∇³yfield
end


function calc_div(fieldx::Array{T, 2}, fieldy::Array{T, 2}, dx::T, periodic::Bool) where T<:AbstractFloat
    ∂x_fieldx, ∂y_fieldx = calc_grad(fieldx, dx, periodic)
    ∂x_fieldy, ∂y_fieldy = calc_grad(fieldy, dx, periodic)
    return ∂x_fieldx + ∂y_fieldy
end


function calc_J(ϕA::Array{T, 2}, ϕB::Array{T, 2}, dx::T, periodic::Bool,
                utility_params::Array{T, 1}, D::T, Γ::T) where T<:AbstractFloat
    # preallocation
    JAx = similar(ϕA)
    JAy = similar(ϕA)
    JBx = similar(ϕB)
    JBy = similar(ϕB)

    πA, πB = utility_func(ϕA, ϕB, utility_params)
    ∇xπA, ∇yπA = calc_grad(πA, dx, periodic)
    ∇xπB, ∇yπB = calc_grad(πB, dx, periodic)
    ∇xϕA, ∇yϕA = calc_grad(ϕA, dx, periodic)
    ∇xϕB, ∇yϕB = calc_grad(ϕB, dx, periodic)
    ∇³xϕA, ∇³yϕA = calc_grad3(ϕA, dx, periodic)
    ∇³xϕB, ∇³yϕB = calc_grad3(ϕB, dx, periodic)

    JAx = -D * ∇xϕA + ϕA * ∇xπA + Γ * ϕA * ∇³xϕA
    JAy = -D * ∇yϕA + ϕA * ∇yπA + Γ * ϕA * ∇³yϕA
    JBx = -D * ∇xϕB + ϕB * ∇xπB + Γ * ϕB * ∇³xϕB
    JBy = -D * ∇yϕB + ϕB * ∇yπB + Γ * ϕB * ∇³yϕB

    return JAx, JAy, JBx, JBy
end


function calc_force(ϕA::Array{T, 2}, ϕB::Array{T, 2}, dx::T, periodic::Bool,
                    utility_params::Array{T, 1},
                    D::T, Γ::T) where T<:AbstractFloat

    JAx, JAy, JBx, JBy = calc_J(ϕA, ϕB, dx, periodic,
                                utility_params,
                                D, Γ)
    FA = -calc_div(JAx, JAy, dx, periodic)
    FB = -calc_div(JBx, JBy, dx, periodic)

    return FA, FB
end


function rk4(field::Array{T, 2}, dt::T, force::Function) where T<:AbstractFloat
    k1 = dt * force(field)
    k2 = dt * force(field + 0.5 * k1)
    k3 = dt * force(field + 0.5 * k2)
    k4 = dt * force(field + k3)
    return field + (k1 + 2 * k2 + 2 * k3 + k4) / 6
end


function update!(ϕA::Array{T, 2}, ϕB::Array{T, 2}, dx::T, dt::T,
                 periodic::Bool, utility_params::Array{T, 1},
                 D::T, Γ::T) where T<:AbstractFloat

    # calculate forces
    FA, FB = calc_force(ϕA, ϕB, dx, periodic,
                        utility_params,
                        D, Γ)

    # update
    ϕA += dt * FA
    ϕB += dt * FB

    return ϕA, ϕB
end


function random_state(Nx::Int64,
                      ϕA0::T, ϕB0::T,
                      δϕA0::T, δϕB0::T) where T<:AbstractFloat
    ϕA = ϕA0 .+ rand(Uniform(-1, 1), Nx, Nx) .* δϕA0
    ϕB = ϕB0 .+ rand(Uniform(-1, 1), Nx, Nx) .* δϕB0

    return ϕA, ϕB
end


# This version starts with a random initial condition
function run_simulation(dx::T, Nx::Int64, dt::T, Nt::Int64,
                        utility_params::Array{T, 1},
                        snapshot::Int64, savepath::String;
                        periodic::Bool=true,
                        ϕA0::T = 0.25, ϕB0::T = 0.25,
                        δϕA0::T = 0.05, δϕB0::T = 0.05,
                        D::T = 0.1, Γ::T = 1.0) where T<:AbstractFloat

    ϕA, ϕB = random_state(Nx, ϕA0, ϕB0, δϕA0, δϕB0)
    t_init = 0.0

    ϕA, ϕB, t = run_simulation(ϕA, ϕB, t_init,
                               dx, Nx, dt, Nt,
                               utility_params,
                               snapshot, savepath,
                               periodic=periodic,
                               D=D, Γ=Γ)
    return ϕA, ϕB, t
end

# this version starts with a given initial condition
function run_simulation(ϕA::Array{T, 2}, ϕB::Array{T, 2}, t_init::T,
                        dx::T, Nx::Int64, dt::T, Nt::Int64,
                        utility_params::Array{T, 1},
                        snapshot::Int64, savepath::String;
                        periodic::Bool=true,
                        D::T = 0.1, Γ::T = 1.0) where T<:AbstractFloat


    if dt * dx^(-4) > 1/8
        @warn "von Neumann stability criterion not satisfied"
    end

    # get spatial variable
    L = Nx * dx
    x = y = collect(-L/2 + dx/2:dx:(L/2) - dx/2)

    # ensure we have a clean directory to dump data into
    if isdir(savepath)
        for file in readdir(savepath, join=true)
            rm(file)
        end
    else
        mkpath(savepath)
    end

    params = Dict("dx" => dx,
                  "Nx" => Nx,
                  "dt" => dt,
                  "Nt" => Nt,
                  "snapshot" => snapshot,
                  "savepath" => savepath,
                  "utility_params" => utility_params,
                  "D" => D,
                  "Γ" => Γ)

    open(savepath * "/params.json", "w") do f
        JSON.print(f, params, 4)
    end

    t::Float64 = t_init
    # save initial condition
    npad = Int(ceil(log10(Nt)) + 1)
    save(ϕA, ϕB, x, y, t, 0, npad, savepath)

    # main loop
    println("Starting main loop...")
    @showprogress for ii in 1:Nt
        ϕA, ϕB = update!(ϕA, ϕB, dx, dt,
                         periodic, utility_params,
                         D, Γ)
        t += dt

        if ii % snapshot == 0
            if nan_check(ϕA)
                save(ϕA, ϕB, x, y, t, ii, npad, savepath)
            else
                throw(OverflowError("got nans"))
            end
        end
    end
    return ϕA, ϕB, t
end

function save(ϕA::Array{T, 2}, ϕB::Array{T, 2},
              x::Array{T, 1}, y::Array{T, 1}, t::T,
              step::Int64, npad::Int64, savepath::String) where T<:AbstractFloat

    filename = "n" * lpad(string(step), npad, '0') * ".hdf5"

    if Sys.isunix()
        filesep = "/"
    else
        filesep = "\\"
    end

    filepath = savepath * filesep * filename

    h5open(filepath, "w") do fid
        d = create_group(fid, "data")
        d["phiA"] = ϕA
        d["phiB"] = ϕB
        d["x"] = x
        d["y"] = y
        d["t"] = t
    end
end

function nan_check(x::Array{T, 2}) where T<:AbstractFloat
    return all(isfinite, x)
end


function utility_func(ϕA::Array{T, 2}, ϕB::Array{T, 2},
                      params::Array{T, 1}) where T<:AbstractFloat
    kaa, kab, kba, kbb = params
    πA = @. kaa * ϕA + kab * ϕB
    πB = @. kba * ϕA + kbb * ϕB

    return πA, πB
end


function load_data(savepath)
    # load data
    files = readdir(savepath, join=true)
    params = JSON.parsefile(files[end])
    N_saved = length(files) - 1

    ϕA_array = Array{Float64}(undef, params["Nx"], params["Nx"], N_saved)
    ϕB_array = Array{Float64}(undef, params["Nx"], params["Nx"], N_saved)
    x_array = Array{Float64}(undef, params["Nx"])
    y_array = Array{Float64}(undef, params["Nx"])
    t_array = Array{Float64}(undef, N_saved)
    for (fidx, file) in enumerate(files[1:end-1])
        h5open(file, "r") do d
            ϕA_array[:, :, fidx] = read(d["data"], "phiA")
            ϕB_array[:, :, fidx] = read(d["data"], "phiB")
            t_array[fidx] = read(d["data"], "t")
            if fidx == 1
                x_array[:] = read(d["data"], "x")
                y_array[:] = read(d["data"], "y")
            end
        end
    end
    return ϕA_array, ϕB_array, x_array, y_array, t_array, params
end


# """
# enforce_bcs(field::Array{T, 2})

# We are enfocing the following at the boundaries:
#     ∇ϕ⋅n̂ = 0
#     ∇³ϕ⋅n̂ = 0

# Let ϕ(j, i) denote the field at each grid point, with
# i = {1, Nx} and j = {1, Ny} denoting the edges.

# Given our central difference scheme, these BCs
# translate to (on the left edge)
#     ϕ(j, 1) = ϕ(j, 5)
#     ϕ(j, 2) = ϕ(j, 4)

# Other edges are done similarly.

# Corners need to be set appropriately to cancel
# out the 3rd order derivatives

# This function assumes that the field has
# been padded with 2 "ghost zones" on every edge
# """
# function enforce_bcs!(field::Array{T, 2}; c::T = 0.1) where T<:AbstractFloat
#     Nx, Ny = size(field)

#     # go along x
#     for ii in 3:Nx-2
#         # bottom
#         field[1, ii] = field[5, ii]
#         field[2, ii] = field[4, ii]
#         # top
#         field[Ny, ii] = field[Ny-4, ii]
#         field[Ny-1, ii] = field[Ny-3, ii]
#     end
#     # go along y
#     for jj in 3:Ny-2
#         # left
#         field[jj, 1] = field[jj, 5]
#         field[jj, 2] = field[jj, 4]
#         # right
#         field[jj, Nx] = field[jj, Nx-4]
#         field[jj, Nx-1] = field[jj, Nx-3]
#     end

#     # corners
#     # bottom left
#     field[1, 1] = field[1, 2] = field[2, 1] = field[2, 2] = field[4, 4]
#     # bottom right
#     field[1, Nx] = field[1, Nx-1] = field[2, Nx] = field[2, Nx-1] = field[4, Nx-3]
#     # top left
#     field[Ny, 1] = field[Ny-1, 1] = field[Ny, 2] = field[Ny-1, 2] = field[Ny-3, 4]
#     # top right
#     field[Ny, Nx] = field[Ny-1, Nx] = field[Ny, Nx-1] = field[Ny-1, Nx-1] = field[Ny-3, Nx-3]
# end



function fitness(ϕA::Array{T, 1}, ϕB::Array{T, 1},
                 δ::T, κ::T) where T<:AbstractFloat
    # fitness
    πA = @. f(ϕA) + (κ - δ) * ϕB / 2
    πB = @. f(ϕB) + (κ + δ) * ϕA / 2

    # fitness derivatives
    dπA_dϕA = @. df(ϕA)
    dπA_dϕB = @. (κ - δ) / 2
    dπB_dϕA = @. (κ + δ) / 2
    dπB_dϕB = @. df(ϕB)

    # global fitness
    U = @. ϕA * πA + ϕB * πB

    # global fitness derivatives
    dU_dϕA = @. πA + ϕA * dπA_dϕA + ϕB * dπB_dϕA
    dU_dϕB = @. πB + ϕB * dπB_dϕB + ϕA * dπA_dϕB

    return πA, πB, dπA_dϕA, dπA_dϕB, dπB_dϕA, dπB_dϕB, U, dU_dϕA, dU_dϕB
end


end
