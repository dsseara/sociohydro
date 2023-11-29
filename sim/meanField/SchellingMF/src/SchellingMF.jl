module SchellingMF

using LinearAlgebra
using Distributions
using StatsBase
using Printf
using HDF5
using JSON
using ProgressMeter

export run_simulation, update!

# # 2nd order finite differences
# grad(Nx::Int64, dx::Float64) = diagm(-(Nx - 1) => [+1/2],
#                                   -1 => -(1/2) * ones(Nx - 1),
#                                   +1 => +(1/2) * ones(Nx - 1),
#                                   +(Nx - 1) => [-(1/2)]) ./ dx


# lap(Nx::Int64, dx::Float64) = diagm(-(Nx - 1) => [1],
#                                   -1 => 1 * ones(Nx - 1),
#                                   0 => -2 * ones(Nx),
#                                   +1 => 1 * ones(Nx - 1),
#                                   +(Nx - 1) => [1]) ./ dx^2


# gradlap(Nx::Int64, dx::Float64) = diagm(-(Nx - 1) => [-13/8],
#                                         -(Nx - 2) => +1 * ones(2),
#                                         -(Nx - 3) => -(1/8) * ones(3),
#                                         -3 => +(1/8) * ones(Nx - 3),
#                                         -2 => -1 * ones(Nx - 2),
#                                         -1 => +(13/8) * ones(Nx - 1),
#                                         +1 => -(13/8) * ones(Nx - 1),
#                                         +2 => +1 * ones(Nx - 2),
#                                         +3 => -(1/8) * ones(Nx - 3),
#                                         +(Nx - 3) => +(1/8) * ones(3),
#                                         +(Nx - 2) => -1 * ones(2),
#                                         +(Nx - 1) => [+13 / 8]) ./ dx^3


f(x::AbstractFloat) = x - 1
df(x::AbstractFloat) = 1

function calc_grad(field::Array{T, 2}, dx::T, periodic::Bool) where T<:AbstractFloat
    ∇xfield = similar(field)
    ∇yfield = similar(field)
    Ny, Nx = size(field)

    # central differences in center region
    for x in 2:Nx-1
        for y in 2:Ny-1
            ∇xfield[y, x] = (field[y, x+1] - field[y, x-1]) / 2
            ∇yfield[y, x] = (field[y+1, x] - field[y-1, x]) / 2
        end
    end

    if periodic # do central differences at edges
        ### corners ###
        # bottom left
        ∇xfield[1, 1] = (field[1, 2] - field[1, Nx]) / 2
        ∇yfield[1, 1] = (field[2, 1] - field[Ny, 1]) / 2

        # bottom right
        ∇xfield[1, Nx] = (field[1, 1] - field[1, Nx-1]) / 2
        ∇yfield[1, Nx] = (field[2, Nx] - field[Ny, Nx]) / 2

        # top left
        ∇xfield[Ny, 1] = (field[Ny, 2] - field[Ny, Nx]) / 2
        ∇yfield[Ny, 1] = (field[1, 1] - field[Ny-1, 1]) / 2

        # top right
        ∇xfield[Ny, Nx] = (field[Ny, 1] - field[Ny, Nx-1]) / 2
        ∇yfield[Ny, Nx] = (field[1, Nx] - field[Ny-1, Nx]) / 2

        ### edges ###
        # vertical
        for y in 2:Ny-1
            # left
            ∇xfield[y, 1] = (field[y, 2] - field[y, Nx]) / 2
            ∇yfield[y, 1] = (field[y+1, 1] - field[y-1, 1]) / 2
            # right
            ∇xfield[y, Nx] = (field[y, 1] - field[y, Nx-1]) / 2
            ∇yfield[y, Nx] = (field[y+1, Nx] - field[y-1, Nx]) / 2
        end
        # horizontal
        for x in 2:Nx-1
            # bottom
            ∇xfield[1, x] = (field[1, x+1] - field[1, x-1]) / 2
            ∇yfield[1, x] = (field[2, x] - field[Ny, x]) / 2
            # top
            ∇xfield[Ny, x] = (field[Ny, x+1] - field[Ny, x-1]) / 2
            ∇yfield[Ny, x] = (field[1, x] - field[Ny-1, x]) / 2
        end
    else # do forward/backward differences at ends
        ### corners ###
        # bottom left
        ∇xfield[1, 1] = field[1, 2] - field[1, 1]
        ∇yfield[1, 1] = field[2, 1] - field[1, 1]

        # bottom right
        ∇xfield[1, Nx] = field[1, Nx] - field[1, Nx-1]
        ∇yfield[1, Nx] = field[2, Nx] - field[1, Nx]

        # top left
        ∇xfield[Ny, 1] = field[Ny, 2] - field[Ny, 1]
        ∇yfield[Ny, 1] = field[Ny, 1] - field[Ny-1, 1]

        # top right
        ∇xfield[Ny, Nx] = field[Ny, Nx] - field[Ny, Nx-1]
        ∇yfield[Ny, Nx] = field[Ny, Nx] - field[Ny-1, Nx]

        ### edges ###
        # vertical
        for y in 2:Ny-1
            # left
            ∇xfield[y, 1] = field[y, 2] - field[y, 1]
            ∇yfield[y, 1] = (field[y+1, 1] - field[y-1, 1]) / 2
            # right
            ∇xfield[y, Nx] = field[y, Nx] - field[y, Nx - 1]
            ∇yfield[y, Nx] = (field[y+1, Nx] - field[y-1, Nx]) / 2
        end
        #horizontal
        for x in 2:Nx-1
            # bottom
            ∇xfield[1, x] = (field[1, x+1] - field[1, x-1]) / 2
            ∇yfield[1, x] = field[2, x] - field[1, x]  # forward
            # top
            ∇xfield[Ny, x] = (field[Ny, x+1] - field[Ny, x-1]) / 2
            ∇yfield[Ny, x] = field[Ny, x] - field[Ny-1, x]  # backward
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


function calc_grad3(field::Array{T, 2}, dx::T, periodic::Bool) where T<:AbstractFloat
    ∇²field = calc_lap(field, dx, periodic)
    ∇³xfield, ∇³yfield = calc_grad(∇²field, dx, periodic)
    return ∇³xfield, ∇³yfield
end


function calc_div(fieldx::Array{T, 2}, fieldy::Array{T, 2}, dx::T, periodic::Bool) where T<:AbstractFloat
    ∇xfieldx, ∇yfieldx = calc_grad(fieldx, dx, periodic)
    ∇xfieldy, ∇yfieldy = calc_grad(fieldy, dx, periodic)
    return ∇xfieldx + ∇yfieldy
end


function calc_J(ϕA::Array{T, 2}, ϕB::Array{T, 2}, dx::T, periodic::Bool,
                utility_func::Function, D::T, Γ::T) where T<:AbstractFloat
    # preallocation
    JAx = similar(ϕA)
    JAy = similar(ϕA)
    JBx = similar(ϕB)
    JBy = similar(ϕB)

    πA, πB = utility_func(ϕA, ϕB)
    ∇xπA, ∇yπA = calc_grad(uA, dx, periodic)
    ∇xπB, ∇yπB = calc_grad(uB, dx, periodic)
    ∇xϕA, ∇yϕA = calc_grad(ϕA, dx, periodic)
    ∇xϕB, ∇yϕB = calc_grad(ϕB, dx, periodic)
    ∇³xϕA, ∇³yϕA = calc_grad3(ϕA, dx, periodic)
    ϕBxxx, ϕBxxy = calc_grad3(ϕB, dx, periodic)

    JAx = -D * ∇xϕA + ϕA * ∇xπA + Γ * ϕA * ∇³xϕA
    JAy = -D * ∇yϕA + ϕA * ∇yπA + Γ * ϕA * ∇³yϕA
    JBx = -D * ∇xϕB + ϕB * ∇xπB + Γ * ϕB * ∇³xϕB
    JBy = -D * ∇yϕB + ϕB * ∇yπB + Γ * ϕB * ∇³yϕB

    return JAx, JAy, JBx, JBy
end


function calc_force(ϕA::Array{T, 2}, ϕB::Array{T, 2}, dx::T, periodic::Bool,
                    utility_func::Function, D::T, Γ::T) where T<:AbstractFloat

    JAx, JAy, JBx, JBy = calc_J(ϕA, ϕB, dx, periodic, utility_func, D, Γ)
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


function mobility(x, y)
    return x * (1 - x - y)
end


function compute_force(ϕA::Array{T, 1}, ϕB::Array{T, 1},
                       dx::T, Nx::Int64;
                       α::T = 0.0, δ::T = 0.0, κ::T = 1.0,
                       temp::T = 0.1, Γ::T = 0.5) where T<:AbstractFloat
    
    ### preallocation ###
    mA = similar(ϕA)
    mB = similar(ϕB)

    T∇²ϕA = similar(ϕA)
    T∇²ϕB = similar(ϕB)

    TϕA∇²ϕB = similar(ϕA)
    TϕB∇²ϕA = similar(ϕB)

    Γ∇³ϕA = similar(ϕA)
    Γ∇³ϕB = similar(ϕB)
    ∇mAΓ∇³ϕA = similar(ϕA)
    ∇mBΓ∇³ϕB = similar(ϕB)

    ∇πA = similar(ϕA)
    ∇πB = similar(ϕB)
    ∇dU_dϕA = similar(ϕA)
    ∇dU_dϕB = similar(ϕB)
    ∇fitnessA = similar(ϕA)
    ∇fitnessB = similar(ϕB)

    FA = similar(ϕA)
    FB = similar(ϕB)
    #########
    
    # mobility
    mA = mobility.(ϕA, ϕB)
    mB = mobility.(ϕB, ϕA)

    # diffusion
    mul!(T∇²ϕA, ∇², ϕA, temp, 0)
    mul!(T∇²ϕB, ∇², ϕB, temp, 0)
    broadcast!(*, TϕA∇²ϕB, ϕA, T∇²ϕB)
    broadcast!(*, TϕB∇²ϕA, ϕB, T∇²ϕA)

    # bilaplacian
    mul!(Γ∇³ϕA, ∇³, ϕA, Γ, 0)
    mul!(Γ∇³ϕB, ∇³, ϕB, Γ, 0)
    mul!(∇mAΓ∇³ϕA, ∇, mA .* Γ∇³ϕA)
    mul!(∇mBΓ∇³ϕB, ∇, mB .* Γ∇³ϕB)

    # fitness dynamics
    πA, πB, dπA_dϕA, dπA_dϕB, dπB_dϕA, dπB_dϕB, U, dU_dϕA, dU_dϕB = fitness(ϕA, ϕB, δ, κ)
    mul!(∇πA, ∇, πA)
    mul!(∇πB, ∇, πB)
    mul!(∇dU_dϕA, ∇, dU_dϕA)
    mul!(∇dU_dϕB, ∇, dU_dϕB)
    mul!(∇fitnessA, ∇, mA .* ((1 - α) .* ∇πA .+ α .* ∇dU_dϕA))
    mul!(∇fitnessB, ∇, mB .* ((1 - α) .* ∇πB .+ α .* ∇dU_dϕB))

    # forces
    FA = @. T∇²ϕA - TϕB∇²ϕA + TϕA∇²ϕB - ∇fitnessA - ∇mAΓ∇³ϕA
    FB = @. T∇²ϕB - TϕA∇²ϕB + TϕB∇²ϕA - ∇fitnessB - ∇mBΓ∇³ϕB

    return FA, FB
end


function update!(ϕA::Array{T, 1}, ϕB::Array{T, 1},
                 dx::T, Nx::Int64, dt::T;
                 α::T = 0.0, δ::T = 0.0, κ::T = 1.0,
                 temp::T = 0.1, Γ::T = 1.0) where T<:AbstractFloat
    # get forces
    FA, FB = compute_force(ϕA, ϕB, dx, Nx,
                           α=α, δ=δ, κ=κ,
                           temp=temp, Γ=Γ)
    
    # update
    ϕA += dt * FA
    ϕB += dt * FB

    return ϕA, ϕB
end


function random_state(Nx::Int64,
                      ϕA0::T, ϕB0::T,
                      δϕA0::T, δϕB0::T) where T<:AbstractFloat
    ϕA = ϕA0 .+ rand(Uniform(-1, 1), Nx) .* δϕA0
    ϕB = ϕB0 .+ rand(Uniform(-1, 1), Nx) .* δϕB0

    return ϕA, ϕB
end


# This version starts with a random initial condition
function run_simulation(dx::T, Nx::Int64,
                        dt::T, Nt::Int64,
                        snapshot::Int64,
                        savepath::String;
                        ϕA0::T = 0.25, ϕB0::T = 0.25,
                        δϕA0::T = 0.05, δϕB0::T = 0.05,
                        α::T = 0.0, δ::T = 0.0, κ::T = 0.0,
                        temp::T = 0.1, Γ::T =1.0) where T<:AbstractFloat

    ϕA, ϕB = random_state(Nx, ϕA0, ϕB0, δϕA0, δϕB0)
    t_init = 0.0

    ϕA, ϕB, t = run_simulation(ϕA, ϕB,  t_init,
                               dx, Nx, dt, Nt,
                               snapshot,savepath,
                               ϕA0=ϕA0, ϕB0=ϕB0, δϕA0=δϕA0, δϕB0=δϕB0,
                               α=α, δ=δ, κ=κ, temp=temp, Γ=Γ)
    return ϕA, ϕB, t
end

# this version starts with a given initial condition
function run_simulation(ϕA::Array{T, 1}, ϕB::Array{T, 1}, t_init::T,
                        dx::T, Nx::Int64, dt::T, Nt::Int64,
                        snapshot::Int64, savepath::String;
                        ϕA0::T = 0.25, ϕB0::T = 0.25,
                        δϕA0::T = 0.05, δϕB0::T = 0.05,
                        α::T = 0.0, δ::T = 0.0, κ::T = 0.0,
                        temp::T = 0.1, Γ::T =1.0) where T<:AbstractFloat


    if dt * dx^(-4) > 1/8
        @warn "von Neumann stability criterion not satisfied"
    end

    # get spatial variable
    L = Nx * dx
    x = collect(-L/2 + dx/2:dx:(L/2) - dx/2)

    # define gradient operators
    global ∇ = grad(Nx, dx)
    global ∇² = lap(Nx, dx)
    global ∇³ = gradlap(Nx, dx)

    # ensure we have a clean directory to dump data into
    if isdir(savepath)
        for file in readdir(savepath, join=true)
            rm(file)
        end
    else
        mkpath(savepath)
    end

    params = Dict("ϕA0" => ϕA0,
                  "ϕB0" => ϕB0,
                  "dx" => dx,
                  "Nx" => Nx,
                  "dt" => dt,
                  "Nt" => Nt,
                  "snapshot" => snapshot,
                  "savepath" => savepath,
                  "α" => α,
                  "δ" => δ,
                  "κ" => κ,
                  "temp" => temp,
                  "Γ" => Γ)

    open(savepath * "/params.json", "w") do f
        JSON.print(f, params, 4)
    end

    t::Float64 = t_init
    # save initial condition
    npad = Int(ceil(log10(Nt)) + 1)
    save(ϕA, ϕB, x, t, 0, npad, savepath)

    # main loop
    println("Starting main loop...")
    @showprogress for ii in 1:Nt
        ϕA, ϕB = update!(ϕA, ϕB, dx, Nx, dt,
                         α=α, δ=δ, κ=κ, temp=temp, Γ=Γ)
        t += dt

        if ii % snapshot == 0
            if nan_check(ϕA)
                save(ϕA, ϕB, x, t, ii, npad, savepath)
            else
                throw(OverflowError("got nans"))
            end
        end
    end
    return ϕA, ϕB, t
end

function save(ϕA::Array{T, 1}, ϕB::Array{T, 1}, x::Array{T, 1}, t::T,
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
        d["t"] = t
    end
end

function nan_check(x::Array{T, 1}) where T<:AbstractFloat
    return all(isfinite, x)
end

end
