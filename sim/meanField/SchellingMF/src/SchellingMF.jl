module SchellingMF

using LinearAlgebra
using Distributions
using StatsBase
using Printf
using HDF5
using JSON
using ProgressMeter
# using Zygote

export run_simulation, update!, load_data, compute_force, grad, lap, gradlap

# fourth order finite differences
function grad(Nx::Int64, dx::Float64)
    diagm(-(Nx - 1) => [+2/3],
          -(Nx - 2) => -(1/12) * ones(2),
          -2 => (1/12) * ones(Nx - 2),
          -1 => -(2/3) * ones(Nx - 1),
          +1 => +(2/3) * ones(Nx - 1),
          +2 => -(1/12) * ones(Nx - 2),
          +(Nx - 2) => (1/12) * ones(2),
          +(Nx - 1) => [-(2/3)]) ./ dx
end


lap(Nx::Int64, dx::Float64) = diagm(-(Nx - 1) => [+4/3],
                                  -(Nx - 2) => -(1/12) * ones(2),
                                  -2 => -(1/12) * ones(Nx - 2),
                                  -1 => (4/3) * ones(Nx - 1),
                                  0 => -(5/2) * ones(Nx),
                                  +1 => (4/3) * ones(Nx - 1),
                                  +2 => -(1/12) * ones(Nx - 2),
                                  +(Nx - 2) => -(1/12) * ones(2),
                                  +(Nx - 1) => [(4/3)]) ./ dx^2


gradlap(Nx::Int64, dx::Float64) = diagm(-(Nx - 1) => [-13/8],
                                        -(Nx - 2) => +1 * ones(2),
                                        -(Nx - 3) => -(1/8) * ones(3),
                                        -3 => +(1/8) * ones(Nx - 3),
                                        -2 => -1 * ones(Nx - 2),
                                        -1 => +(13/8) * ones(Nx - 1),
                                        +1 => -(13/8) * ones(Nx - 1),
                                        +2 => +1 * ones(Nx - 2),
                                        +3 => -(1/8) * ones(Nx - 3),
                                        +(Nx - 3) => +(1/8) * ones(3),
                                        +(Nx - 2) => -1 * ones(2),
                                        +(Nx - 1) => [+13 / 8]) ./ dx^3


# f(x::AbstractFloat) = x - 1
# df(x::AbstractFloat) = 1


# function fitness(ϕA::Array{T, 1}, ϕB::Array{T, 1},
#                  δ::T, κ::T) where T<:AbstractFloat
#     # fitness
#     πA = @. ϕA + (κ - δ) * ϕB / 2
#     πB = @. ϕB * (1 - ϕB) + (κ +s δ) * ϕA / 2

#     # fitness derivatives
#     dπA_dϕA = @. 1
#     dπA_dϕB = @. (κ - δ) / 2
#     dπB_dϕA = @. (κ + δ) / 2
#     dπB_dϕB = @. 1 - 2 * ϕB

#     # global fitness
#     U = @. ϕA * πA + ϕB * πB

#     # global fitness derivatives
#     dU_dϕA = @. πA + ϕA * dπA_dϕA + ϕB * dπB_dϕA
#     dU_dϕB = @. πB + ϕB * dπB_dϕB + ϕA * dπA_dϕB

#     return πA, πB, dπA_dϕA, dπA_dϕB, dπB_dϕA, dπB_dϕB, U, dU_dϕA, dU_dϕB
# end


function mobility(x, y)
    return x * (1 - x - y)
end


# function totalFitness(ϕA::T, ϕB::T,
#                       πA::Function, πB::Function) where T<:AbstractFloat
#     return ϕA * πA(ϕA, ϕB) + ϕB * πB(ϕA, ϕB)
# end

# function ∂U(ϕA::T, ϕB::T, πA::Function, πB::Function) where T<:AbstractFloat
#     return gradient((ϕA, ϕB) -> totalFitness())
# end

# ∂U(ϕA, ϕB) = stack(gradient.((ϕA, ϕB)->totalFitness(ϕA, ϕB, πA, πB)))


function compute_force(ϕA::Array{T, 1}, ϕB::Array{T, 1},
                       dx::T, Nx::Int64,
                       πA::Function, πB::Function;
                       α::T = 0.0, temp::T = 0.1,
                       Γ::T = 0.5) where T<:AbstractFloat
    
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
    # ∇∂U_∂ϕA = similar(ϕA)
    # ∇∂U_∂ϕB = similar(ϕB)
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
    # ∂πA = gradient.(πA, ϕA, ϕB)
    # ∂πA_∂ϕA = [∂πA[i][1] for i in 1:length(ϕA)]
    # ∂πA_∂ϕB = [∂πA[i][2] for i in 1:length(ϕB)]

    # ∂πB = gradient.(πB, ϕA, ϕB)
    # ∂πB_∂ϕA = [∂πB[i][1] for i in 1:length(ϕA)]
    # ∂πB_∂ϕB = [∂πB[i][2] for i in 1:length(ϕB)]

    # U(ϕA, ϕB) = totalFitness(ϕA, ϕB, πA, πB)
    # ∂U = gradient.((ϕA, ϕB) -> totalFitness(ϕA, ϕB, πA, πB), ϕA, ϕB)
    # ∂U_∂ϕA = [∂U[i][1] for i in 1:length(ϕA)]
    # ∂U_∂ϕB = [∂U[i][2] for i in 1:length(ϕB)]

    # πA, πB, dπA_dϕA, dπA_dϕB, dπB_dϕA, dπB_dϕB, U, dU_dϕA, dU_dϕB = fitness(ϕA, ϕB, δ, κ)
    mul!(∇πA, ∇, πA.(ϕA, ϕB))
    mul!(∇πB, ∇, πB.(ϕA, ϕB))
    # mul!(∇∂U_∂ϕA, ∇, ∂U_∂ϕA)
    # mul!(∇∂U_∂ϕB, ∇, ∂U_∂ϕB)
    # mul!(∇fitnessA, ∇, mA .* ((1 - α) .* ∇πA .+ α .* ∇∂U_∂ϕA))
    mul!(∇fitnessA, ∇, mA .* ∇πA)
    # mul!(∇fitnessB, ∇, mB .* ((1 - α) .* ∇πB .+ α .* ∇∂U_∂ϕB))
    mul!(∇fitnessB, ∇, mB .* ∇πB)

    # forces
    FA = @. T∇²ϕA - TϕB∇²ϕA + TϕA∇²ϕB - ∇fitnessA - ∇mAΓ∇³ϕA
    FB = @. T∇²ϕB - TϕA∇²ϕB + TϕB∇²ϕA - ∇fitnessB - ∇mBΓ∇³ϕB

    return FA, FB
end


function update!(ϕA::Array{T, 1}, ϕB::Array{T, 1},
                 dx::T, Nx::Int64, dt::T,
                 πA::Function, πB::Function;
                 α::T = 0.0, temp::T = 0.1,
                 Γ::T = 1.0) where T<:AbstractFloat
    # get forces
    FA, FB = compute_force(ϕA, ϕB, dx, Nx,
                           πA, πB,
                           α=α,# δ=δ, κ=κ,
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
                        πA::Function, πB::Function,
                        snapshot::Int64,
                        savepath::String,
                        filename::String;
                        ϕA0::T = 0.25, ϕB0::T = 0.25,
                        δϕA0::T = 0.05, δϕB0::T = 0.05,
                        α::T = 0.0, temp::T = 0.1,
                        Γ::T =1.0) where T<:AbstractFloat

    ϕA, ϕB = random_state(Nx, ϕA0, ϕB0, δϕA0, δϕB0)
    t_init = 0.0

    ϕA, ϕB, t = run_simulation(ϕA, ϕB, t_init,
                               dx, Nx, dt, Nt,
                               πA, πB,
                               snapshot,savepath, filename,
                               ϕA0=ϕA0, ϕB0=ϕB0,
                               δϕA0=δϕA0, δϕB0=δϕB0,
                               α=α, temp=temp, Γ=Γ)
    return ϕA, ϕB, t
end

# this version starts with a given initial condition
function run_simulation(ϕA::Array{T, 1}, ϕB::Array{T, 1}, t_init::T,
                        dx::T, Nx::Int64, dt::T, Nt::Int64,
                        πA::Function, πB::Function,
                        snapshot::Int64, savepath::String,
                        filename::String;
                        ϕA0::T = 0.25, ϕB0::T = 0.25,
                        δϕA0::T = 0.05, δϕB0::T = 0.05,
                        α::T = 0.0, temp::T = 0.1,
                        Γ::T =1.0) where T<:AbstractFloat


    if dt * dx^(-4) > 1/8
        @warn "von Neumann stability criterion not satisfied"
    end

    # get spatial variable
    L = Nx * dx
    x = collect(-L/2 + dx/2:dx:(L/2) - dx/2)

    # define gradient operators
    global ∇ = grad(Nx, dx)
    global ∇² = lap(Nx, dx)
    global ∇³ = gradlap(Nx, dx)d

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
                  "filename" => filename,
                  "α" => α,
                  # "δ" => δ,
                  # "κ" => κ,
                  "temp" => temp,
                  "Γ" => Γ)

    param_filename = filename * "_params.json"
    open(savepath * "/" * param_filename, "w") do f
        JSON.print(f, params, 4)
    end

    t::Float64 = t_init
    # save initial condition
    npad = Int(ceil(log10(Nt)) + 1)
    save(ϕA, ϕB, x, t, 0, npad, savepath, filename)

    # main loop
    println("Starting main loop...")
    @showprogress for ii in 1:Nt
        ϕA, ϕB = update!(ϕA, ϕB, dx, Nx, dt,
                         πA, πB,
                         α=α,# δ=δ, κ=κ,
                         temp=temp, Γ=Γ)
        t += dt

        if ii % snapshot == 0
            if nan_check(ϕA)
                save(ϕA, ϕB, x, t, ii, npad, savepath, filename)
            else
                throw(OverflowError("got nans"))
            end
        end
    end
    return ϕA, ϕB, t
end

function save(ϕA::Array{T, 1}, ϕB::Array{T, 1}, x::Array{T, 1}, t::T,
              step::Int64, npad::Int64, savepath::String,
              filename::String) where T<:AbstractFloat

    groupname = "n" * lpad(string(step), npad, '0')

    if Sys.isunix()
        filesep = "/"
    else
        filesep = "\\"
    end

    filepath = savepath * filesep * filename * ".hdf5"

    h5open(filepath, "cw") do fid
        d = create_group(fid, groupname)
        d["phiA"] = ϕA
        d["phiB"] = ϕB
        d["x"] = x
        d["t"] = t
    end
end

searchdir(path, key) = filter(x->contains(x, key), readdir(path, join=true))

function load_data(savepath::String)
    datafile = searchdir(savepath, "hdf5")[1]
    paramfile = searchdir(savepath, "json")[1]
    params = JSON.parsefile(paramfile)
    N_saved = Int(params["Nt"] / params["snapshot"]) + 1

    ϕA_array = Array{Float64}(undef, params["Nx"], N_saved)
    ϕB_array = Array{Float64}(undef, params["Nx"], N_saved)
    x_array = Array{Float64}(undef, params["Nx"])
    t_array = Array{Float64}(undef, N_saved)

    h5open(datafile, "r") do d
        for (kidx, key) in enumerate(keys(d))
            # println(keys(d[key]))
            ϕA_array[:, kidx] = read(d[key], "phiA")
            ϕB_array[:, kidx] = read(d[key], "phiB")
            t_array[kidx] = read(d[key], "t")
            if kidx == 1
                x_array[:] = read(d[key], "x")
            end
        end
    end
    return ϕA_array, ϕB_array, x_array ,t_array, params
end

function nan_check(x::Array{T, 1}) where T<:AbstractFloat
    return all(isfinite, x)
end

end
