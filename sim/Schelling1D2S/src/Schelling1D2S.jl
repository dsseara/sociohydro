"""
Module for 2-species, 1-dimensional Schelling model
with bounded neighborhoods
"""

module Schelling1D2S

using ProgressMeter
using HDF5
using Printf
using DSP
using StatsBase
using LinearAlgebra
using JSON

export random_state, move!, gain, glauber_prob, utility, run_simulation!, run_simulation, save, select_particle, pick_sites, run_sweep!


function random_state(params::Dict{String, Any})
    # unpack parameters
    Nx = params["grid_size"]
    capacity = params["capacity"]
    fillA, fillB = params["fill"]
    
    total_occupants_A = Int(floor(Nx * capacity * fillA))
    total_occupants_B = Int(floor(Nx * capacity * fillB))
    locations_A = rand(1:Nx, total_occupants_A)
    locations_B = rand(1:Nx, total_occupants_B)
    
    state_A = [count(==(i), locations_A) for i in 1:Nx]
    state_B = [count(==(i), locations_B) for i in 1:Nx]
    
    state = cat(state_A, state_B, dims=2)
    return state
end


"""
    run_simulation!(state, params)

Run a simulation of the 1D, 2 species Shelling model
Based on Grauwin et al, PNAS 2009

## Inputs
- state : Array
    - Nx2 array of initial state
- params : Dict
    - params["n_sweeps"] : int
        - Number of sweeps. 1 sweep is equal to sum(state) moves
    - params["density_preferred"]: float
        - preferred density, used to calculate the utility of the agents
    - params["capacity"] : int
        - maximum number of occupants allowed at each lattice site
    - params["m"] : float
        - utility of the rho = 1 state. m in [0, 1]
    - params["temperature"] : float
        - temperature used in the Glauber update rule
    - params["alpha"] : float in [0, 1]
        - float governing the weight of individual vs system fitness for updating probabilities.
        - alpha = 0 only considers individual fitness.
          alpha = 1 only considers system fitness
    - params["delta"] : float in [-1, 1]
        - float governing the extent to which each species influences each other
    - params["moveType"] : string
        - either "random" or "local", sets whether moving can
          happen anywhere ("random") or only to nearest neighbors ("local")
    - params["fill"] : 2-element array
        - what fraction of available sites are filled with agents of each type.
          Must sum to less than 1

## Outputs
- state : array
    - NxMx2 array of integers that gives state of
      the system after evolving for n_sweeps
"""
function run_simulation!(state::Matrix{Int64},
                         params::Dict{String, Any})
    # sim params
    dt = params["dt"]
    n_sweeps = params["n_sweeps"]
    capacity = params["capacity"]
    Nx = params["grid_size"]
    fill = params["fill"]
    α = params["alpha"]
    δ = params["delta"]
    κ = params["kappa"]
    temp = params["temperature"]
    # storage params
    snapshot = params["snapshot"]
    savepath = params["savepath"]

    # create savepath
    if isdir(savepath)
        for file in readdir(savepath, join=true)
            rm(file)
        end
    else
        mkpath(savepath)
    end

    # save params and initial state
    open(savepath * "/params.json", "w") do f
        JSON.print(f, params, 4)
    end
    save(state, params, sweep=0)

    # calculate how many steps per sweep
    total_occupants = sum(state)

    # main loop
    @showprogress 1 for ii in 1:n_sweeps
        run_sweep!(state,
                   total_occupants,
                   dt, capacity,
                   Nx, fill,
                   α, δ,
                   κ, temp)
        
        # snapshot = -1 means only
        # save initial and final
        if snapshot > 0
            if ii % snapshot == 0
                save(state, params, sweep=ii)
            end
        end
    end
    
    # save final state.
    # At worst, overwrites itself
    save(state, params, sweep=n_sweeps)

    return state
end


function run_sweep!(state::Matrix{Int64},
                    n_steps::Int64, dt::Float64,
                    capacity::Int64, Nx::Int64,
                    fill::Vector{Float64},
                    alpha::Float64, delta::Float64,
                    kappa::Float64, temp::Float64)
    rs = rand(n_steps)
    for jj in 1:n_steps
        move!(state, dt, capacity, Nx, fill,
              alpha, delta, kappa, temp,
              rs[jj])
    end
    return state
end


function move!(state::Matrix{Int64},
               dt::Float64,
               capacity::Int64,
               Nx::Int64,
               fill::Vector{Float64},
               alpha::Float64,
               delta::Float64,
               kappa::Float64,
               temp::Float64,
               r::Float64)
    # select particle to move and where to move to
    from, tos, weights = pick_sites(state, capacity, Nx, fill)

    # find gains of each move
    Gs = gain(state, from, tos, alpha, delta, kappa, capacity, Nx)
    
    # calculate probability of moving using Glauber-like rule
    probs = glauber_prob(Gs, temp, dt)

    # probs = [w * g for (w, g) in zip(weights, gprobs)]
    # probs ./= sum(probs)

    # draw a random number to select whether to move or not
    i = 1
    p = ps[i]
    while r > p
        i += 1
        p += ps[i]
    end

    if i <= length(tos)
        state[from] -= 1
        state[tos[i]] += 1
    end

    # if r < prob_to_move
    #     state[from] -= 1
    #     state[to] += 1
    # end

end


"""
    pick_sites(state, capacity, Nx, fill)

Function to find where to move a particle
from and to
"""
function pick_sites(state::Matrix{Int64},
                    capacity::Int64,
                    Nx::Int64,
                    fill::Vector{Float64})
    
    # select species weighted by their fill fractions
    species = sample(Weights(fill))
    # select a site to move from weighted by the number of particles in that site
    site = sample(Weights(state[:, species]))
    
    # find neighboring sites
    neighbors = [ifelse(site == Nx, 1, site + 1),  # right
                 ifelse(site == 1, Nx, site - 1)]  # left
    tos = [CartesianIndex(n, species) for n in neighbors]
    # weight neighbors by number of vacancies
    weights = [1 - (sum(state[n, :]) / capacity) for n in neighbors]
    
    # if sum(weights) == 0
    #     # don't move
    #     new_site = site
    # else
    #     new_site = neighbors[sample(Weights(weights))]
    # end
    
    from = CartesianIndex(site, species)
    # to = CartesianIndex(new_site, species)
    
    
    return from, tos, weights
end


"""
    gain(from, to, mover, params)

Calculate gain in utility

## Inputs
- from : 2-element array
    - number of particles at origin site of both species
- to : 2-element array
    - number of particles at new site of both species
- mover : Int
    - Which species is moving, species A or B, denoted by
      mover = 1 or mover = 2, respectively
- params : Dict
    - dictionary of parameters
"""
function gain(state::Matrix{Int64},
              from::CartesianIndex{2},
              tos::Array{CartesianIndex{2}, 1},
              alpha::Float64,
              delta::Float64,
              kappa::Float64,
              capacity::Int64,
              Nx::Int64) 

    # state_new = copy(state_old)
    # state_new[from] -= 1
    # state_new[to] += 1
    ∆N = [0, 0]
    ∆N[from[2]] = 1

    # preallocate gains for each choice of
    # where to move to
    Gs = Array{Float64}(undef, length(tos))

    # calcualte utilities of origin location before and after move
    # common to all options of where to move to
    utility_from_old = utility(state[from[1], :], capacity, delta, kappa)
    utility_from_new = utility((state[from[1], :] - ∆N), capacity, delta, kappa)

    for (toidx, to) in enumerate(tos)
        utility_to_old   = utility(state[to[1], :],  capacity, delta, kappa)
        utility_to_new   = utility((state[to[1], :] + ∆N),  capacity, delta, kappa)

        # calculate change in utility only of mover
        selfish = utility_to_new[from[2]] - utility_from_old[from[2]]

        # calculate change in utility of everyone
        altruistic = sum(+ (state[to[1], :] + ∆N)   .* utility_to_new
                         - state[to[1], :]   .* utility_to_old
                         + (state[from[1], :] - ∆N) .* utility_from_new
                         - state[from[1], :] .* utility_from_old)

        # calculate change in gradient penalizing term, n ∇^2 ϕ
        # got this from using mathematica to find expression for
        # difference of finite differences
        direction = Int(to[1] - from[1] - round((to[1] - from[1]) / Nx) * Nx) # use pbc
        lap_diff = (-state[mod1(from[1] - direction, Nx), from[2]]
                    + 3 * state[from] - 3 * state[to]
                    + state[mod1(to[1] + direction, Nx), to[2]] - 3) / capacity

        Gs[toidx] =alpha * altruistic + (1 - alpha) * selfish + lap_diff
    end

    return Gs
end


"""
    glauber_prob(ΔGs, params)

Calculate update probability using glauber-like dynamics.
Note that we are doing gain maximization,
which introduces an extra minus sign compared to
energy minimization

## Inputs
- deltaG : Float
    - change in whatever gain function is used
- params : Dict
    - params["temperature"] : Float
        - Temperature (quantifies error) 

## Outputs
- p : Float
    - probability of accepting move. 0 < p < 1
"""
function glauber_prob(ΔGs::Array{T, 1},
                      temp::T, dt::T) where T<:AbstractFloat

    # get probability of each move, plus not making a move
    ps = Array{Float64}(undef, length(ΔGs) + 1)

    for (gidx, ΔG) in enumerate(ΔGs)
        ps[gidx] = ifelse(temp == 0,
                          (sign(ΔG) + 1) * 0.5,
                          2 * temp * dt / (1 + exp(-ΔG / temp)))
    end

    # add probability of not moving
    ps[end] = 1 - sum(ps[1:end-1])

    return ps
end


"""
    asymmetric_utility(x, params)

Calculate the asymmetric utility function for an array

## Inputs
- x : 2-element array
    - fill fraction of each species.
      each element of x in [0, 1]
- params : Dict
    - preferred_density : Float
        - value of density that maximizes the utility.
          params["preferred_density"] in [0, 1]
    - m : Float
        - how much you prefer x = 1. params["m"] in [0, 1]
    - delta : Float
        - how much species affect each other. Between [-1, 1]

## Outputs
- utility : 2-element array
    - utility of each species, between [0, 1]
"""
function asymmetric_utility(x::Vector{T} where {T<:Real},
                            params::Dict{String, Any})
    # unpack parameters
    x0 = params["preferred_density"]
    m = params["m"]
    delta = params["delta"]
    
    xA, xB = x
    
    if xA < x0
        uA = 2 * xA
    else
        uA = m +  2 * (1 - m) * (1 - xA)
    end
    
    if xB < x0
        uB = 2 * xB
    else
        uB = m +  2 * (1 - m) * (1 - xB)
    end
    
    utility = [uA + delta * uB,
               uB - delta * uA]

    return utility
end


"""
    quadratic_utility(x, params)

Calculate the quadratic utility function for an array

## Inputs
- x : 2-element array
    - fill fraction of each species.
      each element of x in [0, 1]
- params : Dict
    - delta : Float
        - how much species affect each other. Between [-1, 1]

## Outputs
- utility : 2-element array
    - utility of each species, between [0, 1]
"""
function utility(n::Vector{Int64},
                 capacity::Int64,
                 delta::T,
                 kappa::T) where T<:AbstractFloat
    πs = Vector{Float64}(undef, 2)
    
    nA, nB = n
    ϕA = nA / capacity
    ϕB = nB / capacity
    
    # don't count self as a neighbor
    # uA = 4 * (ϕA - capacity^(-1)) * (1 - (ϕA - capacity^(-1)))
    uA = (ϕA - capacity^(-1)) - 1
    # uB = 4 * (ϕB - capacity^(-1)) * (1 - (ϕB - capacity^(-1)))
    uA = (ϕB - capacity^(-1)) - 1
    
    πs = [uA + (kappa - delta) * ϕB / 2,
               uB + (kappa + delta) * ϕA / 2]

    return πs
end


# function linear_utility(n)


function save(state::Matrix{T},
              params::Dict{String, Any};
              sweep::T = -1) where T<:Integer
    savepath = params["savepath"]
    n_sweeps = params["n_sweeps"]
    npad = Int(ceil(log10(n_sweeps)) + 1)

    filename = "n" * lpad(string(sweep), npad, '0') * ".hdf5"
    
    if Sys.isunix()
        filesep = "/"
    else
        filesep = "\\"
    end
        
    filepath = savepath * filesep * filename
    
    h5open(filepath, "w") do fid
        d = create_group(fid, "data")
        d["state"] = state
        d["sweep"] = sweep
    end
end


# # use this version if you want to compare two snapshots only
# function save(state_init::Matrix{T},
#               state::Matrix{T},
#               params::Dict{String, Any};
#               sweep::T = -1) where {T<:Integer}
    
#     savepath = params["savepath"]
#     mkpath(savepath)
    
#     if sweep < 0
#         filename = "t0_tFinal.hdf5"
#     else
#         filename = "t0_t" * Printf.format(Printf.Format("%04d"), sweep) * ".hdf5"
#     end
    
#     if Sys.isunix()
#         filesep = "/"
#     else
#         filesep = "\\"
#     end
        
#     filepath = savepath * filesep * filename
    
#     h5open(savepath, "w") do fid
#         d = create_group(fid, "data")
#         p = create_group(fid, "params")
#         d["state"] = state
#         d["state_init"] = state_init
#         for (key, val) in params
#             if key == "utility_func"
#                 val = string(val)
#             end
#             p[key] = val
#         end
#     end
# end

# function save(state::Matrix{Int64},
#               params::Dict{String, Any},
#               savefolder::String,
#               filename::String)
    
#     mkpath(savefolder)
#     savepath = savefolder * "/" * filename
    
#     h5open(savepath, "w") do fid
#         d = create_group(fid, "data")
#         p = create_group(fid, "params")
#         d["state"] = state
#         for (key, val) in params
#             if key == "utility_func"
#                 val = string(val)
#             end
#             p[key] = val
#         end
#     end
# end

# function moving_average_periodic(arr::Vector{Int}, box_size::Int)
#     kernel = Windows.gaussian(box_size, 0.1) # create kernel
#     kernel /= sum(kernel)  # normalize kernel
#     half_box = Int(floor(box_size / 2))
#     arr_periodic = cat(arr[end-half_box+1:end], arr, arr[1:half_box], dims=1)
#     c = conv(arr_periodic, kernel) # apply convolution
#     return c[box_size:end-box_size+1]
# end

# function moving_average_periodic(arr::Matrix{Int}, box_size::Int; dims::Int)
#     m(x) = moving_average_periodic(x, box_size)
#     c = mapslices(m, arr, dims=dims)
#     return c
# end

# function local_average(arr::Vector{T}, site::T, box_size::T;
#                        dims::T=1, width::Float64 = 1.0) where T<:Integer
#     kernel = Windows.gaussian(box_size, width) # create kernel
#     kernel /= sum(kernel)  # normalize kernel
#     half_box = Int(floor(box_size / 2))
#     arr_periodic = cat(arr[end-half_box+1:end], arr, arr[1:half_box], dims=dims)
    
#     psite = half_box + site
    
#     c = sum(arr_periodic[psite - half_box:psite + half_box] .* kernel)

#     return c
# end

# function local_average(arr::Matrix{T}, site::T, box_size::T; dims::T=1) where T<:Integer
#     la(x) = local_average(x, site, box_size)
#     c = mapslices(la, arr, dims=dims)
#     return vec(c)
# end

end
