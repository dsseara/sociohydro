"""
Module for 2-species, 2-dimensional Schelling model.
Inspired by Grauwin et al, PNAS 2009
"""

module Schelling2D2S

using ProgressMeter
using HDF5
using Printf

export random_state, move!, gain, glauber_prob, asymmetric_utility, quadratic_utility, run_simulation!, run_simulation, save

"""
    run_simulation!(state, params)

Run a simulation of the 2 species Shelling model
Based on Grauwin et al, PNAS 2009

## Inputs
- state : Array
    - NxMx2 array of initial state
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
function run_simulation!(state::Array{Int64, 3},
                         params::Dict{String, Any})
    n_sweeps = params["n_sweeps"]
    N = params["grid_size"]
    snapshot = params["snapshot"]
    savepath = params["savepath"]
    
    # make sure we have a place to save output
    mkpath(savepath)
    
    # how many steps per sweep
    total_occupants = sum(state)
    
    # save initial state
    save(state, params, sweep=0)
    
    # main loop
    @showprogress 1 for ii in 1:n_sweeps
        for jj in 1:total_occupants
            move!(state, params)
        end
        
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


function run_simulation(state::Array{Int64, 3},
                        params::Dict{String, Any})
    n_sweeps = params["n_sweeps"]
    N = params["grid_size"]
    snapshot = params["snapshot"]
    savepath = params["savepath"]
    
    # make sure we have a place to save output
    mkpath(savepath)
    
    # how many steps per sweep
    total_occupants = sum(state)
    
    # save initial state
    save(state, params, sweep=0)
    
    # allocate place to put new state
    state_new = deepcopy(state)
    
    # main loop
    @showprogress 1 for ii in 1:n_sweeps
        for jj in 1:total_occupants
            move!(state_new, params)
        end
        
        # snapshot = -1 means save
        # initial and final states only
        if snapshot > 0
            if ii % snapshot == 0
                save(state_new, params, sweep=ii)
            end
        end
    end
    
    # save final state.
    # At worst, overwrites itself
    save(state_new, params, sweep=n_sweeps)
    
    return state_new
end


function random_state(params::Dict{String, Any})
    # unpack parameters
    gs = params["grid_size"]
    capacity = params["capacity"]
    fillA, fillB = params["fill"]
    
    total_occupants_A = Int(floor(gs^2 * capacity * fillA))
    total_occupants_B = Int(floor(gs^2 * capacity * fillB))
    locations_A = rand(1:gs^2, total_occupants_A)
    locations_B = rand(1:gs^2, total_occupants_B)
    
    state_A = reshape([count(==(i), locations_A) for i in 1:gs^2], (gs, gs))
    state_B = reshape([count(==(i), locations_B) for i in 1:gs^2], (gs, gs))
    
    state = cat(state_A, state_B, dims=3)
    return state
end


function move!(state::Array{Int64, 3},
               params::Dict{String, Any})
    # unpack parameters
    moveType = params["moveType"]
    capacity = params["capacity"]
    N = params["grid_size"]
    
    # randomly select species
    species = rand(1:2)
    
    # randomly select where to move from,
    # proportionally to number of individuals there
    
    # finds all non-empty 
    nzidxs = findall(>(0), state[:, :, species])
    # use numbers at each site to build
    # a cumulative distribution function
    vals = cumsum([state[idx, species] for idx in nzidxs])
    # take first index above a random number to make weighted
    # selection
    selection = findfirst((vals / maximum(vals)) .> rand())
    from = [nzidxs[selection][1], nzidxs[selection][2]]
    
    if moveType == "random"
        # pick a random place
        to = rand(1:N, 2)

        # make sure destination is not the same place
        # you want to move from, nor is it full 
        while to == from || sum(state[to[1], to[2], :]) >= capacity
            to = rand(1:N, 2)
        end

    elseif moveType == "local"
        # get neighbors including periodic boundaries
        neighbors = [mod1.(from + [+1, 0], N),  # north
                     mod1.(from + [-1, 0], N),  # south
                     mod1.(from + [0, +1], N),  # east
                     mod1.(from + [0, -1], N)]  # west
        
        # only move to places filled less than capacity
        avail = [sum(state[n[1], n[2], :]) < capacity for n in neighbors]
        
        # if all neighbors are full, pick a new "from" location
        while sum(avail) == 0
            selection = findfirst((vals / maximum(vals)) .> rand())
            from = [nzidxs[selection][1], nzidxs[selection][2]]

            neighbors = [mod1.(from + [+1, 0], N),  # north
                         mod1.(from + [-1, 0], N),  # south
                         mod1.(from + [0, +1], N),  # east
                         mod1.(from + [0, -1], N)]  # west
            avail = [sum(state[n[1], n[2], :]) < capacity for n in neighbors]
        end
        to = rand(neighbors[avail])
    else
        error("moveType = random or local")
    end

    # find gain
    G = gain(state[from[1], from[2], :], state[to[1], to[2], :], species, params)

    # calculate probability of moving using Glauber-like rule
    prob_to_move = glauber_prob(G, params)

    # draw a random number to select whether to move or not
    if rand() < prob_to_move
        state[from[1], from[2], species] -= 1
        state[to[1], to[2], species] += 1
    end

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
function gain(from::Vector{Int64},
              to::Vector{Int64},
              mover::Int64,  
              params::Dict{String, Any}) 
    # unpack parameters
    alpha = params["alpha"]
    capacity = params["capacity"]
    utility_func = params["utility_func"]
    delta = params["delta"]

    # calculate new and old densities of origin and destination
    # of the species that is moving
    from_old = from ./ capacity
    to_old = to ./ capacity

    from_new = deepcopy(from_old)
    from_new[mover] -= 1 / capacity
    to_new = deepcopy(to_old)
    to_new[mover] += 1 / capacity
    
    utility_from_old = utility_func(from_old, params)
    utility_from_new = utility_func(from_new, params)
    utility_to_old = utility_func(to_old, params)
    utility_to_new = utility_func(to_new, params)
    
    # calculate change in utility only of mover
    selfish = utility_to_new[mover] - utility_from_old[mover]
    
    # calculate change in utility of everyone
    altruistic = sum(    to_new .* utility_to_new
                     -   to_old .* utility_to_old
                     + from_new .* utility_from_new
                     - from_old .* utility_from_old)

    return alpha * capacity * altruistic + (1 - alpha) * selfish
end


"""
    glauber_prob(deltaG, params)

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
function glauber_prob(deltaG::Float64, params::Dict{String, Any})
    # unpack parameters
    T = params["temperature"]
    
    if T == 0
        # p = 1.0 for ∆G > 0
        # p = 0.5 for ∆G = 0
        # p = 0.0 for ∆G < 0
        p = (sign(deltaG) + 1) * 0.5
    else
        p = 1 / (1 + exp(-deltaG / T))
    end
    
    return p
end


"""
    asymmetric_utility(x, params)

Calculate the asymmetric utility function for an array

## Inputs
- x : 2-element array
    - density of each species. each element of x in [0, 1]
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
    
    utility = [(uA + delta * uB) / (1 + delta),
               (uB - delta * uA + delta) / (1 + delta)]

    return utility
end


"""
    quadratic_utility(x, params)

Calculate the quadratic utility function for an array

## Inputs
- x : 2-element array
    - density of each species. each element of x in [0, 1]
- params : Dict
    - delta : Float
        - how much species affect each other. Between [-1, 1]

## Outputs
- utility : 2-element array
    - utility of each species, between [0, 1]
"""
function quadratic_utility(x::Vector{T} where {T<:Real},
                           params::Dict{String, Any})
    delta = params["delta"]
    
    xA, xB = x
    
    uA = 4 * xA * (1 - xA)
    uB = 4 * xB * (1 - xB)
    
    utility = [(uA + delta * uB) / (1 + delta),
               (uB - delta * uA + delta) / (1 + delta)]
    
    return utility
end


function save(state::Array{T, 3},
              params::Dict{String, Any};
              sweep::T = -1) where T<:Integer
    savepath = params["savepath"]
    mkpath(savepath)
    
    if sweep < 0
        filename = "tFinal.hdf5"
    else
        filename = "t" * Printf.format(Printf.Format("%04d"), sweep) * ".hdf5"
    end
    
    if Sys.isunix()
        filesep = "/"
    else
        filesep = "\\"
    end
        
    filepath = savepath * filesep * filename
    
    h5open(filepath, "w") do fid
        d = create_group(fid, "data")
        p = create_group(fid, "params")
        d["state"] = state
        for (key, val) in params
            if key == "utility_func"
                val = string(val)
            end
            p[key] = val
        end
    end
end


# use this version if you want to compare two snapshots only
function save(state_init::Array{T, 3},
              state::Array{T},
              params::Dict{String, Any};
              sweep::T = -1) where {T<:Integer}
    
    savepath = params["savepath"]
    mkpath(savepath)
    
    if sweep < 0
        filename = "t0_tFinal.hdf5"
    else
        filename = "t0_t" * Printf.format(Printf.Format("%04d"), sweep) * ".hdf5"
    end
    
    if Sys.isunix()
        filesep = "/"
    else
        filesep = "\\"
    end
        
    filepath = savepath * filesep * filename
    
    h5open(savepath, "w") do fid
        d = create_group(fid, "data")
        p = create_group(fid, "params")
        d["state"] = state
        d["state_init"] = state_init
        for (key, val) in params
            if key == "utility_func"
                val = string(val)
            end
            p[key] = val
        end
    end
end

# function periodic_index(idx::CartesianArray{2}, N::Int64)
#     idx_new = similar(idx)
#     idx[findall(idx .> [N, N])] .= 1
#     idx[findall(idx .< [1, 1])] .= N

#     return CartesianIndex((idx[1], idx[2]))
# end

end
