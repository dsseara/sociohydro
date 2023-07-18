"""
Module for 1-species, 1 dimensional Schelling model.
Inspired by Grauwin et al, PNAS 2009
"""

module Schelling
using ProgressMeter
using HDF5
using Printf
export random_state, move!, gain, glauber_prob, asymmetric_utility, quadratic_utility, run_simulation!, run_simulation, save, run_sweep!

"""
    run_simulation(state, params)

## Inputs
- state : Vector
    - N array of initial state
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
    - params["alpha"] : float
        - float governing the weight of individual vs system fitness for updating probabilities.
        - alpha in [0, 1]. alpha = 0 only considers individual fitness. alpha = 1 only considers system fitness
    - params["moveType"] : string
        - either "random" or "local", sets whether moving can happen anywhere ("random") or only to nearest neighbors ("local")
    - params["fill"] : float
        - what fraction of available sites are filled. Must be less than 1

## Outputs
- state : array
    - NxM array of integers that gives state of the system after evolving for n_sweeps
"""
function run_simulation!(state::Vector{Int64},
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
        run_sweep!(state, params, total_occupants)
        
        # snapshot = -1 means only
        # save initial and final
        if snapshot > 0
            if ii % snapshot == 0
                save(state, params, sweep=ii)
            end
        end
    end
    
    # save final state
    # at worst, overwrites itself
    save(state, params, sweep=n_sweeps)
    
    return state
end


function run_simulation(state::Vector{Int64},
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
        run_sweep!(state, params, total_occupants)
        
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


function run_sweep!(state::Vector{Int64},
                    params::Dict{String, Any},
                    n_steps::Integer)
    for jj in 1:n_steps
        move!(state, params, n_steps)
    end
    return state
end


function random_state(params::Dict{String, Any})
    # unpack parameters
    gs = params["grid_size"]
    capacity = params["capacity"]
    fill = params["fill"]
    
    total_occupants = Int(floor(gs * capacity * fill))
    locations = rand(1:gs, total_occupants)
    state = [count(==(i), locations) for i in 1:gs]
    
    return state
end


"""
This function selects a single particle to move.

Unfortunately, we do not have a
list of particles and where they are.

Instead, we select a site proportionally
to how many particles occupy that site.
"""
function select_particle(density::Vector{Float64})
    nzidxs = findall(>(0), density)
    v_sum = 0
    r = rand()
    n = 1
    while v_sum < r
        v_sum += density[nzidxs[n]]
        n += 1
    end
    from = nzidxs[n - 1]
    return from
end


"""
    pick_sites(state, moveType, capacity, N)

Function to find where to move a particle
from and to
"""
function pick_sites(state::Vector{Int64},
                    moveType::String,
                    capacity::Int64,
                    N::Int64)
    
    from = select_particle(state / sum(state))
    
    if moveType == "random"
        # pick a random place
        to = rand(1:N)

        # make sure destination is not the same place
        # you want to move from, nor is it full 
        while to == from || sum(state[to]) >= capacity
            to = rand(1:N)
        end

    elseif moveType == "local"
        neighbors = [mod1(from + 1, N),  # right
                     mod1(from - 1, N)]  # left
        avail = [state[n] < capacity for n in neighbors]
        
        # if all neighbors are full, pick a new "from" location
        while sum(avail) == 0
            from = select_particle(state / sum(state))
            neighbors = [mod1(from + 1, N),  # right
                         mod1(from - 1, N)]  # left
            avail = [state[n] < capacity for n in neighbors]
        end
        to = rand(neighbors[avail])
    else
        error("moveTpe = random or local")
    end

    return from, to
end


function move!(state::Vector{Int64},
               params::Dict{String, Any},
               total_occupants::Int64)
    # unpack parameters
    moveType = params["moveType"]
    capacity = params["capacity"]
    N = params["grid_size"]
    
    # randomly select where to move from,
    # proportionally to number of individuals there
    from, to = pick_sites(state, moveType, capacity, N)

    # find gain
    G = gain(state[from...], state[to...], params)

    # calculate probability of moving using Glauber-like rule
    prob_to_move = glauber_prob(G, params)

    # draw a random number to select whether to move or not
    if rand() < prob_to_move
        state[from...] -= 1
        state[to...] += 1
    end
    
    return state
end


# function pick_loc(state::Vector{Int})

"""
Calculate gain in utility
"""
function gain(from::Int64,
              to::Int64,
              params::Dict{String, Any}) 
    # unpack parameters
    alpha = params["alpha"]
    capacity = params["capacity"]
    utility_func = params["utility_func"]
    
    # calculate new and old densities of origin and destination
    from_old = from / capacity
    to_old = to / capacity
    from_new = (from - 1) / capacity
    to_new = (to + 1) / capacity
    
    # calculate new and old densities
    utility_from_old = utility_func(from_old, params)
    utility_from_new = utility_func(from_new, params)
    utility_to_old = utility_func(to_old, params)
    utility_to_new = utility_func(to_new, params)
    
    # calculate change in utility only of the mover
    selfish = utility_to_new - utility_from_old
    # calculate change in utility of everyone
    altruistic = (    to_new * utility_to_new
                  -   to_old * utility_to_old
                  + from_new * utility_from_new
                  - from_old * utility_from_old)

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
function glauber_prob(deltaG::AbstractFloat, params::Dict{String, Any})
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
- x : float
    - fill fraction, x in [0, 1]
- params : Dict
    - preferred_density : Float
        - value of density that maximizes the utility
          params["preferred_density"] in [0, 1]
    - m : Float
        - how much you prefer x = 1
          params["m"] in [0, 1]

## Outputs
- utility : Float
    - utility of quantity x, given x0 and m. utility in [0, 1]
"""
function asymmetric_utility(x::AbstractFloat,
                            params::Dict{String, Any})
    # unpack parameters
    x0 = params["preferred_density"]
    capacity = params["capacity"]
    m = params["m"]
    
    if x < x0
        utility = 2 * x
    else
        utility = m +  2 * (1 - m) * (1 - x)
    end

    return utility
end

"""
    quadratic_utility(x, params)

Calculate the quadratic utility function for an array

## Inputs
- x : float
    - fill fraction. x in [0, 1]
- params : Dict
    - unused, but passed to match with other utility functions

## Outputs
- utility : float
    - utility of particle. utility in [0, 1]
"""
function quadratic_utility(x::AbstractFloat,
                           params::Dict{String, Any})
    utility = 4 * x * (1 - x)
    
    return utility
end


function save(state::Vector{Int64},
              params::Dict{String, Any};
              sweep::Int64 = -1)
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
function save(state_init::Vector{Int64},
              state::Vector{Int64},
              params::Dict{String, Any};
              sweep::Int64 = -1)
    
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


function save(state::Vector{Int64},
              params::Dict{String, Any},
              savefolder::String,
              filename::String)
    
    mkpath(savefolder)
    savepath = savefolder * "/" * filename
    
    h5open(savepath, "w") do fid
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

# function periodic_index(idx::CartesianArray{2}, N::Integer)
#     idx_new = similar(idx)
#     idx[findall(idx .> [N, N])] .= 1
#     idx[findall(idx .< [1, 1])] .= N

#     return CartesianIndex((idx[1], idx[2]))
# end

end
