module Schelling2D1S
using ProgressMeter
using HDF5
export random_state, move!, gain, glauber_prob, asymmetric_utility, quadratic_utility, run_simulation!, run_simulation, save

"""
    run_simulation(state, params)

Run a simulation of the modified Shelling model presented in Grauwin et al, PNAS 2009

## Inputs
- state : Array
    - NxM array of initial state
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
function run_simulation!(state::Matrix{Int},
                         params::Dict{String, Any})
    n_sweeps = params["n_sweeps"]
    N = params["grid_size"]
    total_occupants = sum(state)
    @showprogress 1 for ii in 1:n_sweeps
        for jj in 1:total_occupants
            move!(state, params)
        end
    end
    return state
end


function run_simulation(state::Matrix{Int},
                        params::Dict{String, Any})
    n_sweeps = params["n_sweeps"]
    N = params["grid_size"]
    total_occupants = sum(state)
    state_new = deepcopy(state)
    @showprogress 1 for ii in 1:n_sweeps
        for jj in 1:total_occupants
            move!(state_new, params)
        end
    end
    return state_new
end


function random_state(params::Dict{String, Any})
    # unpack parameters
    gs = params["grid_size"]
    capacity = params["capacity"]
    fill = params["fill"]
    
    total_occupants = Int(gs^2 * capacity * fill)
    locations = rand(1:gs^2, total_occupants)
    state = reshape([count(==(i), locations) for i in 1:gs^2], (gs, gs))
    
    return state
end


function move!(state::Matrix{Int},
               params::Dict{String, Any})
    # unpack parameters
    moveType = params["moveType"]
    capacity = params["capacity"]
    N = params["grid_size"]
    
    # randomly select where to move from,
    # proportionally to number of individuals there
    nzidxs = findall(>(0), state)
    vals = cumsum([state[idx] for idx in nzidxs])
    selection = findfirst((vals / maximum(vals)) .> rand())
    
    from = [nzidxs[selection][1], nzidxs[selection][2]]
    

    if moveType == "random"
        # pick a random place
        to = rand(1:N, 2)

        # make sure destination is not the same place
        # you want to move from, nor is it full 
        while to == from || state[to...] >= capacity
            to = rand(1:N, 2)
        end

    elseif moveType == "local"
        # get neighbors including periodic boundaries
        neighbors = [mod1.(from + [+1, 0], N),  # north
                     mod1.(from + [-1, 0], N),  # south
                     mod1.(from + [0, +1], N),  # east
                     mod1.(from + [0, -1], N)]  # west
        
        # only move to places filled less than capacity
        avail = [state[n...] < capacity for n in neighbors]
        
        # if all neighbors are full, pick a new "from" location
        while sum(avail) == 0
            selection = findfirst((vals / maximum(vals)) .> rand())
            from = [nzidxs[selection][1], nzidxs[selection][2]]
            
            neighbors = [mod1.(from + [+1, 0], N),  # north
                         mod1.(from + [-1, 0], N),  # south
                         mod1.(from + [0, +1], N),  # east
                         mod1.(from + [0, -1], N)]  # west
            avail = [state[n...] < capacity for n in neighbors]
        end
        to = rand(neighbors[avail])
    else
        error("moveType = random or local")
    end

    # find gain
    G = gain(state[from...], state[to...], params)

    # calculate probability of moving using Glauber-like rule
    prob_to_move = glauber_prob(G, params)

    # draw a random number to select whether to move or not
    if rand() < prob_to_move
        state[from...] -= 1
        state[to...] += 1
    end

end


"""
Calculate gain in utility
"""
function gain(from::Int64, to::Int64,
              params::Dict{String, Any}) 
    # unpack parameters
    alpha = params["alpha"]
    capacity = params["capacity"]
    utility_func = params["utility_func"]
    
    # calculate new and old densities of origin and destination
    from_old = from / capacity
    from_new = (from - 1) / capacity
    to_old = to / capacity
    to_new = (to + 1) / capacity
    
    # calculate change in utility only of the mover
    selfish = utility_func(to_new, params) - utility_func(from_old, params)
    # calculate change in utility of everyone
    altruistic = (    to_new * utility_func(to_new,   params)
                  -   to_old * utility_func(to_old,   params)
                  + from_new * utility_func(from_new, params)
                  - from_old * utility_func(from_old, params))

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
- x : float
    - total number in [0, capacity]
- params : Dict
    - preferred_density : Float
        - value of density that maximizes the utility. params["preferred_density"] in [0, 1]
    - m : Float
        - how much you prefer x = 1. params["m"] in [0, 1]

## Outputs
- utility : Float
    - utility of quantity x, given x0 and m. utility in [0, 1]
"""
function asymmetric_utility(x::Float64, params::Dict{String, Any})
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


function quadratic_utility(x::Float64, params::Dict{String, Any})
    capacity = params["capacity"]
    utility = 4 * x * (1 - x)
    
    return utility
end


function save(state::Matrix{T},
              params::Dict{String, Any}) where T<:Real
    savepath = params["savepath"]
    
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


function save(state::Matrix{T},
              params::Dict{String, Any},
              savefolder::String,
              filename::String) where T<:Real
    
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


function save(state_init::Matrix{T},
              state::Matrix{T},
              params::Dict{String, Any}) where T<:Real
    savepath = params["savepath"]
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
