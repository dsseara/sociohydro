module shelling
using ProgressMeter
export random_state, move, gain, glauber_prob, asymmetric_utility, run_simulation

"""
    run_simulation(state, params)

Run a simulation of the modified Shelling model presented in Grauwin et al, PNAS 2009

## Inputs
- state : Array
    - NxM array of initial state
- params : dictionary
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
function run_simulation(state, params)
    n = params["n_sweeps"]
    total_occupants = sum(state)
    state_temp = deepcopy(state)
    @showprogress 1 for ii in 1:n
        for jj in 1:total_occupants
            state_temp = move(state_temp, params)
        end
    end
    
    return state_temp
end


function random_state(params)
    # unpack parameters
    gs = params["grid_size"]
    capacity = params["capacity"]
    
    total_occupants = Int((gs^2 * capacity) / 2)
    locations = rand(1:gs^2, total_occupants)
    state = reshape([count(==(i), locations) for i in 1:gs^2], (gs, gs))
    
    return state
end


function move(state, params)
    # unpack parameters
    moveType = params["moveType"]
    capacity = params["capacity"]
    
    # get size of array
    nrows, ncols = size(state)

    # randomly select where to move from,
    # taking care to not pick an empty place
    from = rand(findall(state .> 0))

    if moveType == "random"
        # pick a random place that is not full
        to = rand(findall(state .< capacity))

        # make sure destination is not the same place you want to move from
        while to == from
            to = rand(findall(state .< capacity))
        end

    elseif moveType == "local"
        # for annoying reasons having to do with
        # indexing, need to pass the neighbor location
        # as an array to take care of periodic boundaries
        fromArray = [i for i in from.I]
        neighbors = [periodic_index(fromArray + [+1, 0], nrows, ncols),
                     periodic_index(fromArray + [-1, 0], nrows, ncols),
                     periodic_index(fromArray + [0, +1], nrows, ncols),
                     periodic_index(fromArray + [0, -1], nrows, ncols)]

        available_neighbors = neighbors[state[neighbors] .< capacity]

        # make sure that we are not surrounded by
        # full lattice sites
        if length(available_neighbors) == 0
            return state
        end

        to = rand(available_neighbors)
    else
        error("moveType = random or local")
    end

    # find proposed state
    state_proposed = deepcopy(state)
    state_proposed[from] -= 1
    state_proposed[to] += 1

    # find density of current configuration
    density_current = state ./ capacity

    # find density of proposed configuration
    density_proposed = state_proposed ./ capacity

    # find gain
    G = gain(density_current, density_proposed, from, to, params)

    # calculate probability of moving using Glauber-like rule
    prob_to_move = glauber_prob(G, params)

    # draw a random number to select whether to move or not
    if rand() < prob_to_move
        return state_proposed
    else
        return state
    end
end


"""
Calculate gain in utility
"""
function gain(density_current, density_proposed,
              from, to, params)
    # unpack parameters
    alpha = params["alpha"]
    capacity = params["capacity"]
    
    # calculate utility of current configuration
    utility_current = asymmetric_utility(density_current, params)
    # calculate utility of proposed configuration
    utility_proposed = asymmetric_utility(density_proposed, params)

    delta_utility_local = utility_proposed[to] - utility_current[from]
    delta_utility_global = capacity * (sum(utility_proposed .* density_proposed) .- sum(utility_current .* density_current))
    # delta_utility_global = (utility_proposed[to] - utility_current[to]) + (utility_proposed[from] - utility_current[from])

    gain = alpha * delta_utility_global + (1 - alpha) * delta_utility_local

    return gain
end


"""
    glauber_prob(deltaG, params)

calculate update probability using glauber-like dynamics

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
function glauber_prob(deltaG, params)
    # unpack parameters
    T = params["temperature"]
    
    # Note that we are trying to maximize the gain
    # this introduces an extra minus sign compared to
    # minimization of the energy
    p = 1 / (1 + exp(-deltaG / T))
    return p
end


"""
    asymmetric_utility(x, params)

Calculate the asymmetric utility function for an array

## Inputs
- x : Array
    - Array of quantities to consider your utility based on. x in [0, 1]
- params : Dict
    - preferred_density : Float
        - value of density that maximizes the utility. params["preferred_density"] in [0, 1]
    - m : Float
        - how much you prefer x = 1. params["m"] in [0, 1]

## Outputs
- utility : Float
    - utility of quantity x, given x0 and m. utility in [0, 1]
"""
function asymmetric_utility(x, params)
    # unpack parameters
    x0 = params["preferred_density"]
    m = params["m"]
    
    utility = zeros(size(x))
    @. utility[x < x0] = 2 * x[x < x0]
    @. utility[x >= x0] = m + 2 * (1 - m) * (1 - x[x >= x0])

    return utility
end

function periodic_index(idx, nrows, ncols)
    idx[findall(idx .> [nrows, ncols])] .= 1
    idx[findall(idx .< [1, 1])] .= 1

    return CartesianIndex((idx[1], idx[2]))
end

end
