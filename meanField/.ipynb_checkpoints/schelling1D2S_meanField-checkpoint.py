import numpy as np
import dedalus.public as d3
import os
import json
import argparse
import shutil
import warnings
import logging
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-Lx", type=float, default=40,
                        help="system size")
    parser.add_argument("-Nx", type=float, default=64,
                        help="number of spectral modes/spatial points")
    parser.add_argument("-t_stop", type=float, default=100,
                        help="total simulation time")
    parser.add_argument("-dt", type=float, default=0.01,
                        help="simulation time step")
    parser.add_argument("-dealias", type=float, default=3/2,
                        help="dealiasing factor for dedalus")
    parser.add_argument("-alpha", type=float, default=0.0,
                        help="altruism")
    parser.add_argument("-delta", type=float, default=0.0,
                        help="non-reciprocity")
    parser.add_argument("-T", type=float, default=0.1,
                        help="temperature, plays role of diffusion constant")
    parser.add_argument("-fillA", type=float, default=0.25,
                        help="fill of particle A")
    parser.add_argument("-fillB", type=float, default=0.25,
                        help="fill of particle B")
    parser.add_argument("-save_dt", type=float, default=0.1,
                        help="how often to save output in simulation time units")
    parser.add_argument("-savepath", type=str, default=".",
                        help="where to save output")

    args = parser.parse_args()

    # simulation parameters
    Lx = args.Lx
    Nx = args.Nx
    t_stop = args.t_stop 
    Δt = args.dt
    dealias = args.dealias
    timestepper = d3.SBDF2 # default
    dtype = np.float64 # default
    sim_dt = args.save_dt

    # schelling parameters
    α = args.alpha
    δ = args.delta
    T = args.T
    ϕA0 = args.fillA
    ϕB0 = args.fillB

    # check stability
    Δx = Lx / Nx
    if Δt * Δx**(-4) > 1/8:
        warnings.warn("von Neumann stability not satisfied")

    # 1D basis
    coord = d3.Coordinate('x')
    dist = d3.Distributor(coord, dtype=dtype)
    xbasis = d3.RealFourier(coord, Nx, bounds=(-Lx/2, Lx/2), dealias=dealias)

    # fields
    ϕA = dist.Field(name="phiA", bases=(xbasis))
    ϕB = dist.Field(name="phiB", bases=(xbasis))

    # problem
    problem = d3.IVP([ϕA, ϕB], namespace=locals())

    # substitutions
    dx = lambda A: d3.Differentiate(A, coord)
    # total density
    ϕ = ϕA + ϕB 
    π = lambda x: 4 * x * (1 - x)
    dπ = lambda x: 4 * (1 - 2 * x)
    # πA, dπA/dϕA, and dπA/dϕB
    πA = π(ϕA) + δ * π(ϕB)
    dπA_dϕA = dπ(ϕA)
    dπA_dϕB = δ * dπ(ϕB)
    # πB, dπA/dϕA, and dπA/dϕB
    πB = π(ϕB) - δ * π(ϕA)
    dπB_dϕA = -δ * dπ(ϕA)
    dπB_dϕB = dπ(ϕB)
    # total fitness
    UA = ϕA * πA
    UB = ϕB * πB
    U = UA + UB
    dU_dϕA = πA + ϕA * dπA_dϕA + ϕB * dπB_dϕA
    dU_dϕB = πB + ϕB * dπB_dϕB + ϕA * dπA_dϕB

    ### Schelling ####
    # add equation
    problem.add_equation("dt(ϕA) - T*dx(dx(ϕA)) = -dx(ϕA * (1 - ϕ) * ((1 - α) * dx(πA) + α * dx(dU_dϕA))) - dx(ϕA * (1 - ϕ) * dx(dx(dx(ϕA))))")
    problem.add_equation("dt(ϕB) - T*dx(dx(ϕB)) = -dx(ϕB * (1 - ϕ) * ((1 - α) * dx(πB) + α * dx(dU_dϕB))) - dx(ϕB * (1 - ϕ) * dx(dx(dx(ϕB))))")
    #################

    ### solver and stopping ###
    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = t_stop
    #################

    ### initial condition ###
    δϕA = 0.05
    δϕB = 0.05
    randA = np.random.uniform(low=-1, high=+1, size=Nx)
    randA -= randA.mean()
    randB = np.random.uniform(low=-1, high=+1, size=Nx)
    randB -= randB.mean()
    ϕA["g"] = ϕA0 + δϕA * randA
    ϕB["g"] = ϕB0 + δϕB * randB
    #################

    ### storage ###
    shutil.rmtree(args.savepath, ignore_errors=True)
    snapshots = solver.evaluator.add_file_handler(args.savepath,
                                                  sim_dt=sim_dt,
                                                  max_writes=1000)
    snapshots.add_task(ϕA, layout="g", name="phiA")
    snapshots.add_task(ϕB, layout="g", name="phiB")
    snapshots.add_task(UA, layout="g", name="UA")
    snapshots.add_task(UB, layout="g", name="UB")

    # store parameters
    params = {}
    params["Lx"] = args.Lx
    params["Nx"] = args.Nx
    params["t_stop"] = args.t_stop 
    params["dt"] = args.dt
    params["dealias"] = args.dealias
    params["alpha"] = args.alpha
    params["delta"] = args.delta
    params["T"] = args.T
    params["fillA"] = args.fillA
    params["fillB"] = args.fillB
    params["save_dt"] = args.save_dt
    #################
    
    # main loop
    try:
        logger.info('Starting main loop...')
        while solver.proceed:
            solver.step(Δt)

            # nan check
            if np.isnan(np.min(ϕA["g"])) or np.isnan(np.min(ϕB["g"])):
                    raise RuntimeError("got NaNs. Ending main loop.")

            # progress logger
            if solver.iteration % 10000 == 0:
                logger.info(f"Iteration={solver.iteration}, Time={solver.sim_time:0.2f}")
    except:
        logger.error("Exception raised, ending main loop.")
        raise
    finally:
        solver.log_stats()
        with open(os.path.join(args.savepath, "params.json"), "w") as p:
            params_str = json.dumps(vars(args), indent=4)
            p.write(params_str)

