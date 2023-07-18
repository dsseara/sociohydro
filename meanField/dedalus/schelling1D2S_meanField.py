import numpy as np
import dedalus.public as d3
import os
import json
import argparse
import shutil
import warnings
import logging
import h5py
from glob import glob
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)


def load_data(filename):
    with h5py.File(filename, "r") as d:
        ϕA = d['tasks']["phiA"][()]
        ϕB = d['tasks']["phiB"][()]
        x = d['tasks']['phiA'].dims[1][0][()]
        t = d['tasks']['phiA'].dims[0]["sim_time"][()]
    return ϕA, ϕB, x, t


def plotter(savepath):
    savename = savepath.split(os.path.sep)[-1]
    files = glob(os.path.join(savepath, "*h5"))

    # load params
    with open(os.path.join(savepath, "params.json"), "r") as p:
        params = json.load(p)

    fig, ax = plt.subplots(1, 3, dpi=150,
                           figsize=(9, 4),
                           sharey=True,
                           sharex=True)
    for file in files:
        ϕA, ϕB, x, t = load_data(file)
        a0 = ax[0].pcolormesh(x, t, ϕA,
                              cmap="Blues",
                              vmin=0, vmax=1,
                              rasterized=True)
        a1 = ax[1].pcolormesh(x, t, ϕB,
                              cmap="Reds",
                              vmin=0, vmax=1,
                              rasterized=True)
        a2 = ax[2].pcolormesh(x, t, ϕA - ϕB,
                              cmap="RdBu_r",
                              vmin=-1, vmax=1,
                              rasterized=True)

    cax0 = ax[0].inset_axes([1.05, 0.0, 0.05, 1])
    cbar0 = fig.colorbar(a0, cax=cax0, ax=ax[0])
    cbar0.ax.set(title=r"$\phi^A$")
    ax[0].set(xlabel=r"$x$", ylabel=r"$t$")
    cax1 = ax[1].inset_axes([1.05, 0.0, 0.05, 1])
    cbar1 = fig.colorbar(a1, cax=cax1, ax=ax[1])
    cbar1.ax.set(title=r"$\phi^B$")
    ax[1].set(xlabel=r"$x$")
    cax2 = ax[2].inset_axes([1.05, 0.0, 0.05, 1])
    cbar2 = fig.colorbar(a2, cax=cax2, ax=ax[2])
    cbar2.ax.set(title=r"$\phi^A - \phi^B$")
    ax[2].set(xlabel=r"$x$")
    plt.tight_layout()

    fig.savefig(os.path.join(savepath, savename + "_kymo.pdf"))

    return 0


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
    parser.add_argument("-dealias", type=float, default=1.5,
                        help="dealiasing factor for dedalus")
    parser.add_argument("-alpha", type=float, default=0.0,
                        help="altruism")
    parser.add_argument("-delta", type=float, default=0.0,
                        help="non-reciprocity")
    parser.add_argument("-kappa", type=float, default=1.0,
                        help="reciprocal interactions")
    parser.add_argument("-Gamma", type=float, default=1.0,
                        help="stabilizing strength")
    parser.add_argument("-T", type=float, default=0.1,
                        help="temperature, plays role of diffusion constant")
    parser.add_argument("-fillA", type=float, default=0.25,
                        help="fill of particle A")
    parser.add_argument("-fillB", type=float, default=0.25,
                        help="fill of particle B")
    parser.add_argument("-save_dt", type=float, default=1.0,
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
    sim_dt = args.save_dt
    timestepper = d3.SBDF2  # default
    dtype = np.float64  # default

    # schelling parameters
    α = args.alpha
    δ = args.delta
    κ = args.kappa
    T = args.T
    Γ = args.Gamma
    ϕA0 = args.fillA
    ϕB0 = args.fillB

    # check stability
    Δx = Lx / Nx
    if Δt * Δx**(-4) > 1 / 8:
        warnings.warn("von Neumann stability not satisfied")

    # 1D basis
    coord = d3.Coordinate('x')
    dist = d3.Distributor(coord, dtype=dtype)
    xbasis = d3.RealFourier(coord, Nx, bounds=(-Lx / 2, Lx / 2), dealias=dealias)

    # fields
    ϕA = dist.Field(name="phiA", bases=(xbasis))
    ϕB = dist.Field(name="phiB", bases=(xbasis))

    # problem
    problem = d3.IVP([ϕA, ϕB], namespace=locals())

    # substitutions
    dx = lambda A: d3.Differentiate(A, coord)
    # mobilities
    μA = ϕA * (1 - ϕA - ϕB)
    μB = ϕB * (1 - ϕA - ϕB)

    # fitness and derivative
    π = lambda x: x - 1
    dπ = lambda x: 1

    # πA, dπA/dϕA, and dπA/dϕB
    πA = π(ϕA) + (κ - δ) * ϕB / 2
    dπA_dϕA = dπ(ϕA)
    dπA_dϕB = (κ - δ) / 2
    # πB, dπA/dϕA, and dπA/dϕB
    πB = π(ϕB) + (κ + δ) * ϕA / 2
    dπB_dϕA = (κ + δ) / 2
    dπB_dϕB = dπ(ϕB)
    # total fitness
    U = ϕA * πA + ϕB * πB
    dU_dϕA = πA + ϕA * dπA_dϕA + ϕB * dπB_dϕA
    dU_dϕB = πB + ϕB * dπB_dϕB + ϕA * dπA_dϕB


    ### Schelling ####
    # add equation
    problem.add_equation("dt(ϕA) - T*dx(dx(ϕA)) = T*(ϕA * dx(dx(ϕB)) - ϕB * dx(dx(ϕA))) - dx(μA * ((1 - α) * dx(πA) + α * dx(dU_dϕA))) - dx(Γ * μA * dx(dx(dx(ϕA))))")
    problem.add_equation("dt(ϕB) - T*dx(dx(ϕB)) = T*(ϕB * dx(dx(ϕA)) - ϕA * dx(dx(ϕB))) - dx(μB * ((1 - α) * dx(πB) + α * dx(dU_dϕB))) - dx(Γ * μB * dx(dx(dx(ϕB))))")
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
    os.makedirs(args.savepath, exist_ok=True)
    shutil.rmtree(args.savepath, ignore_errors=True)
    snapshots = solver.evaluator.add_file_handler(args.savepath,
                                                  sim_dt=sim_dt,
                                                  max_writes=1000)
    snapshots.add_task(ϕA, layout="g", name="phiA")
    snapshots.add_task(ϕB, layout="g", name="phiB")

    # store parameters
    params = {}
    params["Lx"] = args.Lx
    params["Nx"] = args.Nx
    params["t_stop"] = args.t_stop
    params["dt"] = args.dt
    params["dealias"] = args.dealias
    params["alpha"] = args.alpha
    params["delta"] = args.delta
    params["kappa"] = args.kappa
    params["Gamma"] = args.Gamma
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
            if solver.iteration % 50000 == 0:
                time_spent = solver.get_wall_time() - solver.start_time
                logger.info(rf"Iteration={solver.iteration}, sim time={solver.sim_time:0.2f} / {t_stop}, real Δt ={time_spent:0.2f} s")
    except:
        logger.error("Exception raised, ending main loop.")
        raise
    finally:
        solver.log_stats()
        with open(os.path.join(args.savepath, "params.json"), "w") as p:
            params_str = json.dumps(vars(args), indent=4)
            p.write(params_str)

    # plot kymo
    print("Making kymograph...")
    plotter(args.savepath)

    print("Done.")
