import os
import json
from glob import glob
import fipy as fp
import numpy as np
import argparse

from fvm_utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-duration", type=float, default=1000.0,
                        help="total duration of simulation")
    parser.add_argument("-nt", type=int, default=100,
                        help="number of time points to save")
    parser.add_argument("-dt", type=float, default=1e-3,
                        help="time step")
    parser.add_argument("-nx", type=int, default=40,
                        help="number of cells in x")
    parser.add_argument("-dx", type=float, default=0.25,
                        help="cell size in x")
    parser.add_argument("-ny", type=int, default=40,
                        help="system size in y")
    parser.add_argument("-dy", type=float, default=0.25,
                        help="cell size in y")
    parser.add_argument("-k1", type=float, default=1.0,
                        help="self-utility of group 1")
    parser.add_argument("-k2", type=float, default=1.0,
                        help="self-utility of group 2")
    parser.add_argument("-kp", type=float, default=0.0,
                        help="symmetric cross-utility")
    parser.add_argument("-km", type=float, default=0.0,
                        help="antisymmetric cross-utility")
    parser.add_argument("-nu", type=float, default=0.0,
                        help="strength of non-linearity in utility of group 2")
    parser.add_argument("-phi1", type=float, default=0.25,
                        help="initial uniform state of group 1")
    parser.add_argument("-phi2", type=float, default=0.25,
                        help="initial uniform state of group 2")
    parser.add_argument("-temp", type=float, default=0.1,
                        help="temperature")
    parser.add_argument("-gamma", type=float, default=1.0,
                        help="strength of gradient penalties")
    parser.add_argument("-savefolder", type=str, default=".",
                        help="path to folder to save output")
    parser.add_argument("-filename", type=str, default="data",
                        help="name of file to save output")
    args = parser.parse_args()

    ### set up save environment ###
    datafile = os.path.join(args.savefolder, args.filename + ".hdf5")
    paramfile = os.path.join(args.savefolder, args.filename + "_params.json")
    if not os.path.exists(args.savefolder):
        os.makedirs(args.savefolder)
    else:
        files = glob(os.path.join(args.savefolder, args.filename + "*"))
        for file in files:
            try:
                os.remove(file)
            except Exception as e:
                print(f"could not delete {file}. Reason: {e}")
    ###############

    ### save params ###
    with open(paramfile, "w") as p:
        p.write(json.dumps(vars(args), indent=4))
    ###############

    ### build simulation ###
    # model parameters
    κ1 = args.k1
    κp = args.kp
    κm = args.km
    κ2 = args.k2
    κ = np.array([[κ1, κp - κm],
                  [κp + κm, κ2]])
    nu = args.nu
    Γ1 = args.gamma  # surface tension of 0
    Γ2 = args.gamma  # surface tension of 1
    temp = args.temp # diffusion constant

    ϕ2_init = args.phi1  # uniform state of 0
    ϕ2_init = args.phi2  # uniform state of 1

    # simulation parameters
    nx = args.nx
    ny = args.ny
    dx = args.dx
    dy = args.dy
    dt = args.dt
    duration = args.duration

    mesh = fp.Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)
    
    ϕ1 = fp.CellVariable(name=r"$\phi_0$", mesh=mesh)
    μ̃1 = fp.CellVariable(name=r"$\tilde{\mu}_0$", mesh=mesh)
    ϕ2 = fp.CellVariable(name=r"$\phi_1$", mesh=mesh)
    μ̃2 = fp.CellVariable(name=r"$\tilde{\mu}_1$", mesh=mesh)

    noise1 = fp.GaussianNoiseVariable(mesh=mesh, mean=ϕ2_init,
                                      variance=0.0001).value
    noise2 = fp.GaussianNoiseVariable(mesh=mesh, mean=ϕ2_init,
                                      variance=0.0001).value
    ϕ1[:] = noise1
    ϕ2[:] = noise2


    M1 = ϕ1 * (1 - ϕ1 - ϕ2)
    π1 = κ[0, 0] * ϕ1 + κ[0, 1] * ϕ2
    μ1 = -π1 + temp * (np.log(ϕ1) - np.log(1 - ϕ1 - ϕ2))
    dμ1dϕ1 = -κ[0, 0] + temp * (1 - ϕ2) / (ϕ1 * (1 - ϕ1 - ϕ2))
    
    M2 = ϕ2 * (1 - ϕ1 - ϕ2)
    π2 = κ[1, 0] * ϕ1 + κ[1, 1] * ϕ2 * (1 - nu * ϕ2)
    μ2 = -π2 + temp * (np.log(ϕ2) - np.log(1 - ϕ1 - ϕ2))
    dμ2dϕ2 = -κ[1, 1] + temp * (1 - ϕ1) / (ϕ2 * (1 - ϕ1 - ϕ2))
    
    eq1_1 = (fp.TransientTerm(var=ϕ1) == fp.DiffusionTerm(coeff=M1, var=µ̃1))
    
    eq1_2 = (fp.ImplicitSourceTerm(coeff=1, var=µ̃1)
             == fp.ImplicitSourceTerm(coeff=dμ1dϕ1, var=ϕ1)
             - dμ1dϕ1 * ϕ1 + μ1
             - fp.DiffusionTerm(coeff=Γ1, var=ϕ1))
    
    eq2_1 = (fp.TransientTerm(var=ϕ2) == fp.DiffusionTerm(coeff=M2, var=µ̃2))
    
    eq2_2 = (fp.ImplicitSourceTerm(coeff=1, var=µ̃2)
             == fp.ImplicitSourceTerm(coeff=dμ2dϕ2, var=ϕ2)
             - dμ2dϕ2 * ϕ2 + μ2
             - fp.DiffusionTerm(coeff=Γ2, var=ϕ2))

    eq = eq1_1 & eq1_2 & eq2_1 & eq2_2
    ###############
    
    ### run simulation ###
    nt = args.nt
    t_save = np.linspace(0, duration, nt+1)
    elapsed = 0
    flag = 0
    
    # save things common to all time points
    gname = "common"
    datadict = {"x": mesh.cellCenters.value[0],
                "y": mesh.cellCenters.value[1]}
    dump(datafile, gname, datadict)

    # save initial condition
    gname = f"n{flag:06d}"
    datadict = {"t": t_save[flag],
                "phi1": ϕ1.value,
                "phi2": ϕ2.value}
    dump(datafile, gname, datadict)
    flag += 1
    
    while elapsed < duration:
        eq.solve(dt=dt)
        elapsed += dt
        if elapsed >= t_save[flag]:
            print(f"t={elapsed:0.2f} / {duration}", end="\r")
            gname = f"n{flag:06d}"
            datadict = {"t": t_save[flag],
                        "phi1": ϕ1.value,
                        "phi2": ϕ2.value}
            dump(datafile, gname, datadict)
            flag += 1

    ###############
    # dϕ1dy, dϕ1dx = np.gradient(ϕ_array[0], dy, dx, axis=(1, 2))
    # dϕ2dy, dϕ2dx = np.gradient(ϕ_array[1], dy, dx, axis=(1, 2))
