import os
import json
from glob import glob
import fipy as fp
import numpy as np
import argparse
import h5py

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
    
    parser.add_argument("-k11", type=float, default=1.0)
    parser.add_argument("-k22", type=float, default=1.0)
    parser.add_argument("-k12", type=float, default=0.0)
    parser.add_argument("-k21", type=float, default=0.0)
    parser.add_argument("-nu111", type=float, default=0.0)
    parser.add_argument("-nu112", type=float, default=0.0)
    parser.add_argument("-nu122", type=float, default=0.0)
    parser.add_argument("-nu222", type=float, default=0.0)
    parser.add_argument("-nu212", type=float, default=0.0)
    parser.add_argument("-nu211", type=float, default=0.0)
    parser.add_argument("-temp1", type=float, default=0.1)
    parser.add_argument("-temp2", type=float, default=0.1)
    parser.add_argument("-gamma1", type=float, default=1.0)
    parser.add_argument("-gamma2", type=float, default=1.0)

    parser.add_argument("-icType", type=str, default="uniform",
                        choices=["uniform", "input"])
    parser.add_argument("-phi1", type=float, default=0.25,
                        help="initial uniform state of group 1")
    parser.add_argument("-phi2", type=float, default=0.25,
                        help="initial uniform state of group 2")
    parser.add_argument("-datapath", type=str, default=".",
                        help="path to data with initial condition data")
    
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
    k11 = args.k11
    k12 = args.k12
    k21 = args.k21
    k22 = args.k22
    ν111 = args.nu111
    ν112 = args.nu112
    ν122 = args.nu122
    ν222 = args.nu222
    ν212 = args.nu212
    ν211 = args.nu211
    Γ1 = args.gamma1
    Γ2 = args.gamma2
    temp1 = args.temp1
    temp2 = args.temp2

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

    if args.icType == "uniform":
        ic1 = fp.GaussianNoiseVariable(mesh=mesh, mean=args.phi1,
                                          variance=0.0001).value
        ic2 = fp.GaussianNoiseVariable(mesh=mesh, mean=args.phi2,
                                          variance=0.0001).value
    elif args.icType == "input":
        # assume that we want last snapshot of hdf5 file
        with h5py.File(args.datapath, "r") as d:
            keys = list(d.keys())
            ic1 = d[keys[-1]]["phi1"][:]
            ic2 = d[keys[-1]]["phi2"][:]

    ϕ1[:] = ic1
    ϕ2[:] = ic2

    M1 = ϕ1 * (1 - ϕ1 - ϕ2)
    π1 = k11 * ϕ1 + k12 * ϕ2 + ν111 * ϕ1 * ϕ1 + ν112 * ϕ1 * ϕ2 + ν122 * ϕ2 * ϕ2
    dπ1dϕ1 = k11 + 2 * ν111 * ϕ1 + ν112 * ϕ2
    μ1 = -π1 + temp1 * (np.log(ϕ1) - np.log(1 - ϕ1 - ϕ2))
    dμ1dϕ1 = -dπ1dϕ1 + temp1 * (1 - ϕ2) / (ϕ1 * (1 - ϕ1 - ϕ2))
    
    M2 = ϕ2 * (1 - ϕ1 - ϕ2)
    π2 = k12 * ϕ1 + k22 * ϕ2 + ν211 * ϕ1 * ϕ1 + ν212 * ϕ1 * ϕ2 + ν222 * ϕ2 * ϕ2
    dπ2dϕ2 = k22 + ν212 * ϕ1 + 2 * ν222 * ϕ2
    μ2 = -π2 + temp2 * (np.log(ϕ2) - np.log(1 - ϕ1 - ϕ2))
    dμ2dϕ2 = -dπ2dϕ2 + temp2 * (1 - ϕ1) / (ϕ2 * (1 - ϕ1 - ϕ2))
    
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
