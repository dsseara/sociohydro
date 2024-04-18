import os
import json
from glob import glob
import fipy as fp
import numpy as np
import argparse
import h5py
from scipy import interpolate

from fvm_utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-datafile", type=str, required=True,
                        help="path to hdf5 file with interpolated data")
    parser.add_argument("-duration", type=float, default=1000.0,
                        help="total duration of simulation")
    parser.add_argument("-kW", type=float, default=1.0,
                        help="self-utility of White population")
    parser.add_argument("-kB", type=float, default=1.0,
                        help="self-utility of Black population")
    parser.add_argument("-kp", type=float, default=0.0,
                        help="symmetric cross-utility")
    parser.add_argument("-km", type=float, default=0.0,
                        help="antisymmetric cross-utility")
    parser.add_argument("-nu", type=float, default=0.0,
                        help="strength of non-linearity in Black utility")
    parser.add_argument("-temp", type=float, default=0.1,
                        help="temperature")
    parser.add_argument("-gamma", type=float, default=1.0,
                        help="strength of gradient penalties")
    parser.add_argument("-capacityType", type=str, default="local",
                        choices=["uniform", "local"],
                        help="how to calculate the carrying capacities")
    parser.add_argument("-buffer", type=float, default=1000.0,
                        help="buffer for boundary simplification")
    parser.add_argument("-simplify", type=float, default=1000.0,
                        help="simplify parameter for boundary simplification")
    parser.add_argument("-cellsize", type=float, default=3e3,
                        help="typical size of cells in mesh")
    parser.add_argument("-savefolder", type=str, default=".",
                        help="path to folder to save output")
    parser.add_argument("-filename", type=str, default="data",
                        help="name of file to save output")
    args = parser.parse_args()

    # set up save environment
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

    # save params
    with open(paramfile, "w") as p:
        p.write(json.dumps(vars(args), indent=4))

    with h5py.File(args.datafile, "r") as d:
        x_grid = d["1990"]["x_grid"][:]
        y_grid = d["1990"]["y_grid"][:]
        black = d["1990"]["black_grid_masked"][:]
        black_final = d["2020"]["black_grid_masked"][:]
        white = d["1990"]["white_grid_masked"][:]
        white_final = d["2020"]["white_grid_masked"][:]
        total = d["1990"]["total_grid_masked"][:]
        
        # calculate capacity
        if args.capacityType == "uniform":
            capacity = -1e10
            for key in d.keys():
                    capacity = max(capacity, np.nanmax(d[key]["white_grid_masked"][:] + d[key]["black_grid_masked"][:]))
        elif args.capacityType == "local":
            capacity = np.zeros(x_grid.shape)
            for key in d.keys():
                capacity = np.fmax(capacity, d[key]["white_grid_masked"][:] + d[key]["black_grid_masked"][:])

    # add a small bit to avoid zero, and inflate capacity to avoid one
    ϕW0_grid = (white) / (1.1 * capacity)
    ϕWf_grid = (white_final) / (1.1 * capacity)
    ϕB0_grid = (black) / (1.1 * capacity)
    ϕBf_grid = (black_final) / (1.1 * capacity)

    # all cases fall into this CRS
    crs = "ESRI:102003"
    mesh, simple_boundary, geo_file_contents = make_mesh(total, x_grid, y_grid, crs,
                                                         args.buffer,
                                                         args.simplify,
                                                         args.cellsize)

    # perform interpolation from grid points to cell centers
    grid_points = np.array([[x, y] for x, y in zip(x_grid.ravel(),
                                                   y_grid.ravel())])
    cell_points = np.array([[x, y] for x, y in zip(mesh.cellCenters.value[0],
                                                   mesh.cellCenters.value[1])])
    
    ϕW0_cell = interpolate.griddata(grid_points,
                                    np.nan_to_num(ϕW0_grid.ravel(), nan=1e-3),
                                    cell_points,
                                    fill_value=0)
    ϕWf_cell = interpolate.griddata(grid_points,
                                    np.nan_to_num(ϕWf_grid.ravel(), nan=1e-3),
                                    cell_points,
                                    fill_value=0)

    ϕB0_cell = interpolate.griddata(grid_points,
                                    np.nan_to_num(ϕB0_grid.ravel(), nan=1e-3),
                                    cell_points,
                                    fill_value=0)
    ϕBf_cell = interpolate.griddata(grid_points,
                                    np.nan_to_num(ϕBf_grid.ravel(), nan=1e-3),
                                    cell_points,
                                    fill_value=0)


    # utility parameters
    κW = args.kW
    κp = args.kp
    κm = args.km
    κB = args.kB
    nu = args.nu
    Γ0 = args.gamma
    Γ1 = args.gamma
    temp = args.temp

    # simulation parameters
    duration = args.duration
    
    ϕW = fp.CellVariable(name=r"$\phi_W$", mesh=mesh)
    μW = fp.CellVariable(name=r"$\tilde{\mu}_W$", mesh=mesh)
    ϕB = fp.CellVariable(name=r"$\phi_B$", mesh=mesh)
    μB = fp.CellVariable(name=r"$\tilde{\mu}_B$", mesh=mesh)

    ϕW[:] = ϕW0_cell
    ϕB[:] = ϕB0_cell

    mobilityW = ϕW * (1 - ϕW - ϕB)
    πW = κW * ϕW + (κp - κm) * ϕB
    μW_taylorExpand = -πW + temp * (np.log(ϕW) - np.log(1 - ϕW - ϕB))
    dμWdϕW = -κW + temp * (1 - ϕB) / (ϕW * (1 - ϕW - ϕB))
    
    mobilityB = ϕB * (1 - ϕW - ϕB)
    πB = (κp + κm) * ϕW + κB * ϕB * (1 - nu * ϕB)
    μB_taylorExpand = -πB + temp * (np.log(ϕB) - np.log(1 - ϕW - ϕB))
    dμBdϕB = -κB * (1 - 2 * nu * ϕB) + temp * (1 - ϕW) / (ϕB * (1 - ϕW - ϕB))
    
    eqW_1 = (fp.TransientTerm(var=ϕW) == fp.DiffusionTerm(coeff=mobilityW,
                                                          var=μW))
    eqW_2 = (fp.ImplicitSourceTerm(coeff=1, var=μW)
             == fp.ImplicitSourceTerm(coeff=dμWdϕW, var=ϕW)
             - dμWdϕW * ϕW + μW_taylorExpand
             - fp.DiffusionTerm(coeff=Γ0, var=ϕW))
    
    eqB_1 = (fp.TransientTerm(var=ϕB) == fp.DiffusionTerm(coeff=mobilityB,
                                                          var=μB))
    eqB_2 = (fp.ImplicitSourceTerm(coeff=1, var=μB)
             == fp.ImplicitSourceTerm(coeff=dμBdϕB, var=ϕB)
             - dμBdϕB * ϕB + μB_taylorExpand
             - fp.DiffusionTerm(coeff=Γ1, var=ϕB))

    eq = eqW_1 & eqW_2 & eqB_1 & eqB_2

    dexp = -5

    nt = 100
    ϕW_array = np.zeros((nt + 1, len(ϕW)))
    ϕB_array = np.zeros((nt + 1, len(ϕB)))
    ϕW_array[0] = ϕW.value
    ϕB_array[0] = ϕB.value
    t_save = np.linspace(0, duration, nt+1)

    elapsed = 0
    flag = 0
    while elapsed < duration:
        dt = min(50, np.exp(dexp))
        elapsed += dt
        dexp += 0.01
        eq.solve(dt=dt)
        print(f"t={elapsed:0.2f} / {duration}", end="\r")
        if elapsed >= t_save[flag]:
            ϕW_array[flag] = ϕW.value
            ϕB_array[flag] = ϕB.value
            flag += 1

    mseW = np.mean((ϕWf_cell - ϕW_array[-1])**2)
    mseB = np.mean((ϕBf_cell - ϕB_array[-1])**2)

    print(f"Mean-square error, White: {mseW:0.4f}")
    print(f"Mean-square error, Black: {mseB:0.4f}")

    with h5py.File(datafile, "a") as d:
        d.create_dataset("t_array", data=t_save)
        d.create_dataset("phiW_array", data=ϕW_array)
        d.create_dataset("phiB_array", data=ϕB_array)
        d.create_dataset("cell_centers", data=cell_points)
        d.create_dataset("phiW_2020", data=ϕWf_cell)
        d.create_dataset("phiB_2020", data=ϕBf_cell)
        d.create_dataset("mse_white", data=mseW)
        d.create_dataset("mse_black", data=mseB)