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
    parser.add_argument("-duration", type=float, default=100.0,
                        help="total duration of simulation")
    parser.add_argument("-nt", type=int, default=100,
                        help="number of time points to save")
    parser.add_argument("-timestepper", type=str, default="exp",
                        choices=["exp", "linear"],
                        help="whether to use exponentially increasing time steps or the same time step")
    parser.add_argument("-dt", type=float, default=1e-3,
                        help="size of time step for linear stepper")
    parser.add_argument("-dexp", type=float, default=-5,
                        help="if exponential time stepping, initial time step is exp(dexp)")
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
    parser.add_argument("-buffer", type=float, default=1.0,
                        help="buffer for boundary simplification")
    parser.add_argument("-simplify", type=float, default=1.0,
                        help="simplify parameter for boundary simplification")
    parser.add_argument("-cellsize", type=float, default=3,
                        help="typical size of cells in mesh")
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

    ### load and set-up data ###
    # initial condition
    ϕW0, ϕB0, x, y = get_data(args.datafile,
                              year=1990,
                              region="masked",
                              capacity_method=args.capacityType)
    
    # measure in terms of kilometers
    x /= 1000.0
    y /= 1000.0

    # final condition
    ϕWf, ϕBf, _, _ = get_data(args.datafile,
                              year=2020,
                              region="masked",
                              capacity_method=args.capacityType)
    
    # get correlation length of white population
    ξ, ξvar = get_corrLength(args.datafile, region="masked",
                             capacity_method=args.capacityType,
                             p0=[1, 10, 0])

    # measure distances in units of ξ
    # x /= ξ
    # y /= ξ

    # create mesh
    crs = "ESRI:102003"  # all cases fall into this CRS
    mesh, simple_boundary, geo_file_contents = make_mesh(ϕW0, x, y, crs,
                                                         args.buffer,
                                                         args.simplify,
                                                         args.cellsize)

    # perform interpolation from grid points to cell centers
    grid_points = np.array([[x, y] for x, y in zip(x.ravel(),
                                                   y.ravel())])
    cell_points = np.array([[x, y] for x, y in zip(mesh.cellCenters.value[0],
                                                   mesh.cellCenters.value[1])])
    
    ϕW0_cell = interpolate.griddata(grid_points,
                                    np.nan_to_num(ϕW0.ravel(), nan=1e-3),
                                    cell_points,
                                    fill_value=0)
    ϕWf_cell = interpolate.griddata(grid_points,
                                    np.nan_to_num(ϕWf.ravel(), nan=1e-3),
                                    cell_points,
                                    fill_value=0)

    ϕB0_cell = interpolate.griddata(grid_points,
                                    np.nan_to_num(ϕB0.ravel(), nan=1e-3),
                                    cell_points,
                                    fill_value=0)
    ϕBf_cell = interpolate.griddata(grid_points,
                                    np.nan_to_num(ϕBf.ravel(), nan=1e-3),
                                    cell_points,
                                    fill_value=0)
    ###############

    ### save things common to all time-points ###
    group_name = "common"
    datadict = {"cell_centers": cell_points,
                "phiW_initial": ϕW0_cell,
                "phiB_initial": ϕB0_cell,
                "phiW_final": ϕWf_cell,
                "phiB_final": ϕBf_cell,
                "corr_length": ξ,
                "corr_length_var": ξvar}
    dump(datafile, group_name, datadict)
    ###############

    ### build simulation ###
    # utility parameters
    κW = args.kW
    κp = args.kp
    κm = args.km
    κB = args.kB
    nu = args.nu
    Γ0 = args.gamma
    Γ1 = args.gamma
    temp = args.temp

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
    ###############

    ### set up simulation ###
    duration = args.duration
    dexp = args.dexp

    nt = args.nt
    # ϕW_array = np.zeros((nt + 1, len(ϕW)))
    # ϕB_array = np.zeros((nt + 1, len(ϕB)))
    # ϕW_array[0] = ϕW.value
    # ϕB_array[0] = ϕB.value
    t_save = np.linspace(0, duration, nt+1)

    elapsed = 0
    flag = 1
    dt = args.dt
    ###############

    ### start simulation ###
    # if args.timestepper == "exp":
    #     while elapsed < duration:
    #         dt = min(50, np.exp(dexp))
    #         eq.solve(dt=dt)
    #         elapsed += dt
    #         dexp += 0.01
    #         print(f"t={elapsed:0.2f} / {duration}", end="\r")
    #         if elapsed >= t_save[flag]:
    #             gname = f"n{flag:06d}"
    #             datadict = {"t": t_save[flag],
    #                         "phiW": ϕW.value,
    #                         "phiB": ϕB.value}
    #             dump(datafile, gname, datadict)
    #             flag += 1
    # elif args.timestepper == "linear":
    while elapsed < duration:
        eq.solve(dt=dt)
        elapsed += dt
        if elapsed >= t_save[flag]:
            print(f"t={elapsed:0.2f} / {duration}", end="\r")
            mseW = np.mean((ϕWf_cell - ϕW.value)**2)
            mseB = np.mean((ϕBf_cell - ϕB.value)**2)
            gname = f"n{flag:06d}"
            datadict = {"t": t_save[flag],
                        "phiW": ϕW.value,
                        "phiB": ϕB.value,
                        "mseW": mseW,
                        "mseB": mseB}
            dump(datafile, gname, datadict)
            flag += 1
    ###############

    ### final output ###
    mseW = np.mean((ϕWf_cell - ϕW.value)**2)
    mseB = np.mean((ϕBf_cell - ϕB.value)**2)

    print(f"Final mean-square error:")
    print(f"White: {mseW:0.4f}")
    print(f"Black: {mseB:0.4f}")
    ###############

    # with h5py.File(datafile, "a") as d:
    #     d.create_dataset("t_array", data=t_save)
    #     d.create_dataset("phiW_array", data=ϕW_array)
    #     d.create_dataset("phiB_array", data=ϕB_array)
    #     d.create_dataset("cell_centers", data=cell_points)
    #     d.create_dataset("phiW_2020", data=ϕWf_cell)
    #     d.create_dataset("phiB_2020", data=ϕBf_cell)
    #     d.create_dataset("mse_white", data=mseW)
    #     d.create_dataset("mse_black", data=mseB)
    #     d.create_dataset("corr_length", data=ξ)
    #     d.create_dataset("corr_length_var", data=ξvar)