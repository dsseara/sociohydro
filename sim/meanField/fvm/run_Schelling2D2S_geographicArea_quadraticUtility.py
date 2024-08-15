import os
import json
from glob import glob
import fipy as fp
from fipy.tools.dump import write, read
import numpy as np
import argparse
import h5py
from scipy import interpolate, ndimage

from fvm_utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-inputfile", type=str, required=True,
                        help="path to hdf5 file with interpolated data for initial condition")
    parser.add_argument("-sigma", type=float, default=1.0,
                        help="size of gaussian used to smooth data")
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

    # parameters for White population
    parser.add_argument("-tempW", type=float, default=0.1,
                        help="temperature White popluation")
    parser.add_argument("-gammaW", type=float, default=1.0,
                        help="strength of gradient penalties for White population")
    parser.add_argument("-kWW", type=float, default=1.0,
                        help="self-utility of White population")
    parser.add_argument("-kWB", type=float, default=0.0,
                        help="White cross-utility for Black population")
    parser.add_argument("-nuWWW", type=float, default=0.0,
                        help="strength of non-linearity in White utility")
    parser.add_argument("-nuWWB", type=float, default=0.0,
                        help="strength of non-linearity in White utility")
    parser.add_argument("-nuWBB", type=float, default=0.0,
                        help="strength of non-linearity in White utility")
    parser.add_argument("-growthW", type=float, default=0.0,
                        help="growth term for White population")
    
    # parameters for Black population
    parser.add_argument("-tempB", type=float, default=0.1,
                        help="temperature Black popluation")
    parser.add_argument("-gammaB", type=float, default=1.0,
                        help="strength of gradient penalties for Black population")
    parser.add_argument("-kBB", type=float, default=1.0,
                        help="self-utility of Black population")
    parser.add_argument("-kBW", type=float, default=0.0,
                        help="Black cross-utility for White population")
    parser.add_argument("-nuBBB", type=float, default=0.0,
                        help="strength of non-linearity in Black utility")
    parser.add_argument("-nuBWB", type=float, default=0.0,
                        help="strength of non-linearity in Black utility")
    parser.add_argument("-nuBWW", type=float, default=0.0,
                        help="strength of non-linearity in Black utility")
    parser.add_argument("-growthB", type=float, default=0.0,
                        help="growth term for Black population")
    
    # parameters for simulation
    parser.add_argument("-capacityType", type=str, default="local",
                        choices=["uniform", "local"],
                        help="how to calculate the carrying capacities")
    parser.add_argument("-buffer", type=float, default=1.0,
                        help="buffer for boundary simplification")
    parser.add_argument("-simplify", type=float, default=1.0,
                        help="simplify parameter for boundary simplification")
    parser.add_argument("-cellsize", type=float, default=3,
                        help="typical size of cells in mesh")
    # parameters for saving output
    parser.add_argument("-savefolder", type=str, default=".",
                        help="path to folder to save output")
    parser.add_argument("-filename", type=str, default="data",
                        help="name of file to save output")
    args = parser.parse_args()

    ### set up save environment ###
    h5file = os.path.join(args.savefolder, args.filename + ".hdf5")
    paramfile = os.path.join(args.savefolder, args.filename + "_params.json")
    fipyfolder = os.path.join(args.savefolder, "fipy_output")
    if not os.path.exists(args.savefolder):
        # ensure savefolder is made
        os.makedirs(args.savefolder)
    else:
        # delete files in savefolder
        files = glob(os.path.join(args.savefolder, args.filename + "*"))
        for file in files:
            try:
                os.remove(file)
            except Exception as e:
                print(f"could not delete {file}. Reason: {e}")
        if not os.path.exists(fipyfolder):
            # ensure fipyfolder is made
            os.makedirs(fipyfolder)
        else:
            # delete files in fipyfolder
            fipyfiles = glob(os.path.join(fipyfolder, "*"))
            for fipyfile in fipyfiles:
                try:
                    os.remove(fipyfile)
                except Exception as e:
                    print(f"could not delete {fipyfile}. Reason: {e}")
    ###############

    ### save params ###
    with open(paramfile, "w") as p:
        p.write(json.dumps(vars(args), indent=4))
    ###############

    ### load and set-up data ###
    # initial condition
    ϕW0, ϕB0, x, y = get_data(args.inputfile,
                              year=1990,
                              region="masked",
                              capacity_method=args.capacityType)
    Wnans = np.isnan(ϕW0)
    Bnans = np.isnan(ϕB0)
    ϕW0 = ndimage.gaussian_filter(np.nan_to_num(ϕW0), args.sigma)
    ϕW0[Wnans] = np.nan
    ϕB0 = ndimage.gaussian_filter(np.nan_to_num(ϕB0), args.sigma)
    ϕB0[Bnans] = np.nan
    
    # measure in units of kilometers
    x /= 1000.0
    y /= 1000.0

    # final condition
    ϕWf, ϕBf, _, _ = get_data(args.inputfile,
                              year=2020,
                              region="masked",
                              capacity_method=args.capacityType)
    Wnans = np.isnan(ϕWf)
    Bnans = np.isnan(ϕBf)
    ϕWf = ndimage.gaussian_filter(np.nan_to_num(ϕWf), args.sigma)
    ϕWf[Wnans] = np.nan
    ϕBf = ndimage.gaussian_filter(np.nan_to_num(ϕBf), args.sigma)
    ϕBf[Bnans] = np.nan
    
    # get correlation length of white population
    ξ, ξvar = get_corrLength(args.inputfile, region="masked",
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
                                    fill_value=1e-3)
    ϕWf_cell = interpolate.griddata(grid_points,
                                    np.nan_to_num(ϕWf.ravel(), nan=1e-3),
                                    cell_points,
                                    fill_value=1e-3)

    ϕB0_cell = interpolate.griddata(grid_points,
                                    np.nan_to_num(ϕB0.ravel(), nan=1e-3),
                                    cell_points,
                                    fill_value=1e-3)
    ϕBf_cell = interpolate.griddata(grid_points,
                                    np.nan_to_num(ϕBf.ravel(), nan=1e-3),
                                    cell_points,
                                    fill_value=1e-3)
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
    dump(h5file, group_name, datadict)
    ###############

    ### build simulation ###
    # utility parameters
    TW   = args.tempW
    ΓW   = args.gammaW
    κWW  = args.kWW
    κWB  = args.kWB
    νWWW = args.nuWWW
    νWWB = args.nuWWB
    νWBB = args.nuWBB
    rW = args.growthW

    TB   = args.tempB
    ΓB   = args.gammaB
    κBB  = args.kBB
    κBW  = args.kBW
    νBBB = args.nuBBB
    νBWB = args.nuBWB
    νBWW = args.nuBWW
    rB = args.growthB
    

    ϕW = fp.CellVariable(name=r"$\phi_W$", mesh=mesh, hasOld=True)
    μW = fp.CellVariable(name=r"$\tilde{\mu}_W$", mesh=mesh, hasOld=True)
    ϕB = fp.CellVariable(name=r"$\phi_B$", mesh=mesh, hasOld=True)
    μB = fp.CellVariable(name=r"$\tilde{\mu}_B$", mesh=mesh, hasOld=True)

    ϕW[:] = ϕW0_cell
    ϕB[:] = ϕB0_cell
    ϕ0 = 1 - ϕW - ϕB

    mobilityW = ϕW * ϕ0
    πW = κWW * ϕW + κWB * ϕB + νWWW * ϕW * ϕW + νWWB * ϕW * ϕB + νWBB * ϕB * ϕB
    dπWdϕW = κWW + 2 * νWWW * ϕW + νWWB * ϕB
    μW_taylorExpand = -πW + TW * (np.log(ϕW) - np.log(ϕ0))
    dμWdϕW = -dπWdϕW + TW * (1 - ϕB) / (ϕW * ϕ0)
    SW = rW * ϕW
    
    mobilityB = ϕB * ϕ0
    πB = κBW * ϕW + κBB * ϕB + νBWW * ϕW * ϕW + νBWB * ϕW * ϕB + νBBB * ϕB * ϕB
    dπBdϕB = κBB + νBWB * ϕW + 2 * νBBB * ϕB
    μB_taylorExpand = -πB + TB * (np.log(ϕB) - np.log(ϕ0))
    dμBdϕB = -dπBdϕB + TB * (1 - ϕW) / (ϕB * ϕ0)
    SB = rB * ϕB
    
    eqW_1 = (fp.TransientTerm(var=ϕW) == fp.DiffusionTerm(coeff=mobilityW, var=μW) + SW)
    eqW_2 = (fp.ImplicitSourceTerm(coeff=1, var=μW)
             == fp.ImplicitSourceTerm(coeff=dμWdϕW, var=ϕW)
             - dμWdϕW * ϕW + μW_taylorExpand
             - fp.DiffusionTerm(coeff=ΓW, var=ϕW))
    
    eqB_1 = (fp.TransientTerm(var=ϕB) == fp.DiffusionTerm(coeff=mobilityB, var=μB) + SB)
    eqB_2 = (fp.ImplicitSourceTerm(coeff=1, var=μB)
             == fp.ImplicitSourceTerm(coeff=dμBdϕB, var=ϕB)
             - dμBdϕB * ϕB + μB_taylorExpand
             - fp.DiffusionTerm(coeff=ΓB, var=ϕB))

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
    #             dump(h5file, gname, datadict)
    #             flag += 1
    # elif args.timestepper == "linear":
    while elapsed < duration:
        for var in [ϕW, μW, ϕB, μB]:
            var.updateOld()
        res = 1e10
        while res > 1e-5:
            res = eq.sweep(dt=dt)
        # eq.solve(dt=dt)
        elapsed += dt
        if elapsed >= t_save[flag]:
            print(f"t={elapsed:0.2f} / {duration}", end="\r")
            mseW = np.mean((ϕWf_cell - ϕW.value)**2)
            mseB = np.mean((ϕBf_cell - ϕB.value)**2)
            # save to hdf5
            gname = f"n{flag:06d}"
            datadict = {"t": t_save[flag],
                        "phiW": ϕW.value,
                        "phiB": ϕB.value,
                        "mseW": mseW,
                        "mseB": mseB}
            dump(h5file, gname, datadict)
            # save fipy
            fipyfile = f"n{flag:06d}.fipy"
            write([ϕW, ϕB, t_save[flag]], os.path.join(fipyfolder, fipyfile))
            flag += 1
    ###############

    ### final output ###
    mseW = np.mean((ϕWf_cell - ϕW.value)**2)
    mseB = np.mean((ϕBf_cell - ϕB.value)**2)

    print(f"Final mean-square error:")
    print(f"White: {mseW:0.4f}")
    print(f"Black: {mseB:0.4f}")
    ###############

    # with h5py.File(h5file, "a") as d:
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