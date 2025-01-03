import os
import json
from glob import glob
import fipy as fp
from fipy.tools.dump import write, read
import numpy as np
import argparse
import h5py
from scipy import interpolate, ndimage
from tobler.area_weighted import area_interpolate

from fvm_utils import *

def str_to_bool(s):
    if s.lower() in ["true", "t", "yes", "y", "1"]:
        return True
    elif s.lower() in ["false", "f", "no", "n", "0"]:
        return False
    else:
        raise ValueError("String not recognized as boolean")

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
    parser.add_argument("-timestepper", type=str, default="linear",
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
    parser.add_argument("-growthW", type=float, nargs=6, default=[0, 0, 0, 0, 0, 0],
                        help="growth terms for White population")
    
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
    parser.add_argument("-growthB", type=float, nargs=6, default=[0, 0, 0, 0, 0, 0],
                        help="growth terms for Black population")
    
    # parameters for simulation
    parser.add_argument("-buffer", type=float, default=1.0,
                        help="buffer for boundary simplification")
    parser.add_argument("-simplify", type=float, default=1.0,
                        help="simplify parameter for boundary simplification")
    parser.add_argument("-cellsize", type=float, default=3,
                        help="typical size of cells in mesh")
    parser.add_argument("-use_fill_frac", type=str_to_bool, default=True,
                        help="use local scale for population normalization")
    parser.add_argument("-use_max_scaling", type=str_to_bool, default=False,
                        help="use global scale for population normalization")
    
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
    if not np.logical_xor(args.use_fill_frac, args.use_max_scaling):
        raise ValueError("only one of use_fill_frac and use_max_scaling can be true")
    
    # load data
    wb, x, y, t, housing, mask = get_data(args.inputfile,
                                          spatial_scale=1000,
                                          sigma=args.sigma,
                                          use_fill_frac=args.use_fill_frac,
                                          use_max_scaling=args.use_max_scaling)
    # initial condition 
    wb0 = wb(1990)
    # final condition
    wbf = wb(2020)

    # create meshes for interpolation
    crs = "ESRI:102003"  # all cases fall into this CRS

    # create square mesh
    dx = x[1, 1] - x[0, 0]
    dy = y[1, 1] - y[0, 0]
    ny, nx = x.shape
    grid = fp.Grid2D(dx=dx, dy=dy, nx=nx, ny=ny) + ((x.min(), ), (y.min(), ))
    
    # put all data into geodataframes
    grid_gdf = mesh_to_gdf(grid, crs=crs)
    if args.use_max_scaling:
        grid_gdf["w0"]  = (wb0[0] * housing[mask].max()).ravel()
        grid_gdf["wf"]  = (wbf[0] * housing[mask].max()).ravel()
        grid_gdf["b0"]  = (wb0[1] * housing[mask].max()).ravel()
        grid_gdf["bf"]  = (wbf[1] * housing[mask].max()).ravel()
    elif args.use_fill_frac:
        grid_gdf["w0"]  = (wb0[0] * housing).ravel()
        grid_gdf["wf"]  = (wbf[0] * housing).ravel()
        grid_gdf["b0"]  = (wb0[1] * housing).ravel()
        grid_gdf["bf"]  = (wbf[1] * housing).ravel()
    
    grid_gdf["housing"] = housing.ravel()
    grid_gdf["mask"] = mask.astype(int).ravel()

    # create mesh
    mesh, simple_boundary, geo_file_contents = make_mesh(wb0, x, y, crs,
                                                         args.buffer,
                                                         args.simplify,
                                                         args.cellsize)
    
    mesh_gdf = mesh_to_gdf(mesh, crs=crs)
    interp = area_interpolate(grid_gdf, mesh_gdf,
                              extensive_variables=["w0", "wf", "b0", "bf", "housing"],
                              intensive_variables=["mask"])
    
    if args.use_fill_frac:
        interp["w0"] /= interp["housing"]
        interp["wf"] /= interp["housing"]
        interp["b0"] /= interp["housing"]
        interp["bf"] /= interp["housing"]
    elif args.use_max_scaling:
        interp["w0"] /= interp[interp["mask"] > 0]["housing"].max()
        interp["wf"] /= interp[interp["mask"] > 0]["housing"].max()
        interp["b0"] /= interp[interp["mask"] > 0]["housing"].max()
        interp["bf"] /= interp[interp["mask"] > 0]["housing"].max()


    # perform interpolation from grid points to cell centers
    grid_points = np.array([[x, y] for x, y in zip(x.ravel(),
                                                   y.ravel())])
    cell_points = np.array([[x, y] for x, y in zip(mesh.cellCenters.value[0],
                                                   mesh.cellCenters.value[1])])
    
    # ϕW0_cell = interpolate.griddata(grid_points,
    #                                 np.nan_to_num(wb0[0].ravel(), nan=1e-3),
    #                                 cell_points,
    #                                 fill_value=1e-3)
    # ϕWf_cell = interpolate.griddata(grid_points,
    #                                 np.nan_to_num(wbf[0].ravel(), nan=1e-3),
    #                                 cell_points,
    #                                 fill_value=1e-3)

    # ϕB0_cell = interpolate.griddata(grid_points,
    #                                 np.nan_to_num(wb0[1].ravel(), nan=1e-3),
    #                                 cell_points,
    #                                 fill_value=1e-3)
    # ϕBf_cell = interpolate.griddata(grid_points,
    #                                 np.nan_to_num(wbf[1].ravel(), nan=1e-3),
    #                                 cell_points,
    #                                 fill_value=1e-3)
    
    # housing_cell = interpolate.griddata(grid_points,
    #                                     np.nan_to_num(housing.ravel(), nan=1e-3),
    #                                     cell_points,
    #                                     fill_value=1e-3)
    
    # mask_cell = interpolate.griddata(grid_points,
    #                                  np.nan_to_num(mask.ravel(), nan=1e-3),
    #                                  cell_points,
    #                                  fill_value=1e-3)
    

    ###############

    ### save things common to all time-points ###
    group_name = "common"
    datadict = {
        "cell_centers": cell_points,
        "phiW_initial": interp["w0"].values,
        "phiB_initial": interp["b0"].values,
        "phiW_final": interp["wf"].values,
        "phiB_final": interp["bf"].values,
        "housing": interp["housing"].values,
        "mask": interp["mask"].values
    }
    dump(h5file, group_name, datadict)
    geojsonfile = os.path.join(args.savefolder, args.filename + ".geojson")
    interp.to_file(geojsonfile, driver="GeoJSON")
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
    

    ϕW = fp.CellVariable(name=r"phiW", mesh=mesh, hasOld=True)
    μW = fp.CellVariable(name=r"muW", mesh=mesh, hasOld=True)
    ϕB = fp.CellVariable(name=r"phiB", mesh=mesh, hasOld=True)
    μB = fp.CellVariable(name=r"muB", mesh=mesh, hasOld=True)

    # ϕW[:] = ϕW0_cell
    ϕW[:] = interp["w"]
    # ϕB[:] = ϕB0_cell
    ϕB[:] = interp["b"]
    ϕ0 = 1 - ϕW - ϕB

    mobilityW = ϕW * ϕ0
    πW = κWW * ϕW + κWB * ϕB + νWWW * ϕW * ϕW + νWWB * ϕW * ϕB + νWBB * ϕB * ϕB
    dπWdϕW = κWW + 2 * νWWW * ϕW + νWWB * ϕB
    μW_taylorExpand = -πW + TW * (np.log(ϕW) - np.log(ϕ0))
    dμWdϕW = -dπWdϕW + TW * (1 - ϕB) / (ϕW * ϕ0)
    SW = rW[0] + rW[1] * ϕW + rW[2] * ϕB + rW[3] * ϕW * ϕW + rW[4] * ϕW * ϕB + rW[5] * ϕB * ϕB
    
    mobilityB = ϕB * ϕ0
    πB = κBW * ϕW + κBB * ϕB + νBWW * ϕW * ϕW + νBWB * ϕW * ϕB + νBBB * ϕB * ϕB
    dπBdϕB = κBB + νBWB * ϕW + 2 * νBBB * ϕB
    μB_taylorExpand = -πB + TB * (np.log(ϕB) - np.log(ϕ0))
    dμBdϕB = -dπBdϕB + TB * (1 - ϕW) / (ϕB * ϕ0)
    SB = rB[0] + rB[1] * ϕW + rB[2] * ϕB + rB[3] * ϕW * ϕW + rB[4] * ϕW * ϕB + rB[5] * ϕB * ϕB
    
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
    snapshot = 1
    dt = args.dt
    ###############

    ### start simulation ###
    while elapsed < duration:
        for var in [ϕW, μW, ϕB, μB]:
            var.updateOld()
        res = 1e10
        while res > 1e-5:
            res = eq.sweep(dt=dt)
        # eq.solve(dt=dt)
        elapsed += dt
        if elapsed >= t_save[snapshot]:
            print(f"t={elapsed:0.2f} / {duration}", end="\r")
            mseW = np.mean((interp["wf"].values - ϕW.value)**2)
            mseB = np.mean((interp["bf"].values - ϕB.value)**2)
            # save to hdf5
            gname = f"n{snapshot:06d}"
            datadict = {"t": t_save[snapshot],
                        "phiW": ϕW.value,
                        "phiB": ϕB.value,
                        "mseW": mseW,
                        "mseB": mseB}
            dump(h5file, gname, datadict)
            # save fipy
            fipyfile = f"n{snapshot:06d}.fipy"
            write([ϕW, ϕB, t_save[snapshot]], os.path.join(fipyfolder, fipyfile))
            snapshot += 1
    ###############

    ### final output ###
    mseW = np.mean((interp["wf"].values - ϕW.value)**2)
    mseB = np.mean((interp["bf"].values - ϕB.value)**2)

    print(f"Final mean-square error:")
    print(f"White: {mseW:0.4f}")
    print(f"Black: {mseB:0.4f}")
    ###############