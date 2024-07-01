import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
import json
import h5py
from scipy import ndimage, interpolate
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import argparse
from sociohydro2DInferer import *
from infer_utils import *


def make_coef_plot(coef_df, pearson_coefs, savename):
    pWs, pBs = pearson_coefs
    nfeat = len(coef_df.name.unique())

    fig, ax = plt.subplots(dpi=144, figsize=(4,2))

    for demo_code, color, offset in zip(["W", "B"], ["C0", "C3"], [-0.2, +0.2]):
        demo_df = coef_df.loc[coef_df["demo"]==demo_code][["val", "name"]]
        xvals = pd.Categorical(demo_df.name).codes+offset
        
        ax.plot(xvals, demo_df.val, ".", color=color, alpha=0.1)
        ax.errorbar(np.arange(7)+offset,
                    demo_df.groupby("name").mean().reset_index().val.values,
                    yerr=demo_df.groupby("name").std().reset_index().val.values,
                    fmt="o", color=color, capsize=5)


    ax.set(xticks=range(nfeat),
           xticklabels=pd.Categorical(coef_df.name).categories.values,
           ylabel="values", xlabel="coefficients")
    for n in range(1, nfeat):
        ax.axvline(n - 0.5, color="0.7")

    ax.axhline(0, color="0.95", zorder=-1)


    axp = ax.inset_axes([1.3, 0, 0.2, 1])
    axp.plot([0] * len(pWs), pWs, "C0.", alpha=0.1)
    axp.errorbar([0], np.mean(pWs), yerr=np.std(pWs),
                fmt="o", color="C0", capsize=5)
    axp.plot([1] * len(pBs), pBs, "C3.", alpha=0.1)
    axp.errorbar([1], np.mean(pBs), yerr=np.std(pBs),
                fmt="o", color="C3", capsize=5)
    axp.set(xlim=[-0.5, 1.5], xticks=[],
            ylim=[-0.5, 0.5], yticks=[-0.5, 0, 0.5],
            ylabel="pearson coeff")
    
    fig.savefig(savename, bbox_inches="tight")


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-datafiles', nargs="+", type=str,
                        help='path to hdf5 files with gridded data')
    parser.add_argument('-sigma', type=float, default=2.0,
                        help='size of gaussian to use when smoothing data')
    parser.add_argument('-nt', type=int, default=101,
                        help='number of time points to use for temporal interpolation')
    parser.add_argument('-t_dim', type=int, default=0,
                        help='time dimension of input hdf5 files')
    parser.add_argument('-train_pct', type=float, default=0.75,
                        help='fraction of data to use for fitting, ∈ [0.0, 1.0)')
    parser.add_argument('-ntrials', type=int, default=10,
                        help='number of different train/test splits to train on')
    parser.add_argument('-savefolder', type=str, default='.',
                        help='where to save output')

    args = parser.parse_args()

    ### set up save environment ###
    paramfile = os.path.join(args.savefolder, "inference_params.json")
    if not os.path.exists(args.savefolder):
        os.makedirs(args.savefolder)
    else:
        files = glob(os.path.join(args.savefolder, "*"))
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

    feat_names = [r"$T$", r"$k_{ii}$", r"$k_{ij}$", r"$\Gamma$",
              r"$\nu_{iii}$", r"$\nu_{iij}$", r"$\nu_{ijj}$"]

    ϕWs = []
    ϕBs = []
    xs = []
    ys = []
    ts = []
    ts_interp = []

    for datafile in args.datafiles:
        # get time array
        with h5py.File(datafile, "r") as d:
            t = np.array([int(k) for k in list(d.keys())])
        t_interp = np.linspace(t[0], t[-1], args.nt)

        ϕW = []
        ϕB = []
        for year in t:
            w, b, xx, yy = get_data(datafile, year=year, region="all")
            ϕW.append(w)
            ϕB.append(b)

        # white demographic data
        ϕW = np.asarray(ϕW)
        ϕW_smooth = ndimage.gaussian_filter(np.nan_to_num(ϕW),
                                            args.sigma,
                                            axes=[1, 2])
        cubicW = interpolate.interp1d(t, ϕW_smooth, axis=0, kind="cubic")
        ϕW_smooth_interp = cubicW(t_interp)
        ϕW_smooth_interp[:, np.any(np.isnan(ϕW), axis=0)] = np.nan
        ϕWs.append(ϕW_smooth_interp)

        # black demographic data
        ϕB = np.asarray(ϕB)
        ϕB_smooth = ndimage.gaussian_filter(np.nan_to_num(ϕB),
                                            args.sigma,
                                            axes=[1, 2])
        cubicB = interpolate.interp1d(t, ϕB_smooth, axis=0, kind="cubic")
        ϕB_smooth_interp = cubicB(t_interp)
        ϕB_smooth_interp[:, np.any(np.isnan(ϕB), axis=0)] = np.nan
        ϕBs.append(ϕB_smooth_interp)

        # x position data
        xx /= 1000
        xs.append(np.unique(xx))

        # y position data
        yy /= 1000
        ys.append(np.unique(yy))

    
        # t data
        ts.append(t)
        ts_interp.append(t_interp)

    inferer = SociohydroInfer2D(ϕWs, ϕBs,
                                xs, ys, ts_interp,
                                t_dim=args.t_dim)
    
    names = [r"$T$", r"$k_{ii}$", r"$k_{ij}$", r"$\Gamma$",
             r"$\nu_{iii}$", r"$\nu_{iij}$", r"$\nu_{ijj}$"]
    coeffs = np.array([])
    coeff_names = np.array([])
    coeff_group = np.array([])
    # coeff_trial = np.array([])
    pWs = []
    pBs = []

    ntrials = 30
    train_pct = 0.9
    for trial in tqdm(range(ntrials)):
        fitW, fitB, _, _, _, _, pW, pB = inferer.fit(train_pct)
        coeffs = np.append(coeffs, list(fitW.coef_))
        coeffs = np.append(coeffs, list(fitB.coef_))
        coeff_names = np.append(coeff_names, names * 2)
        coeff_group = np.append(coeff_group, ["W"] * len(fitW.coef_))
        coeff_group = np.append(coeff_group, ["B"] * len(fitW.coef_))
        # coeff_group = np.append(coeff_trial, [trial] * len(names) * 2)
        pWs.append(pW)
        pBs.append(pB)
    
    coef_df = pd.DataFrame({
        "val": coeffs,
        "name": coeff_names,
        "demo": coeff_group,
        "trial": np.repeat(np.arange(ntrials), len(names)*2)
    })

    pearson_df = pd.DataFrame({
        "pearsonW": pWs,
        "pearsonB": pBs,
        "trial": np.arange(ntrials)
    })

    today = datetime.today().strftime("%Y-%m-%d")

    coef_file = os.path.join(args.savefolder, today + "_coefs_trainPct{args.train_pct}_sigma{args.sigma}")
    coef_df.to_csv(coef_file + ".csv", index=False)
    coef_df.groupby(["name", "demo"]).mean().reset_index().to_csv(coef_file + "_mean.csv",
                                                                  index=False)
    
    pearson_file = os.path.join(args.savefolder, today + "_pearson_trainPct{args.train_pct}_sigma{args.sigma}")
    pearson_df.to_csv(pearson_file + ".csv", index=False)

    plot_file = os.path.join(args.savefolder, today + "_coefs_trainPct{args.train_pct}_sigma{args.sigma}")
    make_coef_plot(coef_df, [pWs, pBs], plot_file + ".pdf")