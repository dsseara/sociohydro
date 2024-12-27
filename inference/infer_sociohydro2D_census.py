import numpy as np
import os
from glob import glob
import json
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import argparse
from sociohydroInferer import *
from infer_utils import *

feat_names = [
    r"$T$",
    r"$k_{ii}$",
    r"$k_{ij}$",
    r"$\Gamma$",
    r"$\nu_{iii}$",
    r"$\nu_{iij}$",
    r"$\nu_{ijj}$"
]

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-datafolder', type=str,
                        help='path to folder containing hdf5 files with gridded data')
    parser.add_argument("-state_county", nargs="+", type=str,
                        help="name of state and county to use for inference")
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
    parser.add_argument("-use_fill_frac", type=str2bool, default=False,
                        help="whether to use fill fraction as a feature")
    parser.add_argument("-use_max_scaling", type=str2bool, default=True,
                        help="whether to use max scaling for features")
    parser.add_argument('-savefolder', type=str, default='.',
                        help='where to save output')
    parser.add_argument('-savename', type=str, default='data',
                        help='name of file to save output')

    args = parser.parse_args()

    if not np.logical_xor(args.use_fill_frac, args.use_max_scaling):
        raise ValueError(f"either use_fill_frac ({args.use_fill_frac}) or use_max_scaling ({args.use_max_scaling}) must be true")

    ### set up save environment ###
    savename = os.path.join(args.savefolder, args.savename)
    paramfile = f"{savename}_inferenceParams.json"
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

    phiWs = []
    phiBs = []
    xs = []
    ys = []
    ts = []
    masks = []
    housings = []

    for sc in args.state_county:
        datafile = os.path.join(args.datafolder, f"{sc}.hdf5")
        wb, x, y, t, housing, mask = get_data(datafile,
                                              sigma=args.sigma,
                                              use_fill_frac=args.use_fill_frac,
                                              use_max_scaling=args.use_max_scaling)
        
        t_interp = np.linspace(t.min(), t.max(), args.nt)
        
        phiWs.append(wb(t_interp)[:, 0])
        phiBs.append(wb(t_interp)[:, 1])
        xs.append(np.unique(x))
        ys.append(np.unique(y))
        ts.append(t_interp)
        masks.append(mask)
        housings.append(housing)

    inferer = SociohydroInfer2D(phiWs, phiBs,
                                xs, ys, ts,
                                t_dim = 0,
                                diff_method="savgol",
                                periodic=False,
                                masks=masks)
    
    
    coeff_list = np.array([])
    coeff_names = np.array([])
    coeff_group = np.array([])
    # coeff_trial = np.array([])
    mseW = []
    mseB = []

    ntrials = 30
    train_pct = 0.8
    for trial in tqdm(range(ntrials)):
        coeffs, ddts, feats, mses, _ = inferer.fit(train_pct)
        coeff_list = np.append(coeff_list, list(coeffs[0]))
        coeff_list = np.append(coeff_list, list(coeffs[1]))
        coeff_names = np.append(coeff_names, feat_names*2)
        coeff_group = np.append(coeff_group, ["W"]*len(feat_names))
        coeff_group = np.append(coeff_group, ["B"]*len(feat_names))
        mseW.append(mses[0])
        mseB.append(mses[1])
    
    coef_df = pd.DataFrame({
        "val": coeff_list,
        "name": coeff_names,
        "demo": coeff_group,
        "trial": np.repeat(np.arange(ntrials), len(feat_names)*2)
    })
    coef_df_mean = coef_df.groupby(["name", "demo"]).mean().reset_index()

    mse_df = pd.DataFrame({
    "demo": ["W"] * ntrials + ["B"] * ntrials,
    "coef": mseW + mseB
    })

    growth_rates = inferer.calc_growthRates()
    growth_df = pd.DataFrame({
        "demo": ["W", "B"],
        "growth_rate": growth_rates
    })

    # save outputs
    today = datetime.today().strftime("%Y-%m-%d")
    savename = os.path.join(args.savefolder, args.savename)
    
    coef_df.to_csv(f"{savename}_coefs.csv", index=False)
    coef_df_mean.to_csv(f"{savename}_coefs_mean.csv",index=False)
    mse_df.to_csv(f"{savename}_mse.csv", index=False)
    growth_df.to_csv(f"{savename}_growth.csv", index=False)

    plot_file = os.path.join(args.savefolder, today + f"_coefs")
    plot_coeffs(coef_df, mse_df, savename=savename)
    plot_regression(coeffs, ddts, feats, savename=savename)