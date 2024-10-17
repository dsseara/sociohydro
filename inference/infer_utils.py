from scipy import interpolate, signal, ndimage
import numpy as np
import h5py
import string
import matplotlib.pyplot as plt
import pandas as pd
import fipy as fp

def spline_deriv(x, y, polyorder=3, order=1, axis=0, smoothing=0):
    tck = interpolate.splrep(x, y, k=polyorder, s=smoothing)
    return interpolate.splev(x, tck, der=order)


def spectral_deriv(x, y, order=1, axis=0):
    dx = x[1] - x[0]
    q = 2 * np.pi * np.fft.fftfreq(len(x), dx)
    deriv = np.fft.ifft((1j * q)**order * np.fft.fft(y, axis=axis)).real
    return deriv


def savgol_deriv(x, y, window_length=3, polyorder=3,
                 order=1, axis=0, periodic=False,
                 smooth=False, smooth_polyorder=3):
    # if np.any(np.isnan(y)):
    #     original_nans = np.isnan(y)
    #     y = np.nan_to_num(y)
    nanmask = np.isnan(y)

    if periodic:
        mode="wrap"
    else:
        mode="nearest"
    
    dx = x[1] - x[0]

    deriv = signal.savgol_filter(np.ma.array(y, mask=nanmask),
                                 window_length, polyorder,
                                 deriv=order, axis=axis, mode=mode,
                                 delta=dx)
    if smooth:
        deriv = signal.savgol_filter(np.ma.array(deriv, mask=nanmask),
                                     window_length, smooth_polyorder,
                                     deriv=0, axis=axis, mode=mode)

    # deriv[original_nans] = np.nan
    return deriv


def gauss_deriv(x, y, sigma=1, order=1, axis=0, periodic=False):
    if periodic:
        mode = "wrap"
    else:
        mode = "reflect"

    return ndimage.gaussian_filter(y, sigma=sigma, order=order, axes=(axis,), mode=mode)


# 4th order finite difference derivatives
def d0(Nx, dx):
    return np.eye(Nx)

def d1(Nx, dx):
    grad_utri = (
        +2/3  * np.eye(Nx, k=1) +
        -1/12 * np.eye(Nx, k=2) +
        +1/12 * np.eye(Nx, k=Nx-2) +
        -2/3  * np.eye(Nx, k=Nx-1)  
    )
    grad = grad_utri - grad_utri.T
    return grad / dx

def d2(Nx, dx):
    lap_utri = (
        +4/3  *  np.eye(Nx, k=Nx-1) + 
        -1/12 * np.eye(Nx, k=Nx-2) + 
        -1/12 * np.eye(Nx, k=2) + 
        +4/3  *  np.eye(Nx, k=1)
    )

    lap_diag = np.diag(-5/2 * np.ones(Nx))

    lap = lap_diag + lap_utri + lap_utri.T
    return lap / dx**2

def d3(Nx, dx):
    gradlap_utri = (
        -13/8 * np.eye(Nx, k=1)    +
        1     * np.eye(Nx, k=2)    +
        -1/8  * np.eye(Nx, k=3)    +
        +1/8  * np.eye(Nx, k=Nx-3) +
        -1    * np.eye(Nx, k=Nx-2) +
        13/8  * np.eye(Nx, k=Nx-1)
    )
    gradlap = gradlap_utri - gradlap_utri.T
    return gradlap/dx**3

def d4(Nx, dx):
    laplap_utri = (
        -13/2 * np.eye(Nx, k=1) + 
        2     * np.eye(Nx, k=2) + 
        -1/6  * np.eye(Nx, k=3) + 
        -1/6  * np.eye(Nx, k=Nx-3) + 
        2     * np.eye(Nx, k=Nx-2) + 
        -13/2 * np.eye(Nx, k=Nx-1)
    )

    laplap_diag = np.diag(28/3 * np.ones(Nx))
    
    laplap = laplap_diag + laplap_utri + laplap_utri.T
    return laplap / dx**4

def fd_deriv(x, y, order=1, axis=0):
    ndims = y.ndim
    # index input array
    input_inds = list(string.ascii_lowercase[:ndims])
    # index derivative matrix, assume input array never gets to z
    deriv_inds = ['z', input_inds[axis]]
    # index output array, replacing axis over which derivative is taken with z
    output_inds = list(string.ascii_lowercase[:ndims])
    output_inds[axis] = 'z'
    # create the string for einstein summation
    # e.g. "zb,abcd->azcd"
    einstr = (
        ''.join(deriv_inds) + "," +
        ''.join(input_inds) + "->" +
        ''.join(output_inds)
    )
    Nx = len(x)
    dx = x[1] - x[0]
    derivs = [d0, d1, d2, d3, d4]

    deriv = np.einsum(einstr, derivs[order](Nx, dx),  y)
    return deriv


def get_capacity(file, region="all", method="wb"):
    with h5py.File(file, "r") as d:
        x_grid = d[list(d.keys())[0]]["x_grid"][()]
        capacity = np.zeros(x_grid.shape)

        regions = ["all", "county"]
        if region not in regions:
            raise ValueError("region is either all or county")
        else:
            if region == "all":
                region_str = "masked"
            elif region == "county":
                region_str = "county"
        
        methods = ["wb", "total"]
        if method not in methods:
            raise ValueError("method is either wb or total")

        for key in d.keys():
            if method.lower() == "wb":
                wb = (d[key]["white_grid_" + region_str][()] +
                      d[key]["black_grid_" + region_str][()])
                capacity = np.fmax(capacity, wb)
            elif method.lower() == "total":
                tot = d[key]["total_grid_" + region_str][()]
                capacity = np.fmax(capacity, tot)

        return capacity


def get_data(file, year=1990, region="all", norm=True, method="wb"):
    ykey = str(year)
    
    if (region == "all") | (region == "masked"):
        region_str = "masked"
    elif region == "county":
        region_str = "county"

    with h5py.File(file, "r") as d:
        x_grid = d[ykey]["x_grid"][()]
        y_grid = d[ykey]["y_grid"][()]
        white = d[ykey]["white_grid_" + region_str][()]
        black = d[ykey]["black_grid_" + region_str][()]

        if norm:
            capacity = get_capacity(file, region=region, method=method)
            ϕW = white / (1.1 * capacity)
            ϕB = black / (1.1 * capacity)
        else:
            ϕW = white
            ϕB = black

    return ϕW, ϕB, x_grid, y_grid


def plot_coeffs(coef_df, pearson_df,
                savename="./",
                coefs_true=None):
    fig, ax = plt.subplots(dpi=144, figsize=(4,2))
    demo_codes = coef_df["demo"].unique()
    nfeat = len(coef_df.name.unique())

    for demo_code, color, offset in zip(demo_codes,
                                        ["C0", "C3"],
                                        [-0.2, +0.2]):
        demo_df = coef_df.loc[coef_df["demo"]==demo_code][["val", "name"]]
        xvals = pd.Categorical(demo_df.name).codes+offset
        
        ax.plot(xvals, demo_df.val, ".", color=color, alpha=0.1)
        ax.errorbar(np.arange(7)+offset,
                    demo_df.groupby("name").mean().reset_index().val.values,
                    yerr=demo_df.groupby("name").std().reset_index().val.values,
                    fmt="o", color=color, capsize=5)


    ax.set(xticks=range(7), xticklabels=pd.Categorical(coef_df.name).categories.values,
        ylabel="values", xlabel="coefficients");
    for n in range(1, len(coef_df.name.unique())):
        ax.axvline(n - 0.5, color="0.7")

    ax.axhline(0, color="0.95", zorder=-1)
    
    if coefs_true is not None:
        ax.plot(np.arange(nfeat) - 0.2, coefs_true[0],
                "s", mfc="white", mec="C0", zorder=-1)
        ax.plot(np.arange(nfeat) + 0.2, coefs_true[1],
                "s", mfc="white", mec="C3", zorder=-1)

    pW = pearson_df.loc[pearson_df.demo == "W"]
    pB = pearson_df.loc[pearson_df.demo == "B"]

    axp = ax.inset_axes([1.3, 0, 0.2, 1])
    axp.plot([0] * len(pW), pW.coef, "C0.", alpha=0.1)
    axp.errorbar([0], pW.coef.mean(),
                 yerr=pW.coef.std(),
                 fmt="o", color="C0", capsize=5)
    axp.plot([1] * len(pB), pB.coef, "C3.", alpha=0.1)
    axp.errorbar([1], pB.coef.mean(),
                 yerr=pB.coef.std(),
                 fmt="o", color="C3", capsize=5)
    axp.set(xlim=[-0.5, 1.5],
            xticks=[],
            ylim=[-1, 1],
            yticks=[-1, 0, 1],
            ylabel="pearson coeff")
    
    fig.savefig(savename + "_inferredCoefs.pdf", bbox_inches="tight")


def plot_regression(fit1, fit2, d1dt, d2dt, feat1, feat2,
             savename="./"):
    fig, ax = plt.subplots(dpi=144, figsize=(3, 3))
    ax.plot(d1dt["test"],
            fit1.predict(feat1["test"]),
            "C0.", alpha=0.1, ms=1)
    ax.plot(d2dt["test"],
            fit2.predict(feat2["test"]),
            "C3.", alpha=0.1, ms=1)

    ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0), useMathText=True)
    ax.set(
        xlabel=r"$\dot{\phi}_i$ true",
        ylabel=r"$\dot{\phi}_i$ prediction",
        xlim=[1.1 * np.min([d1dt["test"].min(), d2dt["test"].min()]),
            1.1 * np.max([d1dt["test"].max(), d2dt["test"].max()])],
        ylim=[1.1 * np.min([d1dt["test"].min(), d2dt["test"].min()]),
            1.1 * np.max([d1dt["test"].max(), d2dt["test"].max()])]
    )

    ax.axline([0, 0], slope=1, ls="--", color="0.8", zorder=-1)

    ax.set_aspect(1)
    fig.savefig(savename + "_regressionFit.pdf", bbox_inches="tight")
    fig.savefig(savename + "_regressionFit.jpg", bbox_inches="tight")


def build_term_value(term):
    orig = np.copy(term.var.value)
    dt=0.001
    nsweeps = 10
    eq = fp.TransientTerm(var=term.var) == term
    for sweep in range(nsweeps):
        eq.sweep(dt=dt)
    time_deriv = (term.var.value - orig) / (dt)
    return fp.CellVariable(mesh=term.var.mesh, value=time_deriv)

# def build_term_value(term):
#     solve_var = fp.CellVariable(mesh=term.var.mesh)
#     eq = fp.ImplicitSourceTerm(var=solve_var) == term
#     eq.solve(solve_var)
    
#     return solve_var


def calc_fipyTerms(var1, var2):
    """
    calculate gradients of fipy CellVariables
    used in sociohydro model

    returns: sociohydro_grads
        sociohydro_grads = [
            [
                T1_term,
                k11_term,
                k12_term,
                ν111_term,
                ν112_term,
                ν122_term,
                Γ1_term
            ],
            [
                T2_term,
                k22_term,
                k21_term,
                ν222_term,
                ν212_term,
                ν211_term,
                Γ2_term
            ]
        ]
        where:
            Ti_term = div( φ0 grad(φi) - φi grad(φ0) )
            kij_term = div( φ0 φi grad(φj) )
            νijk_term = div( φ0 φi grad(φj φk))
            Γi_term = div( φ0 φi grad(lap(φi)) )
    """
    # save these for resetting values after running solver
    var1_val = np.copy(var1.value)
    var2_val = np.copy(var2.value)
    
    # scalar gradients
    var1_lap   = build_term_value(fp.DiffusionTerm(coeff=1, var=var1))
    var2_lap   = build_term_value(fp.DiffusionTerm(coeff=1, var=var2))
    var1.setValue(var1_val)
    var2.setValue(var2_val)
    
    var0 = 1 - var1 - var2
    var0_lap = -(var1_lap + var2_lap)

    T1 = var0 * var1_lap - var1 * var0_lap
    # T1.name = "T1"

    k11_term  = fp.DiffusionTerm(coeff=var0 * var1, var=var1)
    k11 = build_term_value(k11_term)
    # k11 = (
    #     var0 * var1.grad.dot(var1.grad).value + 
    #     var1 * var0.grad.dot(var1.grad).value +
    #     var0 * var1 * var1_lap
    # )
    k11.name = "k11"
    var1.setValue(var1_val)
    var2.setValue(var2_val)

    k12_term  = fp.DiffusionTerm(coeff=var0 * var1, var=var2)
    k12 = build_term_value(k12_term)
    k12.name = "k12"
    var1.setValue(var1_val)
    var2.setValue(var2_val)
    
    ν111_term = (
        fp.DiffusionTerm(coeff=2 * var0 * var1 * var1, var=var1)
    )
    ν111 = build_term_value(ν111_term)
    ν111.name = "ν111"
    var1.setValue(var1_val)
    var2.setValue(var2_val)
    
    ν112_term = (
        fp.DiffusionTerm(coeff=var0 * var1 * var2, var=var1) + 
        fp.DiffusionTerm(coeff=var0 * var1 * var1, var=var2)
    )
    # ν112_term = (
    #     fp.DiffusionTerm(coeff=var0 * var1 * var2, var=var1 * var2)
    # )
    ν112 = build_term_value(ν112_term)
    ν112.name = "ν112"
    var1.setValue(var1_val)
    var2.setValue(var2_val)
    
    ν122_term = (
        fp.DiffusionTerm(coeff=2 * var0 * var1 * var2, var=var2)
    )
    ν122 = build_term_value(ν122_term)
    ν122.name = "ν122"
    var1.setValue(var1_val)
    var2.setValue(var2_val)
    
    Γ1_term   = fp.DiffusionTerm(coeff=(var0 * var1, -1), var=var1)
    Γ1 = build_term_value(Γ1_term)
    Γ1.name = "Γ1"
    var1.setValue(var1_val)
    var2.setValue(var2_val)

    
    # k11  = (var0.faceValue * var1.faceValue * var1_grad.faceValue).divergence
    # k12  = (var0.faceValue * var1.faceValue * var2_grad.faceValue).divergence
    # ν111 = (var0.faceValue * var1.faceValue * (var1 * var1).grad.faceValue).divergence 
    # ν112 = (var0.faceValue * var1.faceValue * (var1 * var2).grad.faceValue).divergence 
    # ν122 = (var0.faceValue * var1.faceValue * (var2 * var2).grad.faceValue).divergence 
    # Γ1   = (var0.faceValue * var1.faceValue * var1_gradlap.faceValue).divergence 

    print("var2 features")
    T2 = var0 * var2_lap - var2 * var0_lap
    
    k21_term  = fp.DiffusionTerm(coeff=var0 * var2, var=var1)
    k21 = build_term_value(k21_term)
    var1.setValue(var1_val)
    var2.setValue(var2_val)
    
    k22_term  = fp.DiffusionTerm(coeff=var0 * var2, var=var2)
    k22 = build_term_value(k22_term)
    var1.setValue(var1_val)
    var2.setValue(var2_val)
    
    ν211_term = (
        fp.DiffusionTerm(coeff=2 * var0 * var2 * var1, var=var1)
    )
    ν211 = build_term_value(ν211_term)
    var1.setValue(var1_val)
    var2.setValue(var2_val)
    
    ν212_term = (
        fp.DiffusionTerm(coeff=var0 * var2 * var2, var=var1) + 
        fp.DiffusionTerm(coeff=var0 * var2 * var1, var=var2)
    )
    # ν212_term = (
    #     fp.DiffusionTerm(coeff=var0 * var2 * var2, var=var1 * var2)
    # )
    ν212 = build_term_value(ν212_term)
    var1.setValue(var1_val)
    var2.setValue(var2_val)
    
    ν222_term = (
        fp.DiffusionTerm(coeff=2 * var0 * var2 * var2, var=var2)
    )
    ν222 = build_term_value(ν222_term)
    var1.setValue(var1_val)
    var2.setValue(var2_val)
    
    Γ2_term   = fp.DiffusionTerm(coeff=(var0 * var2, -1), var=var2)
    Γ2 = build_term_value(Γ2_term)
    var1.setValue(var1_val)
    var2.setValue(var2_val)

    
    # k21  = (var0.faceValue * var2.faceValue * var1_grad.faceValue).divergence
    # k22  = (var0.faceValue * var2.faceValue * var2_grad.faceValue).divergence
    # ν211 = (var0.faceValue * var2.faceValue * (var1 * var1).grad.faceValue).divergence 
    # ν212 = (var0.faceValue * var2.faceValue * (var1 * var2).grad.faceValue).divergence 
    # ν222 = (var0.faceValue * var2.faceValue * (var2 * var2).grad.faceValue).divergence 
    # Γ2   = (var0.faceValue * var2.faceValue * var2_gradlap.faceValue).divergence

    sociohydro_grads = [
        [T1.value, k11.value, k12.value, ν111.value, ν112.value, ν122.value, -Γ1.value],
        [T2.value, k22.value, k21.value, ν222.value, ν212.value, ν211.value, -Γ2.value]
    ]

    return sociohydro_grads
