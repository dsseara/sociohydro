from scipy import interpolate, signal, ndimage
import numpy as np
import h5py

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
        mode="interp"
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


def get_data(file, year=1990, region="all"):
    ykey = str(year)
    with h5py.File(file, "r") as d:
        x_grid = d[ykey]["x_grid"][()]
        y_grid = d[ykey]["y_grid"][()]
        capacity = np.zeros(x_grid.shape)
        if region == "county":
            white = d[ykey]["white_grid_county"][()]
            black = d[ykey]["black_grid_county"][()]
            for key in d.keys():
                capacity = np.fmax(capacity, d[key]["white_grid_county"][:] + d[key]["black_grid_county"][:])
        elif region == "all":
            white = d[ykey]["white_grid_masked"][()]
            black = d[ykey]["black_grid_masked"][()]
            for key in d.keys():
                capacity = np.fmax(capacity, d[key]["white_grid_masked"][:] + d[key]["black_grid_masked"][:])

    ϕW = white / (1.1 * capacity)
    ϕB = black / (1.1 * capacity)

    return ϕW, ϕB, x_grid, y_grid