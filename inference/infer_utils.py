from scipy import interpolate, signal, ndimage
import numpy as np


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
    if periodic:    
        mode="wrap"
    else:
        mode="interp"
    dx = x[1] - x[0]
    deriv = signal.savgol_filter(y, window_length, polyorder,
                                 deriv=order, axis=axis, mode=mode,
                                 delta=dx)
    if smooth:
        deriv = signal.savgol_filter(deriv, window_length, smooth_polyorder,
                                     deriv=0, axis=axis, mode=mode)

    return deriv


def gauss_deriv(x, y, sigma=1, order=1, axis=0, periodic=False):
    if periodic:
        mode = "wrap"
    else:
        mode = "reflect"

    return ndimage.gaussian_filter(y, sigma=sigma, order=order, axes=(axis,), mode=mode)