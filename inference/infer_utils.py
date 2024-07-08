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

def fd_deriv(x, y, order=1):
    Nx = len(x)
    dx = x[1] - x[0]
    derivs = [d0, d1, d2, d3, d4]
    deriv = derivs[order](Nx, dx) @ y

    return deriv



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