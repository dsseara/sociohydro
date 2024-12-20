import numpy as np
from sklearn import linear_model as lm
from scipy import stats, optimize
from infer_utils import *

def linear(x, a, b):
    return a * x + b

def exp(x, a, b, c):
    return a * np.exp(b * x) + c

def fit_growth(x, y, form="exp"):
    """
    Fit a growth curve to data.

    Parameters
    ----------
    x : array-like
        Independent variable (time, etc.)
    y : array-like
        Dependent variable (population size, etc.)
    form : str, {'exp', 'linear'}
        Form of the growth curve.

    Returns
    -------
    popt : array-like
        Optimal parameters for the growth curve.
    """
    form_list = ["exp", "linear"]
    
    if form.lower() not in form_list:
        raise ValueError("growth is either exp or linear")
    
    if form.lower() == "exp":
        growth = exp
        p0 = [1e-3, 1e-6, 0]
    elif form.lower() == "linear":
        growth = linear
        p0 = [1e-3, 1e-6]
    
    popt, pcov = optimize.curve_fit(growth, x, y,
                                    p0=p0,
                                    maxfev=int(1e5))
    
    return popt, np.sqrt(np.diag(pcov))


class SociohydroInfer():
    """
    Base class used to perform linear regression on time series data
    to fit to sociohydrodynamic model with generic quadratic utility
    functions. Assumes 2 groups.

    Input
    -----
    ϕ1 : list[array-like]
        list of fields for first group. Each element of list is has 3 dimensions,
        (t, y, x). If time dimension is not at -1, pass t_dim to 
    ϕ2 : list[array-like]
        list of fields for second group. Each element of list is has 3 dimensions,
        (t, y, x). If time dimension is not at -1, pass t_dim to 
    x : list[array-like]
        list of x-positions of fields. Each element of list is a 1D array-like
    t : list[array-like]
        list of time points of fields. Each element of list is a 1D array-like
    t_dim : int (optional)
        location of t-dimension of data, to be moved to last position
    diff_method : str
        string determining how to take derivatives. Options are
        ["savgol", "spline", "spectral", "gauss"].
        note: only savgol currently implemented
    periodic : bool
        whether to use periodic spatial derivatives 

    Attributes
    ----------
    ABts : list[array]
        combined arrays of ϕ1's and ϕ2's
    xs : list[array]
        equal to input, x
    ts : list[array]
        equal to input, t
    diff_method : str
        equal to input, diff_method

    """
    
    diff_method_opts = ["savgol", "spline", "spectral", "gauss", "fd"]
    "options for differentiation methods"
    regressor_opts = ['linear', 'ridge', 'elasticnet', 'sgd', 'lasso']

    def __init__(self,
                 ϕ1: list,
                 ϕ2: list,
                 xs: list,
                 ts: list,
                 t_dim: int = 0,
                 diff_method: str = "savgol",
                 periodic: bool = False,
                 masks = None
                 ):
        if np.any([not isinstance(arg, list) for arg in [ϕ1, ϕ2, xs, ts]]):
            raise TypeError("fields need to be given as lists")
            # we require fields to be given as a list where
            # each element comes from a specific region/county
        
        self.n_regions = len(ϕ1)
        
        # make sure time is last dimension
        self.ABts = [np.stack([np.moveaxis(fi1, t_dim, -1),
                               np.moveaxis(fi2, t_dim, -1)],
                               axis=-1) for fi1, fi2 in zip(ϕ1, ϕ2)]
        self.xs = xs
        self.dxs = [x[1] - x[0] for x in xs]
        self.ts = ts
        self.dts = [t[1] - t[0] for t in ts]
        self.periodic = periodic

        if masks is None:
            masks = [np.full(True, ABt.shape[:2]) for ABt in self.ABts]
        else:
            if len(masks) != self.n_regions:
                raise ValueError("Number of masks should be same as number of regions")
        
        self.masks = masks

        if diff_method.lower() not in self.diff_method_opts:
            raise ValueError(f"diff_method needs to be in {self.diff_method_opts}")

        if diff_method.lower() == "fd" and not self.periodic:
            raise ValueError(f"If using finite difference, must be periodic")

        self.diff_method = diff_method

    def differentiate(self, x, f, order, periodic, axis,
                      window_length=5, polyorder=4,
                      smooth=True, smooth_polyorder=2):
        if self.diff_method == "savgol":
            dfdx = savgol_deriv(x, f,
                                window_length=window_length,
                                polyorder=polyorder,
                                order=order,
                                periodic=periodic,
                                axis=axis,
                                smooth=smooth,
                                smooth_polyorder=smooth_polyorder)
        elif self.diff_method == "fd":
            dfdx = fd_deriv(x, f,
                            order=order,
                            axis=axis)
        else:
            raise NotImplementedError("Only savgol or fd derivatives available")

        return dfdx
    
    def calc_growthRate(self, region_id):
        
        ABt = self.ABts[region_id]
        t = self.ts[region_id]

        spatial_axes = tuple(range(0, ABt.ndim-2))
        ABt_mean = np.nansum(ABt, axis=spatial_axes)
        g = []
        for phi in ABt_mean.T:
            popt, pcov = fit_growth(t, np.log(phi), form="linear")
            g.append(popt[1])
        
        return g

    def calc_features(self, region, window_length=5):
        pass

    def test_train_split(self, train_pct,
                         window_length=5,
                         ddt_minimum=0.0,
                         consider_growth=True):

        dAdt = []
        dBdt = []
        featA = []
        featB = []
        growth_rates = []

        # else:
        #     # create tuple of slices for the spatial dimensions
        #     slices = tuple(slice(start, stop) for start, stop in lims)
        #     # add full slice for all other dimensions
        #     slices += (slice(None),) * (self.ABts[0].ndim - len(slices))

        # loop over all datasets
        # for ABt, x, t in zip(self.ABts, self.xs, self.ts):
        for region in range(self.n_regions):
            if consider_growth:
                growth_rates.append(self.calc_growthRate(region))
            else:
                growth_rates.append(np.zeros(2))

            growth_rates = np.zeros((self.n_regions, 2))
            features = self.calc_features(region, window_length=window_length)
            nfeat = features.shape[-1]
            ABt_dt   = self.differentiate(self.ts[region],
                                          self.ABts[region],
                                          order=1,
                                          axis=-2,
                                          periodic=False,
                                          window_length=window_length)  # ∂/∂t
            # growth_rates = self.calc_growthRate(region)

            # if consider_growth:
            ABt_dt -= self.ABts[region] * growth_rates[region]

            features = features[self.masks[region]]
            ABt_dt = ABt_dt[self.masks[region]]

            large_ddt = np.where((np.sum(np.abs(ABt_dt[..., 0]), axis=-1) >= ddt_minimum) &
                                 (np.sum(np.abs(ABt_dt[..., 1]), axis=-1) >= ddt_minimum))

            features = features[large_ddt]
            ABt_dt = ABt_dt[large_ddt]

            # ∂ϕA/∂t
            At_dt = ABt_dt[..., 0]
            At_dt_notnan = np.all(~np.isnan(At_dt), axis=-1)
            # features of A
            fA = features[..., 0, :]
            fA_notnan = np.all(~np.isnan(fA), axis=(-1, -2))
            # ∂ϕB/∂t
            Bt_dt = ABt_dt[..., 1]
            Bt_dt_notnan = np.all(~np.isnan(Bt_dt), axis=-1)
            # features of B
            fB = features[..., 1, :]
            fB_notnan = np.all(~np.isnan(fB), axis=(-1, -2))

            notnan = np.logical_and(np.logical_and(At_dt_notnan, fA_notnan),
                                    np.logical_and(Bt_dt_notnan, fB_notnan))

            At_dt = At_dt[notnan].ravel()
            fA = np.reshape(fA[notnan], [len(At_dt), nfeat])
            Bt_dt = Bt_dt[notnan].ravel()
            fB = np.reshape(fB[notnan], [len(Bt_dt), nfeat])

            # append data to everything
            dAdt.append(At_dt)
            featA.append(fA)
            dBdt.append(Bt_dt)
            featB.append(fB)

        dAdt = np.concatenate(dAdt)
        featA = np.concatenate(featA)
        dBdt = np.concatenate(dBdt)
        featB = np.concatenate(featB)
        
        npts = len(dAdt)
        train = np.random.choice(npts, int(npts * train_pct), replace=False)
        test = np.array(list(set(range(npts)) - set(train)))

        
        dAdt = {"train": dAdt[train],
                "test" : dAdt[test]}
        dBdt = {"train": dBdt[train],
                "test" : dBdt[test]}
        
        featA = {"train": featA[train, :],
                 "test" : featA[test, :]}
        featB = {"train": featB[train, :],
                 "test" : featB[test, :]}

        return dAdt, dBdt, featA, featB, growth_rates


    def fit(self, train_pct,
            regressor="linear",
            alpha=0.1,
            window_length=5,
            ddt_minimum=0.0,
            consider_growth=True):

        """
        Fits regression models to the dataset using specified parameters.

        Parameters:
            train_pct: (float)
                The percentage of data to be used for training, ranging from 0 to 1.
            regressor: (str, optional)
                Type of regressor to use. Options are "linear", "ridge",
                "elasticnet", "sgd", and "lasso". Default is "linear".
            alpha: (float, optional)
                Regularization strength; must be a positive float. Used by
                "ridge", "elasticnet", and "lasso" regressors. Default is 0.1.
            window_length: (int, optional)
                Length of the window for smoothing operations. Default is 5.
            ddt_minimum: (float, optional)
                Minimum threshold for time derivatives. Default is 0.0.
            consider_growth: (bool, optional)
                Whether to consider growth rates in the fitting process. Default is True.

        Returns:
            fits: (tuple)
                List of fitted regression models for datasets A and B
            ddts: (tuple)
                Dictionaries for datasets A and B containing train and test data for derivatives
            feats: (tuple)
                Dictionaries for datasets A and B containing train and test data for features
            mse: (tuple)
                List of mean squared errors for datasets A and B on test sets
            growth_rates: (tuple)
                Growth rates used during the fitting process
        """
        dAdt, dBdt, featA, featB, growth_rates = self.test_train_split(train_pct,
                                                                       window_length=window_length,
                                                                       ddt_minimum=ddt_minimum,
                                                                       consider_growth=consider_growth)

        if regressor.lower() not in self.regressor_opts:
            raise ValueError(f"Regressor must be one of {self.regressor_opts}. Currently: " + regressor)
        
        if regressor.lower() == "linear":
            regA = lm.LinearRegression()
            regB = lm.LinearRegression()
        elif regressor.lower() == "ridge":
            regA = lm.Ridge(alpha=alpha)
            regB = lm.Ridge(alpha=alpha)
        elif regressor.lower() == "elasticnet":
            regA = lm.ElasticNet(alpha=alpha, l1_ratio=0.5)
            regB = lm.ElasticNet(alpha=alpha, l1_ratio=0.5)
        elif regressor.lower() == "sgd":
            regA = lm.SGDRegressor(loss="squared_error")
            regB = lm.SGDRegressor(loss="squared_error")
        elif regressor.lower() == "lasso":
            regA = lm.Lasso(alpha=alpha)
            regB = lm.Lasso(alpha=alpha)

        # perform fit
        fitA = regA.fit(featA["train"], dAdt["train"])
        fitB = regB.fit(featB["train"], dBdt["train"])

        # pearson correlation coefficient
        mseA = np.mean((dAdt["test"] - fitA.predict(featA["test"]))**2)
        mseB = np.mean((dBdt["test"] - fitB.predict(featB["test"]))**2)

        return [fitA, fitB], [dAdt, dBdt], [featA, featB], [mseA, mseB], growth_rates


class SociohydroInfer2D(SociohydroInfer):
    """
    Class used to perform linear regression on 2D time series data
    to fit to sociohydrodynamic model with generic quadratic utility
    functions. Assumes 2 groups.

    Input
    -----
    ϕ1 : list[array-like]
        list of fields for first group. Each element of list is has 3 dimensions,
        (t, y, x). If time dimension is not at -1, pass t_dim to 
    ϕ2 : list[array-like]
        list of fields for second group. Each element of list is has 3 dimensions,
        (t, y, x). If time dimension is not at -1, pass t_dim to 
    x : list[array-like]
        list of x-positions of fields. Each element of list is a 1D array-like
    y : list[array-like]
        list of y-positions of fields. Each element of list is a 1D array-like
    t : list[array-like]
        list of time points of fields. Each element of list is a 1D array-like
    t_dim : int (optional)
        location of t-dimension of data, to be moved to last position
    diff_method : str
        string determining how to take derivatives. Options are
        ["savgol", "spline", "spectral", "gauss"].
        note: only savgol currently implemented

    Attributes
    ----------
    ABts : list[array]
        combined arrays of ϕ1's and ϕ2's
    xs : list[array]
        equal to input, x
    ys : list[array]
        equal to input, y
    ts : list[array]
        equal to input, t
    diff_method : str
        equal to input, diff_method

    """

    def __init__(self, 
                 ϕ1: list, 
                 ϕ2: list,             
                 x: list, 
                 y: list, 
                 t: list,
                 t_dim: int = 0,
                 diff_method: str = "savgol",
                 periodic: bool = False,
                 masks = None):
        SociohydroInfer.__init__(self, ϕ1, ϕ2, x, t,
                                 t_dim, diff_method, periodic,
                                 masks=masks)
        self.ys = y
        self.dys = [why[1] - why[0] for why in y]

    
    def calc_features(self, region, window_length=5):
        ABt = self.ABts[region]
        x = self.xs[region]
        y = self.ys[region]

        # get all derivatives
        ABt_x    = self.differentiate(x, ABt, order=1, axis=1,
                                      periodic=self.periodic,
                                      window_length=window_length,
                                      smooth_polyorder=2)    # ∂/∂x
        ABt_y    = self.differentiate(y, ABt, order=1, axis=0,
                                      periodic=self.periodic,
                                      window_length=window_length,
                                      smooth_polyorder=2)    # ∂/∂y
        ABt_xx   = self.differentiate(x, ABt, order=2, axis=1,
                                      periodic=self.periodic,
                                      window_length=window_length,
                                      smooth_polyorder=2)    # ∂^2/∂x^2
        ABt_yy   = self.differentiate(y, ABt, order=2, axis=0,
                                      periodic=self.periodic,
                                      window_length=window_length,
                                      smooth_polyorder=2)    # ∂^2/∂y^2
        ABt_xy   = self.differentiate(y, ABt_x, order=1, axis=0,
                                      periodic=self.periodic,
                                      window_length=window_length,
                                      smooth_polyorder=2)       # ∂^2/(∂x ∂y)
        ABt_xxx  = self.differentiate(x, ABt, order=3, axis=1,
                                      periodic=self.periodic,
                                      window_length=window_length,
                                      smooth_polyorder=2)    # ∂^3/∂x^3
        ABt_yyy  = self.differentiate(y, ABt, order=3, axis=0,
                                      periodic=self.periodic,
                                      window_length=window_length,
                                      smooth_polyorder=2)    # ∂^3/∂y^3
        ABt_xxy  = self.differentiate(y, ABt_xx, order=1, axis=0,
                                      periodic=self.periodic,
                                      window_length=window_length,
                                      smooth_polyorder=2)      # ∂^3/(∂x^2 ∂x)
        ABt_yyx  = self.differentiate(x, ABt_yy, order=1, axis=1,
                                      periodic=self.periodic,
                                      window_length=window_length,
                                      smooth_polyorder=2)      # ∂^3/(∂y^2 ∂x)
        ABt_xxxx = self.differentiate(x, ABt, order=4, axis=1,
                                      periodic=self.periodic,
                                      window_length=window_length,
                                      smooth_polyorder=1)    # ∂^4/∂x^4
        ABt_xxyy = self.differentiate(y, ABt_xx, order=2, axis=0,
                                      periodic=self.periodic,
                                      window_length=window_length,
                                      smooth_polyorder=2)      # ∂^4/(∂x^2 ∂y^2)
        ABt_yyyy = self.differentiate(y, ABt, order=4, axis=0,
                                      periodic=self.periodic,
                                      window_length=window_length,
                                      smooth_polyorder=1)    # ∂^4/∂y^4
        # ABt_yyxx = self.differentiate(x, ABt_yy, order=2, periodic=False, axis=1)      # ∂^4/∂x^4

        # construct gradient vector, laplacian, gradient of laplacian vector, and bilaplacian
        ABt_D1 = np.stack([ABt_x, ABt_y], axis=-2)  # ∇ vector, shape (x, y, t, coord, field)
        ABt_D2 = ABt_xx + ABt_yy  # laplacian, shape (x, y, t, field)
        ABt_D3x = ABt_xxx + ABt_yyx # x component of ∇^3
        ABt_D3y = ABt_xxy + ABt_yyy # y component of ∇^3
        ABt_D3 = np.stack([ABt_D3x, ABt_D3y], axis=-2)  # ∇^3 vector, shape (x, y, t, coord, field)
        ABt_D4 = ABt_xxxx + 2 * ABt_xxyy + ABt_yyyy # bilaplacian, shape (x, y, t, field)

        # flip A and B for cross-terms
        BAt = np.flip(ABt, axis=-1)
        BAt_D1 = np.flip(ABt_D1, axis=-1)
        BAt_D2 = np.flip(ABt_D2, axis=-1)
        # BAt_D3 = np.flip(ABt_D3, axis=-1)
        # BAt_D4 = np.flip(ABt_D4, axis=-1)

        # vacancies
        ϕ0 = 1 - ABt.sum(axis=-1)
        ϕ0_x = self.differentiate(x, ϕ0, order=1, periodic=False, axis=1)
        ϕ0_y = self.differentiate(y, ϕ0, order=1, periodic=False, axis=0)
        ϕ0_D1 = np.stack([ϕ0_x, ϕ0_y], axis=-1)
        ϕ0_xx = self.differentiate(x, ϕ0, order=2, periodic=False, axis=1)
        ϕ0_yy = self.differentiate(y, ϕ0, order=2, periodic=False, axis=0)
        ϕ0_D2 = ϕ0_xx + ϕ0_yy

        ϕ0 = ϕ0[..., np.newaxis]
        ϕ0_D1 = ϕ0_D1[..., np.newaxis]
        ϕ0_D2 = ϕ0_D2[..., np.newaxis]

        # div( φ0 grad(φa) - φa grad(φ0) )
        T_term   = (ABt_D2 * ϕ0) - (ABt * ϕ0_D2)

        # note that ∇ϕ1 ⋅ ∇ϕ2 requires summing over the -2 dimension
        dot = "...ij,...ij->...j"
        
        # div( φ0 φa grad(φa) )
        kii_term = (
            ABt * np.einsum(dot, ϕ0_D1, ABt_D1) +
            ϕ0 * np.einsum(dot, ABt_D1, ABt_D1) +
            ϕ0 * ABt * ABt_D2
        )

        # div( φ0 φa grad(φb) )
        kij_term = (
            ABt * np.einsum(dot, ϕ0_D1, BAt_D1) +
            ϕ0 * np.einsum(dot, ABt_D1, BAt_D1) +
            ϕ0 * ABt * BAt_D2
        )

        # div( φ0 φa grad(lap(φa)) )
        Γ_term = (
            ABt * np.einsum(dot, ϕ0_D1, ABt_D3) +
            ϕ0 * np.einsum(dot, ABt_D1, ABt_D3) +
            ϕ0 * ABt * ABt_D4
        )

        # div( φ0 φa grad(φa^2) )
        νiii_term = (
            2 * ABt**2 * np.einsum(dot, ϕ0_D1, ABt_D1) + 
            4 * ϕ0 * ABt * np.einsum(dot, ABt_D1, ABt_D1) + 
            2 * ϕ0 * ABt**2 * ABt_D2
        )
        
        # div( φ0 φa grad(φa φb) )
        νiij_term = (
            ABt * (BAt * np.einsum(dot, ϕ0_D1, ABt_D1) + ABt * np.einsum(dot, ϕ0_D1, BAt_D1)) + 
            ϕ0 * (BAt * np.einsum(dot, ABt_D1, ABt_D1) + ABt * np.einsum(dot, ABt_D1, BAt_D1)) +
            ϕ0 * ABt * (BAt * ABt_D2 + 2 * np.einsum(dot, ABt_D1, BAt_D1) + ABt * BAt_D2)
        )

        # div( φ0 φa grad(φb^2) )
        νijj_term = (
            2 * ABt * BAt * np.einsum(dot, ϕ0_D1, BAt_D1) + 
            2 * ϕ0 * BAt * np.einsum(dot, ABt_D1, BAt_D1) + 
            2 * ϕ0 * ABt * np.einsum(dot, BAt_D1, BAt_D1) + 
            2 * ϕ0 * ABt * BAt * BAt_D2
        )

        features = np.stack([T_term, kii_term, kij_term, Γ_term,
                            νiii_term, νiij_term, νijj_term], axis=-1)

        return features
    

class SociohydroInfer1D(SociohydroInfer):
    """
    Class used to perform linear regression on 1D time series data
    to fit to sociohydrodynamic model with generic quadratic utility
    functions. Assumes 2 groups.

    Input
    -----
    ϕ1 : list[array-like]
        list of fields for first group. Each element of list is has 3 dimensions,
        (t, y, x). If time dimension is not at -1, pass t_dim to 
    ϕ2 : list[array-like]
        list of fields for second group. Each element of list is has 3 dimensions,
        (t, y, x). If time dimension is not at -1, pass t_dim to 
    x : list[array-like]
        list of x-positions of fields. Each element of list is a 1D array-like
    t : list[array-like]
        list of time points of fields. Each element of list is a 1D array-like
    t_dim : int (optional)
        location of t-dimension of data, to be moved to last position
    diff_method : str
        string determining how to take derivatives. Options are
        ["savgol", "spline", "spectral", "gauss"].
        note: only savgol currently implemented

    Attributes
    ----------
    ABts : list[array]
        combined arrays of ϕ1's and ϕ2's
    xs : list[array]
        equal to input, x
    ts : list[array]
        equal to input, t
    diff_method : str
        equal to input, diff_method

    """
    def __init__(self,
                 ϕ1: list,
                 ϕ2: list,
                 x: list,
                 t: list,
                 t_dim: int = 0,
                 diff_method: str = "savgol",
                 periodic: bool = False):
        
        SociohydroInfer.__init__(self,
                                 ϕ1, ϕ2, x, t,
                                 t_dim=t_dim,
                                 diff_method=diff_method,
                                 periodic=periodic,
                                 masks=None)
    
    def calc_features(self, region, window_length=5):
        ABt = self.ABts[region]
        x = self.xs[region]
        # get all derivatives
        ABt_d1 = self.differentiate(x, ABt, order=1, axis=0,
                                    periodic=self.periodic,
                                    window_length=window_length,
                                    smooth_polyorder=2)  # ∂/∂x
        ABt_d2 = self.differentiate(x, ABt, order=2, axis=0,
                                    periodic=self.periodic,
                                    window_length=window_length,
                                    smooth_polyorder=2)  # ∂^2/∂x^2
        ABt_d3 = self.differentiate(x, ABt, order=3, axis=0,
                                    periodic=self.periodic,
                                    window_length=window_length,
                                    smooth_polyorder=2)  # ∂^3/∂x^3
        ABt_d4 = self.differentiate(x, ABt, order=4, axis=0,
                                    periodic=self.periodic,
                                    window_length=window_length,
                                    smooth_polyorder=1)  # ∂^4/∂x^4

        # flip A and B for cross-terms
        BAt = np.flip(ABt, axis=-1)
        BAt_d1 = np.flip(ABt_d1, axis=-1)
        BAt_d2 = np.flip(ABt_d2, axis=-1)
        # BAt_d3 = np.flip(ABt_d3, axis=-1)
        # BAt_d4 = np.flip(ABt_d4, axis=-1)

        # vacancies
        ϕ0 = 1 - ABt.sum(axis=-1)
        ϕ0 = ϕ0[..., np.newaxis]
        ϕ0_d1 = self.differentiate(x, ϕ0, order=1, axis=0,
                                   periodic=self.periodic,
                                   window_length=window_length)
        ϕ0_d2 = self.differentiate(x, ϕ0, order=2, axis=0,
                                   periodic=self.periodic,
                                   window_length=window_length)

        # ∂x( φ0 ∂x φi - φi ∂x φ0 )
        # T_term   = (ABt_d2 * ϕ0) - (ABt * ϕ0_d2)
        T_term = (1 - BAt) * ABt_d2 + ABt * BAt_d2

        # ∂x( φ0 φa ∂x(φa) )
        # kii_term = (
        #     (ϕ0_d1 * ABt * ABt_d1) +
        #     (ϕ0 * ABt_d1 * ABt_d1) +
        #     (ϕ0 * ABt * ABt_d2)
        # )
        kii_term = self.differentiate(x, ABt * ϕ0 * ABt_d1,
                                      order=1, axis=0,
                                      periodic=self.periodic,
                                      window_length=window_length)
        
        # ∂x( φ0 φa ∂x(φb) )
        # kij_term = (
        #     (ϕ0_d1 * ABt * BAt_d1) +
        #     (ϕ0 * ABt_d1 * BAt_d1) +
        #     (ϕ0 * ABt * BAt_d2)
        # )
        kij_term = self.differentiate(x, ABt * ϕ0 * BAt_d1,
                                      order=1, axis=0,
                                      periodic=self.periodic,
                                      window_length=window_length)

        # ∂x(φ0 φa ∂xxx(φa))
        # Γ_term   = (
        #     (ϕ0_d1 * ABt * ABt_d3) +
        #     (ϕ0 * ABt_d1 * ABt_d3) +
        #     (ϕ0 * ABt * ABt_d4)
        # )
        Γ_term = self.differentiate(x, ABt * ϕ0 * ABt_d3,
                                    order=1, axis=0,
                                    periodic=self.periodic,
                                    window_length=window_length)

        # ∂x( φ0 φa ∂x(φa^2) )
        # νiii_term = (
        #     (ϕ0_d1 * ABt    * (2 * ABt * ABt_d1)) + 
        #     (ϕ0    * ABt_d1 * (2 * ABt * ABt_d1)) +
        #     (ϕ0    * ABt    * (2 * (ABt_d1**2 + ABt * ABt_d2)))
        # )
        ABt_ABt_d1 = self.differentiate(x, ABt * ABt, order=1, axis=0,
                                        periodic=self.periodic,
                                        window_length=window_length,
                                        smooth_polyorder=2)  # ∂/∂x
        νiii_term = self.differentiate(x, ABt * ϕ0 * ABt_ABt_d1,
                                       order=1, axis=0,
                                       periodic=self.periodic,
                                       window_length=window_length)
        
        # ∂x( φ0 φa ∂x(φa φb) )
        # νiij_term = (
        #     (ϕ0_d1 * ABt    * (ABt_d1 * BAt + ABt * BAt_d1)) +
        #     (ϕ0    * ABt_d1 * (ABt_d1 * BAt + ABt * BAt_d1)) +
        #     (ϕ0    * ABt    * (ABt_d2 * BAt + 2 * ABt_d1 * BAt_d1 + ABt * BAt_d2))
        # )
        ABt_BAt_d1 = self.differentiate(x, ABt * BAt, order=1, axis=0,
                                        periodic=self.periodic,
                                        window_length=window_length,
                                        smooth_polyorder=2)  # ∂/∂x
        νiij_term = self.differentiate(x, ABt * ϕ0 * ABt_BAt_d1,
                                       order=1, axis=0,
                                       periodic=self.periodic,
                                       window_length=window_length)
        
        # ∂x( φ0 φa ∂x(φb^2) )
        # νijj_term = (
        #     (ϕ0_d1 * ABt    * (2 * BAt * BAt_d1)) +
        #     (ϕ0    * ABt_d1 * (2 * BAt * BAt_d1)) +
        #     (ϕ0    * ABt    * (2 * (BAt_d1**2 + BAt * BAt_d2)))
        # )
        BAt_BAt_d1 = self.differentiate(x, ABt * BAt, order=1, axis=0,
                                        periodic=self.periodic,
                                        window_length=window_length,
                                        smooth_polyorder=2)  # ∂/∂x
        νijj_term = self.differentiate(x, ABt * ϕ0 * BAt_BAt_d1,
                                       order=1, axis=0,
                                       periodic=self.periodic,
                                       window_length=window_length)

        features = np.stack([T_term, kii_term, kij_term, Γ_term,
                             νiii_term, νiij_term, νijj_term], axis=-1)

        return features