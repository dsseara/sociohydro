import numpy as np
from sklearn import linear_model as lm
from scipy import stats
from infer_utils import *

class SociohydroInfer2D():
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
    
    diff_method_opts = ["savgol", "spline", "spectral", "gauss"]
    "options for differentiation methods"

    def __init__(self,
                 ϕ1: list, 
                 ϕ2: list,
                 x: list, 
                 y: list, 
                 t: list,
                 t_dim: int = 0,
                 diff_method: str = "savgol"):
        
        if np.any([not isinstance(arg, list) for arg in [ϕ1, ϕ2, x, y, t]]):
            raise TypeError("fields need to be given as lists")
            # we require fields to be given as a list where
            # each element comes from a specific region/county
        
        self.n_regions = len(ϕ1)
        
        # make sure time is last dimension
        self.ABts = [np.stack([np.moveaxis(fi1, t_dim, -1),
                               np.moveaxis(fi2, t_dim, -1)],
                               axis=-1) for fi1, fi2 in zip(ϕ1, ϕ2)]
        self.xs = x
        self.dxs = [eks[1] - eks[0] for eks in x]
        self.ys = y
        self.dys = [why[1] - why[0] for why in y]
        self.ts = t
        self.dts = [tee[1] - tee[0] for tee in t]

        if diff_method.lower() not in self.diff_method_opts:
            raise ValueError(f"diff_method needs to be in {self.diff_method_opts}")
        
        self.diff_method = diff_method

    def differentiate(self, x, y, order, periodic, axis):
        if self.diff_method == "savgol":
            dydx = savgol_deriv(x, y,
                                window_length=5,
                                polyorder=3,
                                order=order,
                                periodic=periodic,
                                axis=axis,
                                smooth=True,
                                smooth_polyorder=2)
        else:
            raise NotImplementedError("Only derivative available is savgol")

        return dydx
    
    def calc_features(self, ABt, x, y):
        # get all derivatives
        ABt_x    = self.differentiate(x, ABt, order=1, periodic=False, axis=1)    # ∂/∂x
        ABt_y    = self.differentiate(y, ABt, order=1, periodic=False, axis=0)    # ∂/∂y
        ABt_xx   = self.differentiate(x, ABt, order=2, periodic=False, axis=1)    # ∂^2/∂x^2
        ABt_yy   = self.differentiate(y, ABt, order=2, periodic=False, axis=0)    # ∂^2/∂y^2
        ABt_xy   = self.differentiate(y, ABt_x, order=1, periodic=False, axis=0)       # ∂^2/(∂x ∂y)
        ABt_xxx  = self.differentiate(x, ABt, order=3, periodic=False, axis=1)    # ∂^3/∂x^3
        ABt_yyy  = self.differentiate(y, ABt, order=3, periodic=False, axis=0)    # ∂^3/∂y^3
        ABt_xxy  = self.differentiate(y, ABt_xx, order=1, periodic=False, axis=0)      # ∂^3/(∂x^2 ∂x)
        ABt_yyx  = self.differentiate(x, ABt_yy, order=1, periodic=False, axis=1)      # ∂^3/(∂y^2 ∂x)
        ABt_xxxx = self.differentiate(x, ABt, order=4, periodic=False, axis=1)    # ∂^4/∂x^4
        ABt_xxyy = self.differentiate(y, ABt_xx, order=2, periodic=False, axis=0)      # ∂^4/(∂x^2 ∂y^2)
        ABt_yyyy = self.differentiate(x, ABt, order=4, periodic=False, axis=0)    # ∂^4/∂y^4
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

        T_term   = (ABt_D2 * ϕ0) - (ABt * ϕ0_D2)

        # note that ∇ϕ1 ⋅ ∇ϕ2 requires summing over the -2 dimension
        dot = "...ij,...ij->...j"
        kii_term = (
            ABt * np.einsum(dot, ϕ0_D1, ABt_D1) +
            ϕ0 * np.einsum(dot, ABt_D1, ABt_D1) +
            ϕ0 * ABt * ABt_D2
        )

        kij_term = (
            ABt * np.einsum(dot, ϕ0_D1, BAt_D1) +
            ϕ0 * np.einsum(dot, ABt_D1, BAt_D1) +
            ϕ0 * ABt * BAt_D2
        )

        Γ_term = (
            ABt * np.einsum(dot, ϕ0_D1, ABt_D3) +
            ϕ0 * np.einsum(dot, ABt_D1, ABt_D3) +
            ϕ0 * ABt * ABt_D4
        )

        νiii_term = (
            2 * ABt**2 * np.einsum(dot, ϕ0_D1, ABt_D1) + 
            4 * ϕ0 * ABt * np.einsum(dot, ABt_D1, ABt_D1) + 
            2 * ϕ0 * ABt**2 * ABt_D2
        )

        νiij_term = (
            ABt * (BAt * np.einsum(dot, ϕ0_D1, ABt_D1) + ABt * np.einsum(dot, ϕ0_D1, BAt_D1)) + 
            ϕ0 * (BAt * np.einsum(dot, ABt_D1, ABt_D1) + ABt * np.einsum(dot, ABt_D1, BAt_D1)) +
            ϕ0 * ABt * (BAt * ABt_D2 + 2 * np.einsum(dot, ABt_D1, BAt_D1) + ABt * BAt_D2)
        )

        νijj_term = (
            2 * ABt * BAt * np.einsum(dot, ϕ0_D1, BAt_D1) + 
            2 * ϕ0 * BAt * np.einsum(dot, ABt_D1, BAt_D1) + 
            2 * ϕ0 * ABt * np.einsum(dot, BAt_D1, BAt_D1) + 
            2 * ϕ0 * ABt * BAt * BAt_D2
        )

        features = np.stack([T_term, kii_term, kij_term, Γ_term,
                            νiii_term, νiij_term, νijj_term], axis=-1)

        return features
    

    def test_train_split(self, train_pct):

        # assume that all fields occur at the same time so
        # we assume that train/test split occurs at same times
        # for all
        train = np.random.choice(len(self.ts[0]),
                                 int(len(self.ts[0]) * train_pct),
                                 replace=False)
        test = np.array([i for i in np.arange(len(self.ts[0])) if i not in train])

        dAdt_train = []
        dAdt_test = []
        dBdt_train = []
        dBdt_test = []
        featA_train = []
        featA_test = []
        featB_train = []
        featB_test = []
        
        # loop over all datasets
        for ABt, x, y, t in zip(self.ABts, self.xs, self.ys, self.ts):
            features = self.calc_features(ABt, x, y)
            nfeat = features.shape[-1]
            ABt_dt   = self.differentiate(t, ABt, order=1, periodic=False, axis=-2)   # ∂/∂t

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

            At_dt_train = At_dt[..., train][notnan].ravel()
            At_dt_test  = At_dt[...,  test][notnan].ravel()
            fA_train = np.reshape(fA[..., train, :][notnan], [len(At_dt_train), nfeat])
            fA_test  = np.reshape(fA[..., test,  :][notnan], [len(At_dt_test),  nfeat])
            Bt_dt_train = Bt_dt[..., train][notnan].ravel()
            Bt_dt_test  = Bt_dt[...,  test][notnan].ravel()
            fB_train = np.reshape(fB[..., train, :][notnan], [len(Bt_dt_train), nfeat])
            fB_test  = np.reshape(fB[...,  test, :][notnan], [len(Bt_dt_test),  nfeat])

            # append data to everything
            # nanmask = np.logical_and(np.logical_and(np.all(~np.isnan(At_dt), axis=-1), np.all(~np.isnan(fA), axis=(-1, -2))),
            #                          np.logical_and(np.all(~np.isnan(Bt_dt), axis=-1), np.all(~np.isnan(fB), axis=(-1, -2))))
            # mask_train = np.all(~np.isnan(fA_train), axis=1)
            # mask_test = np.all(~np.isnan(fA_test), axis=1)
            
            dAdt_train.append(At_dt_train)
            dAdt_test.append(At_dt_test)
            featA_train.append(fA_train)
            featA_test.append(fA_test)
            dBdt_train.append(Bt_dt_train)
            dBdt_test.append(Bt_dt_test)
            featB_train.append(fB_train)
            featB_test.append(fB_test)
        
        dAdt = {"train": np.concatenate(dAdt_train),
                "test" : np.concatenate(dAdt_test)}
        dBdt = {"train": np.concatenate(dBdt_train),
                "test" : np.concatenate(dBdt_test)}
        
        featA = {"train": np.concatenate(featA_train),
                 "test" : np.concatenate(featA_test)}
        featB = {"train": np.concatenate(featB_train),
                 "test" : np.concatenate(featB_test)}

        return dAdt, dBdt, featA, featB
        

    
    def fit(self, train_pct, regressor="linear", alpha=0.1):
        dAdt, dBdt, featA, featB = self.test_train_split(train_pct)

        if regressor.lower() == "linear":
            regrA = lm.LinearRegression()
            regrB = lm.LinearRegression()
        elif regressor.lower() == "elasticnet":
            regrA = lm.ElasticNet(alpha=alpha, l1_ratio=0.5)
            regrB = lm.ElasticNet(alpha=alpha, l1_ratio=0.5)
        elif regressor.lower() == "sgd":
            regrA = lm.SGDRegressor(loss="squared_error")
            regrB = lm.SGDRegressor(loss="squared_error")
        elif regressor.lower() == "lasso":
            regrA = lm.Lasso(alpha=alpha)
            regrB = lm.Lasso(alpha=alpha)
        else:
            raise ValueError("Regressor must be one of ['linear', 'elastic', 'sgd', 'lasso']. Currently: " + regressor)
        
        # perform fit
        fitA = regrA.fit(featA["train"], dAdt["train"])
        fitB = regrB.fit(featB["train"], dBdt["train"])

        # pearson correlation coefficient
        pearsonr_A = stats.pearsonr(dAdt["test"], fitA.predict(featA["test"])).statistic
        pearsonr_B = stats.pearsonr(dBdt["test"], fitB.predict(featB["test"])).statistic
        

        return fitA, fitB, pearsonr_A, pearsonr_B

