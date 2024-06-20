import numpy as np
from sklearn import linear_model as lm
from scipy import stats
from infer_utils import *

class SociohydroInfer2D():
    def __init__(self,
                 ϕ1, ϕ2,
                 x, y, t,
                 t_dim=0,
                 diff_method="savgol"):
        
        
        # make sure time is last dimension
        self.ABt = np.stack([np.moveaxis(ϕ1, t_dim, -1),
                             np.moveaxis(ϕ2, t_dim, -1)],
                            axis=-1)
        self.x = x
        self.dx = x[1] - x[0]
        self.y = y
        self.dy = y[1] - y[0]
        self.t = t
        self.dt = t[1] - t[0]
        diff_method_opts = ["savgol", "spline", "spectral", "gauss"]
        if diff_method.lower() not in diff_method_opts:
            raise ValueError(f"diff_method needs to be in {diff_method_opts}")
        
        self.diff_method = diff_method
        
        self.features = self.calc_features()

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
    
    def calc_features(self):
        # get all derivatives
        ABt_x    = self.differentiate(self.x, self.ABt, order=1, periodic=False, axis=1)    # ∂/∂x
        ABt_y    = self.differentiate(self.y, self.ABt, order=1, periodic=False, axis=0)    # ∂/∂y
        ABt_xx   = self.differentiate(self.x, self.ABt, order=2, periodic=False, axis=1)    # ∂^2/∂x^2
        ABt_yy   = self.differentiate(self.y, self.ABt, order=2, periodic=False, axis=0)    # ∂^2/∂y^2
        ABt_xy   = self.differentiate(self.y, ABt_x, order=1, periodic=False, axis=0)       # ∂^2/(∂x ∂y)
        ABt_xxx  = self.differentiate(self.x, self.ABt, order=3, periodic=False, axis=1)    # ∂^3/∂x^3
        ABt_yyy  = self.differentiate(self.y, self.ABt, order=3, periodic=False, axis=0)    # ∂^3/∂y^3
        ABt_xxy  = self.differentiate(self.y, ABt_xx, order=1, periodic=False, axis=0)      # ∂^3/(∂x^2 ∂x)
        ABt_yyx  = self.differentiate(self.x, ABt_yy, order=1, periodic=False, axis=1)      # ∂^3/(∂y^2 ∂x)
        ABt_xxxx = self.differentiate(self.x, self.ABt, order=4, periodic=False, axis=1)    # ∂^4/∂x^4
        ABt_xxyy = self.differentiate(self.y, ABt_xx, order=2, periodic=False, axis=0)      # ∂^4/(∂x^2 ∂y^2)
        ABt_yyyy = self.differentiate(self.x, self.ABt, order=4, periodic=False, axis=0)    # ∂^4/∂y^4
        # ABt_yyxx = self.differentiate(self.x, ABt_yy, order=2, periodic=False, axis=1)      # ∂^4/∂x^4

        # construct gradient vector, laplacian, gradient of laplacian vector, and bilaplacian
        ABt_D1 = np.stack([ABt_x, ABt_y], axis=-2)  # ∇ vector, shape (x, y, t, coord, field)
        ABt_D2 = ABt_xx + ABt_yy  # laplacian, shape (x, y, t, field)
        ABt_D3x = ABt_xxx + ABt_yyx # x component of ∇^3
        ABt_D3y = ABt_xxy + ABt_yyy # y component of ∇^3
        ABt_D3 = np.stack([ABt_D3x, ABt_D3y], axis=-2)  # ∇^3 vector, shape (x, y, t, coord, field)
        ABt_D4 = ABt_xxxx + 2 * ABt_xxyy + ABt_yyyy # bilaplacian, shape (x, y, t, field)

        # flip A and B for cross-terms
        BAt = np.flip(self.ABt, axis=-1)
        BAt_D1 = np.flip(ABt_D1, axis=-1)
        BAt_D2 = np.flip(ABt_D2, axis=-1)
        # BAt_D3 = np.flip(ABt_D3, axis=-1)
        # BAt_D4 = np.flip(ABt_D4, axis=-1)

        # vacancies
        ϕ0 = 1 - self.ABt.sum(axis=-1)
        ϕ0_x = self.differentiate(self.x, ϕ0, order=1, periodic=False, axis=1)
        ϕ0_y = self.differentiate(self.y, ϕ0, order=1, periodic=False, axis=0)
        ϕ0_D1 = np.stack([ϕ0_x, ϕ0_y], axis=-1)
        ϕ0_xx = self.differentiate(self.x, ϕ0, order=2, periodic=False, axis=1)
        ϕ0_yy = self.differentiate(self.y, ϕ0, order=2, periodic=False, axis=0)
        ϕ0_D2 = ϕ0_xx + ϕ0_yy

        ϕ0 = ϕ0[..., np.newaxis]
        ϕ0_D1 = ϕ0_D1[..., np.newaxis]
        ϕ0_D2 = ϕ0_D2[..., np.newaxis]

        T_term   = (ABt_D2 * ϕ0) - (self.ABt * ϕ0_D2)

        # note that ∇ϕ1 ⋅ ∇ϕ2 requires summing over the -2 dimension
        dot = "...ij,...ij->...j"
        kii_term = (
            self.ABt * np.einsum(dot, ϕ0_D1, ABt_D1) +
            ϕ0 * np.einsum(dot, ABt_D1, ABt_D1) +
            ϕ0 * self.ABt * ABt_D2
        )

        kij_term = (
            self.ABt * np.einsum(dot, ϕ0_D1, BAt_D1) +
            ϕ0 * np.einsum(dot, ABt_D1, BAt_D1) +
            ϕ0 * self.ABt * BAt_D2
        )

        Γ_term = (
            self.ABt * np.einsum(dot, ϕ0_D1, ABt_D3) +
            ϕ0 * np.einsum(dot, ABt_D1, ABt_D3) +
            ϕ0 * self.ABt * ABt_D4
        )

        νiii_term = (
            2 * self.ABt**2 * np.einsum(dot, ϕ0_D1, ABt_D1) + 
            4 * ϕ0 * self.ABt * np.einsum(dot, ABt_D1, ABt_D1) + 
            2 * ϕ0 * self.ABt**2 * ABt_D2
        )

        νiij_term = (
            self.ABt * (BAt * np.einsum(dot, ϕ0_D1, ABt_D1) + self.ABt * np.einsum(dot, ϕ0_D1, BAt_D1)) + 
            ϕ0 * (BAt * np.einsum(dot, ABt_D1, ABt_D1) + self.ABt * np.einsum(dot, ABt_D1, BAt_D1)) +
            ϕ0 * self.ABt * (BAt * ABt_D2 + 2 * np.einsum(dot, ABt_D1, BAt_D1) + self.ABt * BAt_D2)
        )

        νijj_term = (
            2 * self.ABt * BAt * np.einsum(dot, ϕ0_D1, BAt_D1) + 
            2 * ϕ0 * BAt * np.einsum(dot, ABt_D1, BAt_D1) + 
            2 * ϕ0 * self.ABt * np.einsum(dot, BAt_D1, BAt_D1) + 
            2 * ϕ0 * self.ABt * BAt * BAt_D2
        )

        features = np.stack([T_term, kii_term, kij_term, Γ_term,
                            νiii_term, νiij_term, νijj_term], axis=-1)

        return features
    
    def fit(self, train_pct):
        train = np.random.choice(len(self.t), int(len(self.t) * train_pct),
                                 replace=False)
        test = [i for i in np.arange(len(self.t)) if i not in train]

        t_train = self.t[train]
        t_test = self.t[test]

        ABt_dt   = self.differentiate(self.t, self.ABt, order=1, periodic=False, axis=-2)   # ∂/∂t

        # ∂ϕA/∂t
        At_dt = ABt_dt[..., 0]
        At_dt_train = np.reshape(At_dt[..., train], len(self.x) * len(self.y) * len(t_train))
        At_dt_test = np.reshape(At_dt[..., test], len(self.x) * len(self.y) * len(t_test))
        # features of A
        fA = self.features[..., 0, :]
        fA_train = np.reshape(fA[..., train, :], [len(self.x) * len(self.y) * len(t_train), fA.shape[-1]])
        fA_test = np.reshape(fA[..., test, :], [len(self.x) * len(self.y) * len(t_test), fA.shape[-1]])

        # ∂ϕB/∂t
        Bt_dt = ABt_dt[..., 1]
        Bt_dt_train = np.reshape(Bt_dt[..., train], len(self.x) * len(self.y) * len(t_train))
        Bt_dt_test = np.reshape(Bt_dt[..., test], len(self.x) * len(self.y) * len(t_test))
        # features of B
        fB = self.features[..., 1, :]
        fB_train = np.reshape(fB[..., train, :], [len(self.x) * len(self.y) * len(t_train), fB.shape[-1]])
        fB_test = np.reshape(fB[..., test, :], [len(self.x) * len(self.y) * len(t_test), fB.shape[-1]])

        regrA = lm.LinearRegression()
        regrB = lm.LinearRegression()
        # regressor = lm.ElasticNet(alpha=0.1, l1_ratio=0.5)
        # regressor = lm.SGDRegressor(loss="squared_error")
        # regressor = lm.Lasso(alpha=1e-3)

        # perform fit
        fitA = regrA.fit(fA_train, At_dt_train)
        fitB = regrB.fit(fB_train, Bt_dt_train)
        # extract coefficients
        coeffsA = fitA.coef_
        coeffsB = fitB.coef_
        # 
        pearsonr_A = stats.pearsonr(At_dt_test, fitA.predict(fA_test)).statistic
        pearsonr_B = stats.pearsonr(Bt_dt_test, fitB.predict(fB_test)).statistic
        

        return fitA, fitB, pearsonr_A, pearsonr_B
