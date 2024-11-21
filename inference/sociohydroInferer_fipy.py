import numpy as np
from sklearn import linear_model as lm
from scipy import stats, optimize, ndimage
import fipy as fp
from tqdm import tqdm
from infer_utils import *

class SociohydroInfer_fipy():
    def __init__(self, varA, varB, mesh, t, t_dim=0, smooth_ddt=False):
        self.mesh = mesh
        self.varA = np.moveaxis(np.asarray(varA), t_dim, 0)
        self.varB = np.moveaxis(np.asarray(varB), t_dim, 0)
        self.ts = t
        self.nt = len(t)
        self.ncell = mesh.numberOfCells
        self.nfeat = 7

        self.calc_features()
        self.calc_ddt(smooth=smooth_ddt)

    def calc_features(self):
        """
        features are calculated at each time step, and come in order
        [Ti, kii, kij, nuiii, nuiij, nuijj, Γi]
        """
        print("calculating features...")
        featA = np.zeros((self.nt, self.nfeat, self.ncell))
        featB = np.zeros((self.nt, self.nfeat, self.ncell))
        for tidx, (vA, vB) in enumerate(zip(tqdm(self.varA), self.varB)):
            featA[tidx], featB[tidx] = calc_fipyTerms(
                fp.CellVariable(mesh=self.mesh, value=vA),
                fp.CellVariable(mesh=self.mesh, value=vB)
            )

        self.featA = np.moveaxis(featA,1,2)
        self.featB = np.moveaxis(featB,1,2)

    def calc_ddt(self, smooth=False):
        print("calculating time derivative...")
        dt = np.diff(self.ts)[0]
        dAdt = np.gradient(self.varA, dt, axis=0)
        dBdt = np.gradient(self.varB, dt, axis=0)
        if smooth:
            dAdt = ndimage.median_filter(dAdt,
                                         size=3,
                                         axes=0,
                                         mode="nearest")
            dBdt = ndimage.median_filter(dBdt,
                                         size=3,
                                         axes=0,
                                         mode="nearest")

        self.dAdt = dAdt
        self.dBdt = dBdt
    
    def train_test_split(self, train_pct,
                         ddt_minimum=0.0,
                         consider_growth=True,
                         growth_rates=[]):

        if consider_growth:
            if len(growth_rates) != 2:
                raise ValueError("need a grwoth rate for each population")
        else:
            growth_rates = np.zeros(2)

        # dAdt, dBdt = self.calc_ddt(smooth=smooth_ddt)
        # featA, featB = self.calc_features()

        large_ddt = np.where(
            (np.abs(self.dAdt).sum(axis=0) >= ddt_minimum) &
            (np.abs(self.dBdt).sum(axis=0) >= ddt_minimum)
        )

        # look only at regions with large time derivatives
        dAdt = np.squeeze(self.dAdt[:, large_ddt])
        dBdt = np.squeeze(self.dBdt[:, large_ddt])
        featA = np.squeeze(self.featA[:, large_ddt, :])
        featB = np.squeeze(self.featB[:, large_ddt, :])

        # subtract off exponential growth rates from rhs
        dAdt -= dAdt * growth_rates[0]
        dBdt -= dBdt * growth_rates[1]

        # reshape
        featA = featA.reshape(-1, self.nfeat)
        featB = featB.reshape(-1, self.nfeat)
        dAdt = dAdt.reshape(-1)
        dBdt = dBdt.reshape(-1)

        npts = len(dAdt)
        train = np.random.choice(npts, int(npts * train_pct), replace=False)
        test = np.array(list(set(range(npts)) - set(train)))

        dAdt = {"train": dAdt[train],
                "test": dAdt[test]}
        dBdt = {"train": dBdt[train],
                "test": dBdt[test]}
        featA = {"train": featA[train],
                "test": featA[test]}
        featB = {"train": featB[train],
                "test": featB[test]}

        return dAdt, dBdt, featA, featB, growth_rates

    def fit(self, train_pct,
            regressor="linear", alpha=0.1,
            smooth_ddt=False, ddt_minimum=0.0,
            consider_growth=True, growth_rates=[]):

        dAdt, dBdt, featA, featB, growth_rates = self.train_test_split(
            train_pct,
            ddt_minimum=ddt_minimum,
            consider_growth=consider_growth,
            growth_rates=growth_rates
        )

        # print("fitting...")
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

        fitA = regA.fit(featA["train"], dAdt["train"])
        fitB = regB.fit(featB["train"], dBdt["train"])

        return fitA, fitB, dAdt, dBdt, featA, featB