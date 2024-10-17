import numpy as np
from sklearn import linear_model as lm
from scipy import stats, optimize
import fipy as fp
from infer_utils import *

class SociohydroInfer_fipy():
    def __init__(self, var1, var2, mesh, t, t_dim = 0):
        self.mesh = mesh
        self.var1 = np.moveaxis(np.asarray(var1), t_dim, 0)
        self.var2 = np.moveaxis(np.asarray(var2), t_dim, 0)
        self.ts = t
        self.nt
        if (len(self.var1) != self.nt) or (len(var2) != self.nt):
            raise ValueError("var1 and var2 need to have same number of time points")
        self.ncell = mesh.numberOfCells
        self.nt = len(t)
        

    def calc_features(self, nfeat=7):
        """
        features are calculated at each time step, and come in order
        [Ti, kii, kij, nuiii, nuiij, nuijj, Γi]
        """
        feat1 = np.zeros(self.nt, self.ncell, nfeat)
        feat2 = np.zeros(self.nt, self.ncell, nfeat)
        for tidx, (v1, v2) in enumerate(zip(self.var1, self.var2)):
            feat1[tidx], feat2[tidx] = calc_fipyTerms(
                fp.CellVariable(mesh=self.mesh, value=v1),
                fp.CellVariable(mesh=self.mesh, value=v2)
                )
            
        return feat1, feat2
    
    def train_test_split(self, train_pct):
        pass

            