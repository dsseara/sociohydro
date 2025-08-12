import torch
from scipy.interpolate import interp1d
import numpy as np
import h5py
import json
from glob import glob

class SimulationDataset(torch.utils.data.Dataset):
    def __init__(self,
                path,
                seq_len=10,
                tmin=0,
                tmax=100):
        self.train = True
        self.path = path
        self.seq_len = seq_len
        self.tmin = tmin
        self.tmax = tmax
        self.init_data()
    
    def init_data(self):
        data_file = glob(f'{self.path}/*hdf5')
        param_file = glob(f'{self.path}/*json')

        with open(param_file[0]) as pfile:
            params = json.load(pfile)
        
        with h5py.File(data_file[0], 'r') as h5f:
            phiAB = np.zeros((len(h5f.keys()), 2, params['grid_size']))
            t = np.zeros(len(h5f.keys()))

            for ii, key in enumerate(h5f.keys()):
                phiAB[ii] = h5f[key]['state'][()]
                t[ii] = h5f[key]['sweep'][()]
            
        self.t = t * params['dt']
        self.dt = np.ptp(self.t) / len(self.t)
        self.x = np.linspace(0, 1, params['grid_size'])
        phiAB = phiAB / params['capacity']

        self.phiAB = interp1d(self.t, phiAB, axis=0, fill_value='extrapolate')
        print(self.t.min(), self.t.max(), self.t.shape)
        self.t = self.t[self.tmin:self.tmax]
    
    def __len__(self):
        return len(self.t) - self.seq_len - 1
    
    def __getitem__(self, idx):
        t = self.t[idx:idx+self.seq_len]

        sample = {
            'x': self.x,
            'dt': self.dt,
            't': t,
            'ab': torch.FloatTensor(self.phiAB(t)),
        }
        return sample