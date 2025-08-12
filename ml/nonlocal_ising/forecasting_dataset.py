import torch
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import numpy as np
import h5py
import json
from glob import glob

class SimulationDataset(torch.utils.data.Dataset):
	def __init__(self,
				path,
				seq_len=5,
				sigma=0,
				tmin=1,
				tmax=-1):
		self.train = True
		self.path = path
		self.seq_len = seq_len
		self.sigma = sigma
		self.tmin = tmin
		self.tmax = tmax
		self.init_data()
	
	def init_data(self):
		data_file = f'{self.path}/dataset.hdf5'
		param_file = f'{self.path}/params.json'

		with open(param_file) as pfile:
			params = json.load(pfile)

		with h5py.File(data_file, 'r') as h5f:
			phi = None
			t = np.zeros(len(h5f.keys()))
			for ii, key in enumerate(h5f.keys()):
				if phi is None:
					phi = np.zeros((len(h5f.keys()), 1, *h5f[key]['state'].shape))
				if self.sigma > 0:
					phi[ii] = gaussian_filter(h5f[key]['state'].astype(float), sigma=self.sigma, mode='wrap')
				else:
					phi[ii] = h5f[key]['state']
				t[ii] = h5f[key]['sweep'][()]
		
		# Grid parameters
		self.grid_size = phi.shape[-2:]
		x = np.linspace(0, 1, self.grid_size[0])
		y = np.linspace(0, 1, self.grid_size[1])
		self.x, self.y = np.meshgrid(x, y)

		self.t = t / params['skip'] # Units of simulation output frequency, they're all N=500
		self.dt = 1

		self.phi = interp1d(self.t, phi, axis=0, fill_value='extrapolate')
		self.t = self.t[self.tmin:self.tmax]

		
	def __len__(self):
		return len(self.t) - self.seq_len - 1
	
	def __getitem__(self, idx):
		t = self.t[idx:idx+self.seq_len]

		sample = {
			'x': self.x,
			'y': self.y, 
			'dt': self.dt,
			't': t,
			'phi': torch.FloatTensor(self.phi(t)),
		}
		return sample
