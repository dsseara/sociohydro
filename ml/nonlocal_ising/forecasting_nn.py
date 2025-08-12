import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import h5py
from tqdm.auto import trange, tqdm

'''
Building blocks
'''
class Sin(nn.Module):
	def forward(self, x):
		return torch.sin(x)

class ConvNextBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=7, dropout_rate=0.):
		super().__init__()

		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
							   padding='same', padding_mode='circular', groups=in_channels)
		self.conv2 = nn.Conv2d(out_channels, 4*out_channels, kernel_size=1)
		self.conv3 = nn.Conv2d(4*out_channels, out_channels, kernel_size=1)
		
		self.dropout = nn.Dropout(p=dropout_rate)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = torch.sin(x)
		x = self.conv3(x)
		
		x = self.dropout(x)
		return x
	
class SimulationForecasting(nn.Module):
	'''
	Since the simulations are on a regular grid, we can use a standard 
	residual network to predict the next frame
	'''
	def __init__(self,
				 input_dim=1,
				 output_dim=1,
				 kernel_size=7,
				 N_hidden=8,
				 hidden_size=64,
				 dropout_rate=0.1,
				 smooth=False):
		super().__init__()
		
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.kernel_size = kernel_size
		self.N_hidden = N_hidden
		self.hidden_size = hidden_size
		self.smooth = smooth
	
		self.read_in = ConvNextBlock(input_dim, hidden_size, kernel_size, dropout_rate)

		self.downsample = nn.Sequential(
			nn.Conv2d(hidden_size, hidden_size, kernel_size=4, stride=4),
			nn.GELU()
		)
		self.read_out = nn.Conv2d(2*hidden_size, output_dim, kernel_size=1)
		
		self.cnn1 = nn.ModuleList()
		self.cnn2 = nn.ModuleList()
		for i in range(N_hidden):
			self.cnn1.append(ConvNextBlock(hidden_size, hidden_size, kernel_size))
			self.cnn2.append(ConvNextBlock(hidden_size, hidden_size, kernel_size))
			
		self.device = torch.device('cpu')		 
	
	def batch_step(self, sample):
		phi1 = self.simulate(sample['phi'][:, 0], sample['phi'].shape[1]-1)

		if self.smooth:
			loss = F.l1_loss(phi1, sample['phi'][:, 1:])
		else:
			# Convert input and target to class probabilities
			loss = F.binary_cross_entropy(
				0.5 * (phi1 + 1),
				0.5 * (sample['phi'][:, 1:] + 1),
			)

		return loss
	
	def forward(self, phi0):		
		x = self.read_in(phi0)
		for cell in self.cnn1:
			x = x + cell(x)
		latent = self.downsample(x)
		for cell in self.cnn2:
			latent = latent + cell(latent)
		latent = F.interpolate(latent, x.shape[-2:])
		x = torch.cat([x, latent], dim=1)
		x = self.read_out(x)
		return x
	
	def simulate(self, phi, steps):
		b, c, h, w = phi.shape
		preds = torch.zeros([b, steps, c, h, w], dtype=phi.dtype, device=phi.device)
		for tt in range(steps):
			phi = phi + self(phi)
			phi = F.tanh(phi) # Bound from -1 to 1
			preds[:, tt] += phi
									
		return preds
