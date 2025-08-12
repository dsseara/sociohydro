import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import gaussian_blur


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
                               padding='same', padding_mode='replicate', groups=in_channels)
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
    
'''
Neural network
'''
class CensusForecasting(nn.Module):
    def __init__(self,
                 input_dim=2,
                 output_dim=2,
                 kernel_size=7,
                 N_hidden=8,
                 hidden_size=64,
                 dropout_rate=0.1,
                 use_housing=False):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.N_hidden = N_hidden
        self.hidden_size = hidden_size
        self.use_housing = use_housing

        if self.use_housing:
            assert input_dim == output_dim + 1
    
        if hidden_size % input_dim != 0:
            self.read_in = nn.Sequential(
                nn.Conv2d(input_dim, 4, kernel_size=1),
                ConvNextBlock(4, hidden_size, kernel_size, dropout_rate)
            )
        else:
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

        # To be applied before nearest-neighbor interpolation on the mesh
        self.training_blur = transforms.GaussianBlur(kernel_size=self.kernel_size, sigma=(0.1, 3))
        self.inference_blur = transforms.GaussianBlur(kernel_size=self.kernel_size, sigma=1.) #1.5 for model_id=gridded

    def forward(self, x, housing=None):
        '''
        Predict time derivative given population distribution
        '''
        if self.use_housing:
            x = torch.cat([x, housing], dim=-3)

        x = self.read_in(x)
        for cell in self.cnn1:
            x = x + cell(x)
        latent = self.downsample(x)
        for cell in self.cnn2:
            latent = latent + cell(latent)
        latent = F.interpolate(latent, x.shape[-2:])
        x = torch.cat([x, latent], dim=1)
        x = self.read_out(x)

        if self.training:
            x = self.training_blur(x)
        else:
            x = self.inference_blur(x)

        return x
    
    def simulate(self, wb, n_steps=40, dt=1, housing=None):
        b, c, h, w = wb.shape
        preds = torch.zeros([b, n_steps, c, h, w], dtype=wb.dtype, device=wb.device)
        ddts = torch.zeros([b, n_steps, c, h, w], dtype=wb.dtype, device=wb.device)
        for tt in range(n_steps):
            ddts[:, tt] = self(wb, housing=housing) # Forward difference time stepping
            wb = wb + dt * ddts[:, tt]
            preds[:, tt] += wb
        
        return preds, ddts