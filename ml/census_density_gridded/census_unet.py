import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import gaussian_blur

def _init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.)

class ConvNextBlock(nn.Module):
    ''' Convolutional block, ignoring layernorm for now 
    '''
    def __init__(self, in_channels, out_channels, kernel_size=7, dropout_rate=0.1):
        super().__init__()

        self.depth_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                            padding='same', padding_mode='replicate', groups=in_channels)
        self.conv1 = nn.Conv2d(out_channels, 4*out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(4*out_channels, out_channels, kernel_size=1)
        
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.batch_norm(x)
        x = self.conv1(x)
        x = torch.sin(x)
        x = self.conv2(x)
        
        x = self.dropout(x)
        return x

class UNetEncoder(nn.Module):
    """ UNet encoder - preserves outputs and sends across latent bottleneck
    """
    def __init__(self,
                input_dims=3,
                stage_dims=8,
                blocks_per_stage=2,
                num_stages=3,
                conv_stride=2):
        super().__init__()
        self.downsample_blocks = nn.ModuleList()
        self.stages = nn.ModuleList()

        self.read_in = nn.Sequential(
            nn.Conv2d(input_dims, stage_dims, kernel_size=3, padding='same'),
            *[ConvNextBlock(stage_dims, stage_dims) for j in range(blocks_per_stage)])

        for i in range(num_stages):
            stage = nn.Sequential(
                nn.BatchNorm2d(stage_dims),
                nn.Conv2d(stage_dims, stage_dims*2, kernel_size=conv_stride, stride=conv_stride)
            )
            self.downsample_blocks.append(stage)
            stage_dims = stage_dims * 2
            
            stage = nn.ModuleList([ConvNextBlock(stage_dims, stage_dims) for j in range(blocks_per_stage)])
            self.stages.append(stage)

    def forward(self, x):
        encoder_outputs = []
        x = self.read_in(x)
        for i in range(len(self.stages)):
            encoder_outputs.append(x)
            x = self.downsample_blocks[i](x)
            for cell in self.stages[i]:
                x = x + cell(x) # Residual connection
        return x, encoder_outputs

class UNetDecoder(nn.Module):
    """ UNet Decoder - accepts incoming skip connections
    """
    def __init__(self,
                output_dims, 
                stage_dims=8,
                blocks_per_stage=2,
                num_stages=3,
                conv_stride=2):
        super().__init__()
        self.read_out = nn.Conv2d(stage_dims, output_dims, kernel_size=1)

        self.upsample_blocks = nn.ModuleList()
        self.combiners = nn.ModuleList()
        self.stages = nn.ModuleList()

        stage_dims = stage_dims * (2**num_stages)

        for i in range(num_stages):
            stage = nn.Sequential(
                nn.BatchNorm2d(stage_dims),
                nn.ConvTranspose2d(stage_dims, stage_dims//2, kernel_size=conv_stride, stride=conv_stride)
            )
            self.upsample_blocks.append(stage)
            stage_dims = stage_dims // 2

            self.combiners.append(nn.Conv2d(2*stage_dims, stage_dims, kernel_size=1))

            stage = nn.ModuleList([ConvNextBlock(stage_dims, stage_dims) for j in range(blocks_per_stage)])
            self.stages.append(stage)
    
    def forward(self, x, encoder_outputs):
        for i in range(len(self.stages)):
            x = self.upsample_blocks[i](x)
            x2 = encoder_outputs[-(i+1)]
            diffY = x2.size()[-2] - x.size()[-2]
            diffX = x2.size()[-1] - x.size()[-1]
            x = torch.nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                                            diffY // 2, diffY - diffY // 2])
            x = torch.cat([x, x2], dim=-3)
            x = self.combiners[i](x)
            for cell in self.stages[i]:
                x = x + cell(x) # Residual connection
        x = self.read_out(x)
        return x

class CensusForecastingUNet(torch.nn.Module):
    """ UNet - encoder decoder architecture with skip connections
        across latent bottleneck.
        See if this does better forecasting than the other model
    """
    def __init__(self,
                input_dim=2,
                output_dim=2,
                stage_dims=16,
                kernel_size=7,
                blocks_per_stage=3,
                num_stages=2,
                conv_stride=4,
                use_housing=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.stage_dims = stage_dims
        self.blocks_per_stage = blocks_per_stage
        self.num_stages = num_stages
        self.conv_stride = conv_stride
        self.use_housing = use_housing
        self.kernel_size = kernel_size

        if self.use_housing:
            assert input_dim == output_dim + 1

        self.encoder = UNetEncoder(input_dim, stage_dims, blocks_per_stage, num_stages, conv_stride)
        self.decoder = UNetDecoder(output_dim, stage_dims, blocks_per_stage, num_stages, conv_stride)
        self.apply(_init_weights)

        # To be applied before nearest-neighbor interpolation on the mesh
        self.training_blur = transforms.GaussianBlur(kernel_size=self.kernel_size, sigma=(0.1, 3))
        self.inference_blur = transforms.GaussianBlur(kernel_size=self.kernel_size, sigma=1.)

    def forward(self, x, housing=None):
        """ Predict time derivative given population distribution
        """
        if self.use_housing:
            x = torch.cat([x, housing], dim=-3)

        _, _, h0, w0 = x.shape
        x, en_outputs = self.encoder(x) #[B, C, H, W]
        y = self.decoder(x, en_outputs)

        if self.training:
            y = self.training_blur(y)
        else:
            y = self.inference_blur(y)

        return y

    def simulate(self, wb, n_steps=40, dt=1, housing=None):
        b, c, h, w = wb.shape
        preds = torch.zeros([b, n_steps, c, h, w], dtype=wb.dtype, device=wb.device)
        for tt in range(n_steps):
            wb = wb + dt * self(wb, housing=housing) # Forward difference time stepping
            preds[:, tt] += wb
        
        return preds