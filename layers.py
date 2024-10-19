import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from math import ceil, log2
from collections import OrderedDict

NORM_LAYERS = { 'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln': nn.LayerNorm }

### NETWORK LAYERS ###
class Conv2DwithBN(nn.Module):
    def __init__(self, in_fea, out_fea, 
                kernel_size=3, stride=1, padding=1,
                bn=True, relu_slop=0.2, dropout=None):
        super(Conv2DwithBN,self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if bn:
            layers.append(nn.BatchNorm2d(num_features=out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.Conv2DwithBN = nn.Sequential(*layers)

    def forward(self, x):
        return self.Conv2DwithBN(x)

class ResizeConv2DwithBN(nn.Module):
    def __init__(self, in_fea, out_fea, scale_factor=2, mode='nearest'):
        super(ResizeConv2DwithBN, self).__init__()
        layers = [nn.Upsample(scale_factor=scale_factor, mode=mode)]
        layers.append(nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(num_features=out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.ResizeConv2DwithBN = nn.Sequential(*layers)

    def forward(self, x):
        return self.ResizeConv2DwithBN(x)
 
class Conv2DwithBN_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1):
        super(Conv2DwithBN_Tanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        layers.append(nn.BatchNorm2d(num_features=out_fea))
        layers.append(nn.Tanh())
        self.Conv2DwithBN = nn.Sequential(*layers)

    def forward(self, x):
        return self.Conv2DwithBN(x)

class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slop=0.2, dropout=None):
        super(ConvBlock,self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvBlock_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn'):
        super(ConvBlock_Tanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=2, stride=2, padding=0, output_padding=0, norm='bn'):
        super(DeconvBlock, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ResizeBlock(nn.Module):
    def __init__(self, in_fea, out_fea, scale_factor=2, mode='nearest', norm='bn'):
        super(ResizeBlock, self).__init__()
        layers = [nn.Upsample(scale_factor=scale_factor, mode=mode)]
        layers.append(nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=3, stride=1, padding=1))
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
# fourier blocks tests

# class FourierConvLayer(nn.Module):
#     def __init__(self, in_channels, out_channels, modes1, modes2):
#         super(FourierConvLayer, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.modes1 = modes1
#         self.modes2 = modes2
#         self.scale = (1 / (in_channels * out_channels))
#         self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
#         self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        
#         # Adicionando a convolução 1x1 - polarizer
#         self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def compl_mul2d(self, input, weights):
#         # (batch, in_channel, x, y), (in_channel, out_channel, x, y)
#         return torch.einsum("bixy,ioxy->boxy", input, weights)

#     def forward(self, x):
#         batchsize = x.shape[0]
        
#         # Caminho 1: Transformada de Fourier
#         x_ft = torch.fft.rfftn(x, dim=[-2, -1])
        
#         out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2+1, dtype=torch.cfloat, device=x.device)
#         out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
#         out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
#         # Retornando ao domínio espacial
#         out1 = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)))
        
#         # Caminho 2: Convolução 1x1
#         out2 = self.conv1x1(x)
        
#         # Somando os dois caminhos
#         out = out1 + out2
        
#         return out


#### UNET BLOCKS ####
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
#######################################
    
class LargeUFourierConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, bilinear=False):
        super(LargeUFourierConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2        
        
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        
        # Caminho 2: Transformada linear
        self.linear = nn.Linear(in_channels, out_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        
        # Caminho 3: U-Net

        self.bilinear = bilinear

        self.inc = (DoubleConv(in_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, out_channels))

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def add_padding(self, x, min_size):
        original_size = x.shape[-2:]
        target_size = [max(2**ceil(log2(s)), min_size) for s in original_size]
        
        padding = []
        for orig, target in zip(original_size, target_size):
            pad_total = target - orig
            pad_start = 1  # Sempre adiciona 1 de padding no início
            pad_end = pad_total - pad_start
            padding.extend([pad_start, pad_end])
        
        #  ta como [top, bottom, left, right]
        # left, right, top and bottom on F.pad documentation
        padding = [padding[2], padding[3], padding[0], padding[1]]
        padded_x = F.pad(x, padding)
        return padded_x, original_size

    def remove_padding(self, padded_x, original_size):
        # Removemos o padding começando da posição (1,1)
        return padded_x[..., 1:1+original_size[0], 1:1+original_size[1]]

    def forward(self, x):
        batchsize = x.shape[0]

        # Caminho 2: transformada linear
        x = x.permute(0, 2, 3, 1)
        # x.shape == [batch_size, grid_size, grid_size, in_dim]
        res = self.linear(x)
        # res.shape == [batch_size, grid_size, grid_size, out_dim]
        out2 = res.permute(0, 3, 1, 2)

        x = rearrange(x, 'b m n i -> b i m n')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]
        
        # Caminho 1: Transformada de Fourier
        x_ft = torch.fft.rfftn(x, dim=[-2, -1])
        
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2+1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        out1 = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)))
        
    
        
        # Caminho 3: U-Net simplificada maior
        # Calcula o padding necessário para alcançar a próxima potência de 2 (min: [16,16])
        # aplica primeiro 1 em top e left, o resto right e bottom (para inputs assimetricos)


        x_padded, original_size = self.add_padding(x,min_size=16)

        enc1 = self.inc(x_padded)
        enc2 = self.down1(enc1)
        enc3 = self.down2(enc2)
        enc4 = self.down3(enc3)
        enc5 = self.down4(enc4)
        dec = self.up1(enc5, enc4)
        dec = self.up2(dec, enc3)
        dec = self.up3(dec, enc2)
        dec = self.up4(dec,enc1)
        logits = self.outc(dec)

        # Remove o padding
        out3 = self.remove_padding(logits, original_size=original_size)
    
        # Combinação dos caminhos
        out = self.act1(out1+out2)
        out = self.act2(out+out3)
        #out = out1 + out2 + out3
        
        
        return out
    
class UFourierConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, bilinear=False):
        super(UFourierConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2        
        
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        
        # Caminho 2: Transformada linear
        self.linear = nn.Linear(in_channels, out_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        
        # Caminho 3: U-Net

        self.bilinear = bilinear

        self.inc = (DoubleConv(in_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        factor = 2 if bilinear else 1
        self.down3 = (Down(256, 512 // factor))
        self.up1 = (Up(512, 256 // factor, bilinear))
        self.up2 = (Up(256, 128 // factor, bilinear))
        self.up3 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, out_channels))

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def add_padding(self, x, min_size):
        original_size = x.shape[-2:]
        target_size = [max(2**ceil(log2(s)), min_size) for s in original_size]
        
        padding = []
        for orig, target in zip(original_size, target_size):
            pad_total = target - orig
            pad_start = 1  # Sempre adiciona 1 de padding no início
            pad_end = pad_total - pad_start
            padding.extend([pad_start, pad_end])
        
        #  ta como [top, bottom, left, right]
        # left, right, top and bottom on F.pad documentation
        padding = [padding[2], padding[3], padding[0], padding[1]]
        padded_x = F.pad(x, padding)
        return padded_x, original_size

    def remove_padding(self, padded_x, original_size):
        # Removemos o padding começando da posição (1,1)
        return padded_x[..., 1:1+original_size[0], 1:1+original_size[1]]

    def forward(self, x):
        batchsize = x.shape[0]

        # Caminho 2: transformada linear
        x = x.permute(0, 2, 3, 1)
        # x.shape == [batch_size, grid_size, grid_size, in_dim]
        res = self.linear(x)
        # res.shape == [batch_size, grid_size, grid_size, out_dim]
        out2 = res.permute(0, 3, 1, 2)

        x = rearrange(x, 'b m n i -> b i m n')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]
        
        # Caminho 1: Transformada de Fourier
        x_ft = torch.fft.rfftn(x, dim=[-2, -1])
        
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2+1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        out1 = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)))
        
        # Caminho 3: U-Net simplificada maior
        # Calcula o padding necessário para alcançar a próxima potência de 2 (min: [8,8])
        # aplica primeiro 1 em top e left, o resto right e bottom (para inputs assimetricos)


        x_padded, original_size = self.add_padding(x,min_size=8)

        enc1 = self.inc(x_padded)
        enc2 = self.down1(enc1)
        enc3 = self.down2(enc2)
        enc4 = self.down3(enc3)
        dec = self.up1(enc4, enc3)
        dec = self.up2(dec, enc2)
        dec = self.up3(dec, enc1)
        logits = self.outc(dec)

        # Remove o padding
        out3 = self.remove_padding(logits, original_size=original_size)
    
        # Combinação dos caminhos
        out = self.act1(out1+out2)
        out = self.act2(out+out3)
        #out = out1 + out2 + out3
        
        
        return out
    
class SmallUFourierConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, bilinear=False):
        super(SmallUFourierConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2        
        
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        
        # Caminho 2: Transformada linear
        self.linear = nn.Linear(in_channels, out_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        
        # Caminho 3: U-Net

        self.bilinear = bilinear

        self.inc = (DoubleConv(in_channels, 64))
        self.down1 = (Down(64, 128))
        factor = 2 if bilinear else 1
        self.down2 = (Down(128, 256 // factor))
        self.up1 = (Up(256, 128 // factor, bilinear))
        self.up2 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, out_channels))

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def add_padding(self, x, min_size):
        original_size = x.shape[-2:]
        target_size = [max(2**ceil(log2(s)), min_size) for s in original_size]
        
        padding = []
        for orig, target in zip(original_size, target_size):
            pad_total = target - orig
            pad_start = 1  # Sempre adiciona 1 de padding no início
            pad_end = pad_total - pad_start
            padding.extend([pad_start, pad_end])
        
        #  ta como [top, bottom, left, right]
        # left, right, top and bottom on F.pad documentation
        padding = [padding[2], padding[3], padding[0], padding[1]]
        padded_x = F.pad(x, padding)
        return padded_x, original_size

    def remove_padding(self, padded_x, original_size):
        # Removemos o padding começando da posição (1,1)
        return padded_x[..., 1:1+original_size[0], 1:1+original_size[1]]

    def forward(self, x):
        batchsize = x.shape[0]

        # Caminho 2: transformada linear
        x = x.permute(0, 2, 3, 1)
        # x.shape == [batch_size, grid_size, grid_size, in_dim]
        res = self.linear(x)
        # res.shape == [batch_size, grid_size, grid_size, out_dim]
        out2 = res.permute(0, 3, 1, 2)

        x = rearrange(x, 'b m n i -> b i m n')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]
        
        # Caminho 1: Transformada de Fourier
        x_ft = torch.fft.rfftn(x, dim=[-2, -1])
        
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2+1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        out1 = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)))

        # Caminho 3: U-Net simplificada maior
        # Calcula o padding necessário para alcançar a próxima potência de 2 (min: [4,4])
        # aplica primeiro 1 em top e left, o resto right e bottom (para inputs assimetricos)


        x_padded, original_size = self.add_padding(x,min_size=4)

        enc1 = self.inc(x_padded)
        enc2 = self.down1(enc1)
        enc3 = self.down2(enc2)
        dec = self.up1(enc3, enc2)
        dec = self.up2(dec, enc1)
        logits = self.outc(dec)

        # Remove o padding
        out3 = self.remove_padding(logits, original_size=original_size)
    
        # Combinação dos caminhos
        out = self.act1(out1+out2)
        out = self.act2(out+out3)
        #out = out1 + out2 + out3
        
        
        return out
    
class UFourierConvLayer_conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, bilinear=False):
        super(UFourierConvLayer_conv1x1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

        # Camadas de ativação para FILM
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        
        # Caminho 2: Convolução 1x1
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        
        # Caminho 3: U-Net

        self.bilinear = bilinear

        self.inc = (DoubleConv(in_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        factor = 2 if bilinear else 1
        self.down3 = (Down(256, 512 // factor))
        self.up1 = (Up(512, 256 // factor, bilinear))
        self.up2 = (Up(256, 128 // factor, bilinear))
        self.up3 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, out_channels))

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def add_padding(self, x, min_size):
        original_size = x.shape[-2:]
        target_size = [max(2**ceil(log2(s)), min_size) for s in original_size]
        
        padding = []
        for orig, target in zip(original_size, target_size):
            pad_total = target - orig
            pad_start = 1  # Sempre adiciona 1 de padding no início
            pad_end = pad_total - pad_start
            padding.extend([pad_start, pad_end])
        
        #  ta como [top, bottom, left, right]
        # left, right, top and bottom on F.pad documentation
        padding = [padding[2], padding[3], padding[0], padding[1]]
        padded_x = F.pad(x, padding)
        return padded_x, original_size

    def remove_padding(self, padded_x, original_size):
        # Removemos o padding começando da posição (1,1)
        return padded_x[..., 1:1+original_size[0], 1:1+original_size[1]]

    def forward(self, x):
        batchsize = x.shape[0]
        
        # Caminho 1: Transformada de Fourier
        x_ft = torch.fft.rfftn(x, dim=[-2, -1])
        
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2+1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        out1 = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)))
        
        # Caminho 2: Convolução 1x1
        out2 = self.conv1x1(x)
        
        # Caminho 3: U-Net simplificada maior
        # Calcula o padding necessário para alcançar a próxima potência de 2 (min: [8,8])
        # aplica primeiro 1 em top e left, o resto right e bottom (para inputs assimetricos)


        x_padded, original_size = self.add_padding(x,min_size=8)

        enc1 = self.inc(x_padded)
        enc2 = self.down1(enc1)
        enc3 = self.down2(enc2)
        enc4 = self.down3(enc3)
        dec = self.up1(enc4, enc3)
        dec = self.up2(dec, enc2)
        dec = self.up3(dec, enc1)
        logits = self.outc(dec)

        # Remove o padding
        out3 = self.remove_padding(logits, original_size=original_size)
    
        # Combinação dos caminhos
        out = self.act1(out1+out2)
        out = self.act2(out+out3)
        
        return out
    
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
    
class SpectralConv2d_res(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_res, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.linear = nn.Linear(in_channels, out_channels)
        self.act = nn.ReLU(inplace=True)

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):

        x = x.permute(0, 2, 3, 1)
        # x.shape == [batch_size, grid_size, grid_size, in_dim]
        res = self.linear(x)
        # res.shape == [batch_size, grid_size, grid_size, out_dim]

        x = rearrange(x, 'b m n i -> b i m n')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]


        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)


        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

        x = rearrange(x, 'b i m n -> b m n i')
        x = self.act(x + res)
        x = x.permute(0, 3, 1, 2)

        return x

PRIMITIVES = [
    'skip_connect',
    'spectral_conv2d',
    'spectral_conv2d_res',
    'conv_block',
    'double_conv',
    'large_u_fourier',
    'u_fourier',
    'small_u_fourier',
]

class MixedOp(nn.Module):
    def __init__(self, C_in, C_out, stride, modes1, modes2):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            if primitive == 'skip_connect':
                op = Identity() if C_in == C_out else FactorizedReduce(C_in, C_out)
            elif primitive == 'spectral_conv2d':
                op = SpectralConv2d(C_in, C_out, modes1, modes2)
            elif primitive == 'spectral_conv2d_res':
                op = SpectralConv2d_res(C_in, C_out, modes1, modes2)
            elif primitive == 'conv_block':
                op = ConvBlock(C_in, C_out)
            elif primitive == 'double_conv':
                op = DoubleConv(C_in,C_out)
            elif primitive == 'large_u_fourier':
                op = LargeUFourierConvLayer(C_in, C_out, modes1, modes2)
            elif primitive == 'u_fourier':
                op = UFourierConvLayer(C_in, C_out, modes1, modes2)
            elif primitive == 'small_u_fourier':
                op = SmallUFourierConvLayer(C_in, C_out, modes1, modes2)
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))

class Cell(nn.Module):
    def __init__(self, steps, C_prev, C, modes1, modes2):
        super(Cell, self).__init__()
        self.preprocess = ReLUConvBN(C_prev, C, 1, 1, 0)
        self._steps = steps
        self._ops = nn.ModuleList()
        for i in range(self._steps):
            op = MixedOp(C, C, stride=1, modes1=modes1, modes2=modes2)
            self._ops.append(op)

    def forward(self, s0, weights):
        s0 = self.preprocess(s0)
        states = [s0]
        for i in range(self._steps):
            s = self._ops[i](states[-1], weights[i])
            states.append(s)
        return states[-1]

class DARTSBlock(nn.Module):
    def __init__(self, C_in, C_out, modes1, modes2, steps=4):
        super(DARTSBlock, self).__init__()
        self.cell = Cell(steps, C_in, C_out, modes1, modes2)
        self.alphas = nn.Parameter(torch.randn(steps, len(PRIMITIVES)))

    def forward(self, x):
        weights = F.softmax(self.alphas, dim=-1)
        return self.cell(x, weights)

class Identity(nn.Module):
    def forward(self, x):
        return x

class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out):
        super(FactorizedReduce, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out)

    def forward(self, x):
        out = torch.cat([self.conv1(x), self.conv2(x[:,:,1:,1:])], dim=1)
        out = self.bn(out)
        return out

class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out)
        )

    def forward(self, x):
        return self.op(x)