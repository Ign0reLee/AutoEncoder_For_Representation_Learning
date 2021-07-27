import torch
import torch.nn as nn

from .layers import *

class netE(nn.Module):
    def __init__(self, in_channels, nker=32, norm="bnrom"):
        super(netE, self).__init__()

        # Input [B, 1280, 960, 3]
        self.enc1 = CBR2d(in_channels=in_channels, out_channels= 1 * nker, norm=norm)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # [B, 640, 480, nker]
        self.enc2 = CBR2d(in_channels=1 * nker, out_channels= 2 * nker, norm=norm)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # [B, 320, 240, 2 * nker]
        self.enc3 = CBR2d(in_channels=2 * nker, out_channels= 4 * nker, norm=norm)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # [B, 160, 120, 4 * nker]
        self.enc4 = CBR2d(in_channels=4 * nker, out_channels= 8 * nker, norm=norm)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # [B, 80, 60, 8 * nker]
        self.enc5 = CBR2d(in_channels=8 * nker, out_channels= 16 * nker, norm=norm)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        # [B, 40, 30, 16 * nker]
        self.enc6 = CBR2d(in_channels=16 * nker, out_channels= 32 * nker, norm=norm)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        # [B, 20, 15, 32 * nker]
    
    def forward(self, x):

        h = self.enc1(x)
        h = self.pool1(h)
        h = self.enc2(h)
        h = self.pool2(h)
        h = self.enc3(h)
        h = self.pool3(h)
        h = self.enc4(h)
        h = self.pool4(h)
        h = self.enc5(h)
        h = self.pool5(h)
        h = self.enc6(h)
        out = self.pool6(h)

        return out

class netD(nn.Module):

    def __init__(self, out_channels, nker=32, norm="bnrom"):
        super(netD, self).__init__()

        # [B, 20, 15, 32 * nker]
        self.dec1 = DECBR2d(in_channels=32 * nker, out_channels= 16 * nker, norm=norm)
        # [B, 40, 30, 16 * nker]
        self.dec2 = DECBR2d(in_channels=16 * nker, out_channels= 8 * nker, norm=norm)
        # [B, 80, 60, 8 * nker]
        self.dec3 = DECBR2d(in_channels=8 * nker, out_channels= 4 * nker, norm=norm)
        # [B, 160, 120, 4 * nker]
        self.dec4 = DECBR2d(in_channels=4 * nker, out_channels= 2 * nker, norm=norm)
        # [B, 320, 240, 2 * nker]
        self.dec5 = DECBR2d(in_channels=2 * nker, out_channels= 1 * nker, norm=norm)
        # [B, 640, 480, 1 * nker]
        self.dec6 = DECBR2d(in_channels=1 * nker, out_channels= out_channels, norm=norm, relu=None)
        # [B, 1280, 960, 3]
    
    def forward(self, x):
        h = self.dec1(x)
        h = self.dec2(h)
        h = self.dec3(h)
        h = self.dec4(h)
        h = self.dec5(h)
        h = self.dec6(h)
        out = torch.sigmoid(h)

        return out

class AutoEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, nker=32, norm="bnrom"):
        super(AutoEncoder, self).__init__()

        self.netE = netE(in_channels=in_channels, nker=nker, norm=norm).to("cuda:0")
        self.netD = netD(out_channels=out_channels, nker=nker, norm=norm).to("cuda:1")
    
    def forward(self, x):

        latent_vector = self.netE(x)
        output = self.netD(latent_vector.to("cuda:1"))
        return output

        