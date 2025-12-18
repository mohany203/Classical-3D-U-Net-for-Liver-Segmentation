import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
    """
    (Conv3D -> BN -> ReLU) * 2
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # First Conv
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            # Second Conv
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ClassicalUNet3D(nn.Module):
    """
    Standard 3D U-Net architecture.
    """
    def __init__(self, n_channels=1, n_classes=1):
        super().__init__()
        
        # --- ENCODER ---
        self.inc = DoubleConv3D(n_channels, 32)       
        self.down1 = DoubleConv3D(32, 64)             
        self.down2 = DoubleConv3D(64, 128)             
        self.down3 = DoubleConv3D(128, 256)
        self.down4 = DoubleConv3D(256, 512)
        
        # --- BOTTLENECK ---
        self.bottleneck = DoubleConv3D(512, 1024)
        
        # --- DECODER ---
        self.up1 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv3D(1024, 512) 
        
        self.up2 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv3D(512, 256)
        
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv3D(256, 128)
        
        self.up4 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv3D(128, 64)

        self.up5 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.conv_up5 = DoubleConv3D(64, 32)
        
        self.outc = nn.Conv3d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(F.max_pool3d(x1, 2))
        x3 = self.down2(F.max_pool3d(x2, 2))
        x4 = self.down3(F.max_pool3d(x3, 2))
        x5 = self.down4(F.max_pool3d(x4, 2))
        
        # Bottleneck
        x_bot = self.bottleneck(F.max_pool3d(x5, 2))
        
        # Decoder
        
        # Up 1
        x = self.up1(x_bot)
        if x.shape != x5.shape:
             x = F.interpolate(x, size=x5.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x5, x], dim=1)
        x = self.conv_up1(x)
        
        # Up 2
        x = self.up2(x)
        if x.shape != x4.shape:
             x = F.interpolate(x, size=x4.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x4, x], dim=1)
        x = self.conv_up2(x)
        
        # Up 3
        x = self.up3(x)
        if x.shape != x3.shape:
             x = F.interpolate(x, size=x3.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x3, x], dim=1)
        x = self.conv_up3(x)

        # Up 4
        x = self.up4(x)
        if x.shape != x2.shape:
             x = F.interpolate(x, size=x2.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x2, x], dim=1)
        x = self.conv_up4(x)

        # Up 5
        x = self.up5(x)
        if x.shape != x1.shape:
             x = F.interpolate(x, size=x1.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x1, x], dim=1)
        x = self.conv_up5(x)
        
        return self.outc(x)
