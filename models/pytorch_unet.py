import torch
import torch.nn as nn

class PyTorchUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=5):
        super().__init__()
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.middle = self.conv_block(256, 512)

        self.up4 = self.up_block(512, 256)
        self.up3 = self.up_block(256, 128)
        self.up2 = self.up_block(128, 64)
        self.up1 = self.up_block(64, 32)

        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        m = self.middle(self.pool(e4))

        d4 = self.up4(m)
        d4 = self.enc4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.enc3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.enc2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.enc1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)