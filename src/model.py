import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.pool = nn.MaxPool2d(2, 2)
        self.down1 = ConvBlock(3, 64)
        self.down2 = ConvBlock(64, 128)
        self.down3 = ConvBlock(128, 256)
        self.down4 = ConvBlock(256, 512)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder
        self.up_trans_1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.up_conv_1 = ConvBlock(1024, 512)
        self.up_trans_2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.up_conv_2 = ConvBlock(512, 256)
        self.up_trans_3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up_conv_3 = ConvBlock(256, 128)
        self.up_trans_4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.up_conv_4 = ConvBlock(128, 64)

        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.down1(x)
        p1 = self.pool(c1)
        c2 = self.down2(p1)
        p2 = self.pool(c2)
        c3 = self.down3(p2)
        p3 = self.pool(c3)
        c4 = self.down4(p3)
        p4 = self.pool(c4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder (with Skip Connections)
        u1 = self.up_trans_1(b)
        x1 = torch.cat([u1, c4], dim=1)
        x1 = self.up_conv_1(x1)

        u2 = self.up_trans_2(x1)
        x2 = torch.cat([u2, c3], dim=1)
        x2 = self.up_conv_2(x2)

        u3 = self.up_trans_3(x2)
        x3 = torch.cat([u3, c2], dim=1)
        x3 = self.up_conv_3(x3)

        u4 = self.up_trans_4(x3)
        x4 = torch.cat([u4, c1], dim=1)
        x4 = self.up_conv_4(x4)

        return torch.sigmoid(self.out(x4))