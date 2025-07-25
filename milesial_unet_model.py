# UNET Model file

# Unified, cleaned, and fixed PyTorch U-Net + CA + ASPC code
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------- Shared Blocks ----------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
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
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# ---------------------------------- U-Net Base ----------------------------------
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


# ---------------------------------- ASPC Block ----------------------------------
class ASPC_Block(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_rate=0.3):
        super(ASPC_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=3, dilation=3)
        self.conv4 = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.projection = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None
        self.batch_norm = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        out = self.conv1(x) + self.conv2(x) + self.conv3(x) + self.conv4(x)
        residual = self.projection(x) if self.projection else x
        out = out + residual
        out = self.batch_norm(out)
        out = self.dropout(out)
        return self.relu(out)


# ---------------------------------- Attention ----------------------------------
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi


# ---------------------------------- APAU-Net ----------------------------------
class APAU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(APAU_Net, self).__init__()
        self.Maxpool = nn.MaxPool2d(2)
        self.Conv1 = ASPC_Block(img_ch, 64)
        self.Conv2 = ASPC_Block(64, 128)
        self.Conv3 = ASPC_Block(128, 256)
        self.Conv4 = ASPC_Block(256, 512)
        self.Conv5 = ASPC_Block(512, 1024)

        self.Up5 = Up(1024, 512)
        self.Att5 = AttentionBlock(512, 512, 256)
        self.Up_conv5 = ASPC_Block(1024, 512)

        self.Up4 = Up(512, 256)
        self.Att4 = AttentionBlock(256, 256, 128)
        self.Up_conv4 = ASPC_Block(512, 256)

        self.Up3 = Up(256, 128)
        self.Att3 = AttentionBlock(128, 128, 64)
        self.Up_conv3 = ASPC_Block(256, 128)

        self.Up2 = Up(128, 64)
        self.Att2 = AttentionBlock(64, 64, 32)
        self.Up_conv2 = ASPC_Block(128, 64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.Conv2(self.Maxpool(x1))
        x3 = self.Conv3(self.Maxpool(x2))
        x4 = self.Conv4(self.Maxpool(x3))
        x5 = self.Conv5(self.Maxpool(x4))

        d5 = self.Up5(x5, x4)
        d5 = self.Up_conv5(torch.cat([self.Att5(d5, x4), d5], dim=1))
        d4 = self.Up4(d5, x3)
        d4 = self.Up_conv4(torch.cat([self.Att4(d4, x3), d4], dim=1))
        d3 = self.Up3(d4, x2)
        d3 = self.Up_conv3(torch.cat([self.Att3(d3, x2), d3], dim=1))
        d2 = self.Up2(d3, x1)
        d2 = self.Up_conv2(torch.cat([self.Att2(d2, x1), d2], dim=1))

        return torch.sigmoid(self.Conv_1x1(d2))


# ---------------------------------- Neural Cellular Automata ----------------------------------
class NeuralCA(nn.Module):
    def __init__(self, num_channels=64):
        super(NeuralCA, self).__init__()
        self.perception = nn.Sequential(
            nn.Conv2d(num_channels, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.update = nn.Sequential(
            nn.Conv2d(128, num_channels, 1),
            nn.GroupNorm(8, num_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, steps=3):
        for _ in range(steps):
            dx = self.update(self.perception(x))
            x = x + dx
        return x


# ---------------------------------- U-Net + CA ----------------------------------
class UNetWithCA(nn.Module):
    def __init__(self, n_channels=3, ca_channels=64, n_classes=1, ca_steps=3):
        super(UNetWithCA, self).__init__()
        self.unet = UNet(n_channels, ca_channels)
        self.ca = NeuralCA(ca_channels)
        self.output_layer = nn.Conv2d(ca_channels, n_classes, kernel_size=1)
        self.ca_steps = ca_steps

    def forward(self, x):
        unet_out = self.unet(x)
        ca_out = self.ca(unet_out, steps=self.ca_steps)
        final = self.output_layer(ca_out + unet_out)
        return torch.sigmoid(final)
