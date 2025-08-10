import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# -------------------- Core UNet components -------------------- #
class DoubleConv(nn.Module):
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# -------------------- Neural Cellular Automata w/ Filters -------------------- #
class Perception(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.filters = nn.Parameter(torch.stack([
            self.make_sobel_x(),
            self.make_sobel_y(),
            torch.eye(3).reshape(1, 1, 3, 3)[0]  # Identity
        ], dim=0), requires_grad=False)
        self.in_channels = in_channels

    def forward(self, x):
        outputs = []
        for f in self.filters:
            f = f.expand(self.in_channels, 1, 3, 3)
            outputs.append(F.conv2d(x, f, padding=1, groups=self.in_channels))
        return torch.cat(outputs, dim=1)

    def make_sobel_x(self):
        return torch.tensor([[[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]]], dtype=torch.float32)

    def make_sobel_y(self):
        return torch.tensor([[[-1, -2, -1],
                              [0, 0, 0],
                              [1, 2, 1]]], dtype=torch.float32)


class NeuralCA(nn.Module):
    def __init__(self, feature_channels=19, n_steps=3, hidden_dim=64, alpha_threshold=0.3):
        super(NeuralCA, self).__init__()
        self.n_steps = n_steps
        self.alpha_channel = feature_channels  # last channel is alpha
        self.alpha_threshold = alpha_threshold
        self.perception = Perception(feature_channels + 1)  # +1 for alpha
        self.update = nn.Sequential(
            nn.Conv2d((feature_channels + 1) * 3, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, feature_channels + 1, kernel_size=1)
        )

    def forward(self, x):
        for _ in range(self.n_steps):
            x = checkpoint(self.step, x, use_reentrant=False)
        return torch.clamp(x[:, self.alpha_channel:self.alpha_channel+1, :, :], 0.0, 1.0)

    def step(self, x):
        y = self.perception(x)
        dx = self.update(y)
        alpha = x[:, self.alpha_channel:self.alpha_channel+1, :, :]
        living_mask = (alpha > self.alpha_threshold).float()
        return x + dx * living_mask


# -------------------- UNet with Post-Processing CA -------------------- #
class UNetWithPostCA19(nn.Module):
    def __init__(self, n_channels, n_classes, feature_channels=19):
        super(UNetWithPostCA19, self).__init__()
        self.unet = UNet1(n_channels, n_classes)
        self.ca = NeuralCA(feature_channels=feature_channels)

    def forward(self, x):
        logits = self.unet(x)
        probs = torch.sigmoid(logits)

        if self.training:
            probs.requires_grad_()

        # Concatenate original 19 features and 1 alpha channel (from U-Net)
        ca_input = torch.cat([x, probs.clone()], dim=1)
        refined = self.ca(ca_input)
        return refined


# -------------------- Base UNet Definition -------------------- #
class UNet1(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet1, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

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
