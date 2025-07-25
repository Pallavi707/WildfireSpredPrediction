import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------- CA Module (Post-processing) ---------------------
class PostProcessingCA(nn.Module):
    def __init__(self, channels=64, ca_steps=5, dropout_p=0.1):
        super(PostProcessingCA, self).__init__()
        self.ca_steps = ca_steps

        self.perception = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p)
        )

        self.update_conv = nn.Conv2d(128, channels, kernel_size=1)
        self.norm_layer = nn.GroupNorm(8, channels)  # GroupNorm to avoid size issues
        self.tanh = nn.Tanh()

    def forward(self, x):
        for _ in range(self.ca_steps):
            dx = self.update_conv(self.perception(x))
            dx = self.norm_layer(dx)
            dx = self.tanh(dx)
            x = x + dx
        return x


# --------------------- U-Net Building Blocks ---------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class WildfireUNet(nn.Module):
    def __init__(self, in_channels=19, hidden_channels=64):
        super(WildfireUNet, self).__init__()
        self.inc = DoubleConv(in_channels, hidden_channels)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(hidden_channels, hidden_channels)
        )

        # Only one down-up path to maintain 64x64 spatial size
        self.up1 = nn.Sequential(
            # No upsampling needed since input is 64x64
            DoubleConv(2 * hidden_channels, hidden_channels)
        )

    def forward(self, x):
        x1 = self.inc(x)        # (B, 64, 64, 64)
        x2 = self.down1(x1)     # (B, 64, 32, 32)

        x = self.up1(torch.cat([
            x1,
            F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        ], dim=1))  # Already same size, no upsampling needed


        return x  # Final output: (B, 64, 64, 64)


# --------------------- Combined Model ---------------------
class UNetWithPostCA(nn.Module):
    def __init__(self, in_channels=19, hidden_channels=64, out_channels=1, ca_steps=5):
        super(UNetWithPostCA, self).__init__()
        self.unet = WildfireUNet(in_channels=in_channels, hidden_channels=hidden_channels)
        self.ca = PostProcessingCA(channels=hidden_channels, ca_steps=ca_steps)
        self.decoder = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        input_spatial_shape = x.shape[2:]  # (H, W)
        x = self.unet(x)                   # (B, 64, 64, 64)
        x = self.ca(x)                     # (B, 64, 64, 64)
        x = self.decoder(x)                # (B, 1, 64, 64)
        x = self.activation(x)

        # Sanity check to ensure shape correctness
        expected_shape = (64, 64)
        assert x.shape[2:] == expected_shape, f"Expected {expected_shape}, got {x.shape[2:]}"

        return x
