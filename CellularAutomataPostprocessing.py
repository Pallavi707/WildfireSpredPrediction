import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- Basic U-Net Blocks (LeeJunHyun-style) -------------------- #

class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)

# -------------------- Stable Feature-space Neural CA (Bottleneck) -------------------- #

class FeaturePerception(nn.Module):
    """
    Depthwise perception over C channels using fixed 3x3 filters:
    Sobel-x, Sobel-y, Identity. Output has 3C channels.
    """
    def __init__(self, channels: int):
        super().__init__()
        # normalized Sobel to avoid large gradients
        kx = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=torch.float32) / 8.0
        ky = torch.tensor([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=torch.float32) / 8.0
        id3 = torch.tensor([[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]], dtype=torch.float32)
        self.register_buffer("kx", kx.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("ky", ky.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("ki", id3.view(1, 1, 3, 3), persistent=False)
        self.channels = channels

    def forward(self, x):
        C = self.channels
        gx = F.conv2d(x, self.kx.expand(C, 1, 3, 3), padding=1, groups=C)
        gy = F.conv2d(x, self.ky.expand(C, 1, 3, 3), padding=1, groups=C)
        gi = F.conv2d(x, self.ki.expand(C, 1, 3, 3), padding=1, groups=C)
        return torch.cat([gx, gy, gi], dim=1)  # [B, 3C, H, W]

class FeatureCAStable(nn.Module):
    """
    Stable Neural CA update in feature space for bottleneck usage.
    Uses GroupNorm, tanh-bounded updates, zero-init last layer, and a small learnable step size.
    """
    def __init__(self, channels: int, hidden: int = 128, steps: int = 3, groups: int = 16, dropout_p: float = 0.0):
        super().__init__()
        self.steps = steps
        self.perc = FeaturePerception(channels)

        g_hidden = min(groups, hidden) if hidden % min(groups, hidden) == 0 else 1
        g_out    = min(groups, channels) if channels % min(groups, channels) == 0 else 1

        self.update = nn.Sequential(
            nn.Conv2d(3 * channels, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(g_hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.GroupNorm(g_out, channels)  # keep bottleneck stats stable
        )
        # near-identity start: zero the last conv so initial dx≈0
        nn.init.zeros_(self.update[-2].weight)
        nn.init.zeros_(self.update[-2].bias)

        # small learnable step size
        self.alpha = nn.Parameter(torch.tensor(0.1), requires_grad=True)

    def forward(self, x):
        # iterate a few small, safe steps
        for _ in range(self.steps):
            y  = self.perc(x)
            dx = self.update(y)
            dx = torch.tanh(dx)           # bound the update
            x  = x + self.alpha * dx
            x  = torch.nan_to_num(x)      # kill accidental NaNs/Infs
        return x

# -------------------- U-Net with CA only in the bottleneck -------------------- #

class UNet1_CA(nn.Module):
    """
    A LeeJunHyun-style U-Net with a stable Neural CA inserted in the bottleneck.
    API matches U_Net(img_ch, output_ch).
    """
    def __init__(self, img_ch=3, output_ch=1, ca_hidden=128, ca_steps=3, ca_dropout_p=0.0):
        super().__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(ch_in=img_ch, ch_out=64)
        self.Conv2 = ConvBlock(ch_in=64,   ch_out=128)
        self.Conv3 = ConvBlock(ch_in=128,  ch_out=256)
        self.Conv4 = ConvBlock(ch_in=256,  ch_out=512)
        self.Conv5 = ConvBlock(ch_in=512,  ch_out=1024)

        # ---- CA at bottleneck ----
        self.ca = FeatureCAStable(channels=1024, hidden=ca_hidden, steps=ca_steps, dropout_p=ca_dropout_p)

        self.Up5     = UpConv(ch_in=1024, ch_out=512)
        self.Up_conv5= ConvBlock(ch_in=1024, ch_out=512)

        self.Up4     = UpConv(ch_in=512,  ch_out=256)
        self.Up_conv4= ConvBlock(ch_in=512,  ch_out=256)

        self.Up3     = UpConv(ch_in=256,  ch_out=128)
        self.Up_conv3= ConvBlock(ch_in=256,  ch_out=128)

        self.Up2     = UpConv(ch_in=128,  ch_out=64)
        self.Up_conv2= ConvBlock(ch_in=128,  ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoder
        x1 = self.Conv1(x)
        x2 = self.Conv2(self.Maxpool(x1))
        x3 = self.Conv3(self.Maxpool(x2))
        x4 = self.Conv4(self.Maxpool(x3))
        x5 = self.Conv5(self.Maxpool(x4))

        # bottleneck CA
        x5 = self.ca(x5)

        # decoder
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        return d1
