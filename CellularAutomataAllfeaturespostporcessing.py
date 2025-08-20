import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- Stable Feature-space Neural CA (Bottleneck) -------------------- #
class FeaturePerception(nn.Module):
    """
    Depthwise perception over C channels using fixed 3x3 filters:
    Sobel-x, Sobel-y, Identity. Output has 3C channels.
    """
    def __init__(self, channels: int):
        super().__init__()
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
    Randomizes steps during training (optional) without changing the public API.
    """
    def __init__(self, channels: int, hidden: int = 128, steps: int = 3, groups: int = 16,
                 dropout_p: float = 0.0, rand_steps: bool = True, min_steps: int = 2, max_steps: int = 8):
        super().__init__()
        self.steps = steps
        self.rand_steps = rand_steps
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.perc = FeaturePerception(channels)

        def _gn_groups(nc, g):
            g = min(g, nc)
            return g if nc % g == 0 else 1

        g_hidden = _gn_groups(hidden, groups)
        g_out    = _gn_groups(channels, groups)

        self.update = nn.Sequential(
            nn.Conv2d(3 * channels, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(g_hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.GroupNorm(g_out, channels)
        )
        # near-identity start: zero the last conv so initial dx≈0
        nn.init.zeros_(self.update[-2].weight)
        nn.init.zeros_(self.update[-2].bias)

        # small learnable step size
        self.alpha = nn.Parameter(torch.tensor(0.1), requires_grad=True)

    def _num_steps(self):
        if self.training and self.rand_steps:
            return int(torch.randint(self.min_steps, self.max_steps + 1, (1,)).item())
        return self.steps

    def forward(self, x):
        T = self._num_steps()
        for _ in range(T):
            y  = self.perc(x)
            dx = self.update(y)
            dx = torch.tanh(dx)           # bound the update
            x  = x + self.alpha * dx
            x  = torch.nan_to_num(x)      # kill accidental NaNs/Infs
        return x

# -------------------- Logit-level CA Refiner (post-decoder) -------------------- #
class LogitCARefiner(nn.Module):
    """
    Tiny CA that edits 1-channel logits directly.
    Steps randomized during training for robustness; fixed at eval.
    """
    def __init__(self, steps: int = 6, hidden: int = 32, groups: int = 8,
                 rand_steps: bool = True, min_steps: int = 3, max_steps: int = 10):
        super().__init__()
        self.steps = steps
        self.rand_steps = rand_steps
        self.min_steps = min_steps
        self.max_steps = max_steps

        kx = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=torch.float32) / 8.0
        ky = torch.tensor([[-1,-2,-1],
                           [ 0, 0, 0],
                           [ 1, 2, 1]], dtype=torch.float32) / 8.0
        id3 = torch.tensor([[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]], dtype=torch.float32)
        self.register_buffer("kx", kx.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("ky", ky.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("ki", id3.view(1, 1, 3, 3), persistent=False)

        def _gn_groups(nc, g):
            g = min(g, nc)
            return g if nc % g == 0 else 1

        self.update = nn.Sequential(
            nn.Conv2d(3, hidden, 1, bias=False),
            nn.GroupNorm(_gn_groups(hidden, groups), hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, bias=True),
        )
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)

        self.alpha = nn.Parameter(torch.tensor(0.25), requires_grad=True)

    def _num_steps(self):
        if self.training and self.rand_steps:
            return int(torch.randint(self.min_steps, self.max_steps + 1, (1,)).item())
        return self.steps

    def forward(self, logits):
        x = logits
        T = self._num_steps()
        for _ in range(T):
            gx = F.conv2d(x, self.kx, padding=1)
            gy = F.conv2d(x, self.ky, padding=1)
            gi = F.conv2d(x, self.ki, padding=1)
            y  = torch.cat([gx, gy, gi], dim=1)
            dx = torch.tanh(self.update(y))
            x  = x + self.alpha * dx
            x  = torch.nan_to_num(x)
        return x

# -------------------- U²-Net building blocks (RSU) -------------------- #
class REBNCONV(nn.Module):
    def __init__(self, in_ch, out_ch, dirate=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dirate, dilation=dirate, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

def _upsample_like(x, ref):
    # avoid subtle misalignment
    return F.interpolate(x, size=ref.shape[2:], mode='bilinear', align_corners=False)

class RSU7(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, 1); self.pool1 = nn.MaxPool2d(2,2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, 1); self.pool2 = nn.MaxPool2d(2,2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, 1); self.pool3 = nn.MaxPool2d(2,2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, 1); self.pool4 = nn.MaxPool2d(2,2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, 1); self.pool5 = nn.MaxPool2d(2,2, ceil_mode=True)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, 1)

        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch*2, mid_ch, 1)
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, 1)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, 1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, 1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, 1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, 1)

        self.rebnconvout = REBNCONV(out_ch*2, out_ch, 1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin); hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx);   hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx);   hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx);   hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx);   hx = self.pool5(hx5)
        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx6, _upsample_like(hx7, hx6)), 1))
        hx5d = self.rebnconv5d(torch.cat((hx5, _upsample_like(hx6d, hx5)), 1))
        hx4d = self.rebnconv4d(torch.cat((hx4, _upsample_like(hx5d, hx4)), 1))
        hx3d = self.rebnconv3d(torch.cat((hx3, _upsample_like(hx4d, hx3)), 1))
        hx2d = self.rebnconv2d(torch.cat((hx2, _upsample_like(hx3d, hx2)), 1))
        hx1d = self.rebnconv1d(torch.cat((hx1, _upsample_like(hx2d, hx1)), 1))

        hxout = self.rebnconvout(torch.cat((hxin, hx1d), 1))
        return hxout

class RSU6(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, 1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, 1); self.pool1 = nn.MaxPool2d(2,2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, 1); self.pool2 = nn.MaxPool2d(2,2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, 1); self.pool3 = nn.MaxPool2d(2,2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, 1); self.pool4 = nn.MaxPool2d(2,2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, 1)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, 1)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, 1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, 1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, 1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, 1)

        self.rebnconvout = REBNCONV(out_ch*2, out_ch, 1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin); hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx);   hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx);   hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx);   hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx5, _upsample_like(hx6, hx5)),1))
        hx4d = self.rebnconv4d(torch.cat((hx4, _upsample_like(hx5d, hx4)),1))
        hx3d = self.rebnconv3d(torch.cat((hx3, _upsample_like(hx4d, hx3)),1))
        hx2d = self.rebnconv2d(torch.cat((hx2, _upsample_like(hx3d, hx2)),1))
        hx1d = self.rebnconv1d(torch.cat((hx1, _upsample_like(hx2d, hx1)),1))

        hxout = self.rebnconvout(torch.cat((hxin, hx1d),1))
        return hxout

class RSU5(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, 1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, 1); self.pool1 = nn.MaxPool2d(2,2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, 1); self.pool2 = nn.MaxPool2d(2,2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, 1); self.pool3 = nn.MaxPool2d(2,2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, 1)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, 1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, 1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, 1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, 1)

        self.rebnconvout = REBNCONV(out_ch*2, out_ch, 1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin); hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx);   hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx);   hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx4, _upsample_like(hx5, hx4)),1))
        hx3d = self.rebnconv3d(torch.cat((hx3, _upsample_like(hx4d, hx3)),1))
        hx2d = self.rebnconv2d(torch.cat((hx2, _upsample_like(hx3d, hx2)),1))
        hx1d = self.rebnconv1d(torch.cat((hx1, _upsample_like(hx2d, hx1)),1))

        hxout = self.rebnconvout(torch.cat((hxin, hx1d),1))
        return hxout

class RSU4(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, 1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, 1); self.pool1 = nn.MaxPool2d(2,2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, 1); self.pool2 = nn.MaxPool2d(2,2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, 1)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, 1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, 1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, 1)

        self.rebnconvout = REBNCONV(out_ch*2, out_ch, 1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin); hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx);   hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx3, _upsample_like(hx4, hx3)),1))
        hx2d = self.rebnconv2d(torch.cat((hx2, _upsample_like(hx3d, hx2)),1))
        hx1d = self.rebnconv1d(torch.cat((hx1, _upsample_like(hx2d, hx1)),1))

        hxout = self.rebnconvout(torch.cat((hxin, hx1d),1))
        return hxout

class RSU4F(nn.Module):
    """RSU4F: same spatial scale, dilated conv pyramid (no pooling)."""
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, 1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

        self.rebnconvout = REBNCONV(out_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx3, hx4),1))
        hx2d = self.rebnconv2d(torch.cat((hx2, hx3d),1))
        hx1d = self.rebnconv1d(torch.cat((hx1, hx2d),1))

        hxout = self.rebnconvout(torch.cat((hxin, hx1d),1))
        return hxout

# -------------------- U²-Net with CA in the bottleneck + Logit CA -------------------- #
class U2Net_CA(nn.Module):
    """
    U²-Net backbone with:
      (1) Feature-space CA in the bottleneck (randomized steps during training),
      (2) Logit-level CA refiner at the very end (also randomized during training).

    API stays the same: forward(x) -> logits of shape [B, output_ch, H, W].
    You can optionally call forward_with_sides(x) to also get side outputs for deep supervision.
    """
    def __init__(self,
                 img_ch=3,
                 output_ch=1,
                 ca_hidden=128,
                 ca_steps=3,
                 ca_dropout_p=0.0,
                 base_ch=64,
                 # CA randomization
                 ca_rand_steps=True, ca_min_steps=2, ca_max_steps=8,
                 # Logit CA settings
                 use_logit_ca=True, logit_ca_steps=6, logit_ca_hidden=32, logit_ca_groups=8,
                 logit_ca_rand=True, logit_ca_min_steps=3, logit_ca_max_steps=10,
                 # Deep supervision wiring (kept optional)
                 deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.use_logit_ca = use_logit_ca

        # ---------------- Encoder ----------------
        self.stage1 = RSU7(img_ch,       base_ch,      base_ch)        # -> C=64
        self.pool12 = nn.MaxPool2d(2,2, ceil_mode=True)

        self.stage2 = RSU6(base_ch,      base_ch,      base_ch*2)      # -> C=128
        self.pool23 = nn.MaxPool2d(2,2, ceil_mode=True)

        self.stage3 = RSU5(base_ch*2,    base_ch,      base_ch*4)      # -> C=256
        self.pool34 = nn.MaxPool2d(2,2, ceil_mode=True)

        self.stage4 = RSU4(base_ch*4,    base_ch*2,    base_ch*8)      # -> C=512
        self.pool45 = nn.MaxPool2d(2,2, ceil_mode=True)

        self.stage5 = RSU4(base_ch*8,    base_ch*2,    base_ch*8)      # -> C=512
        self.pool56 = nn.MaxPool2d(2,2, ceil_mode=True)

        self.stage6 = RSU4F(base_ch*8,   base_ch*2,    base_ch*8)      # -> C=512

        # ---------------- CA bottleneck ----------------
        self.ca = FeatureCAStable(
            channels=base_ch*8, hidden=ca_hidden, steps=ca_steps, dropout_p=ca_dropout_p,
            rand_steps=ca_rand_steps, min_steps=ca_min_steps, max_steps=ca_max_steps
        )

        # ---------------- Decoder ----------------
        self.stage5d = RSU4F(base_ch*16, base_ch*2,    base_ch*8)
        self.stage4d = RSU4(base_ch*16,  base_ch*2,    base_ch*4)
        self.stage3d = RSU5(base_ch*8,   base_ch,      base_ch*2)
        self.stage2d = RSU6(base_ch*4,   base_ch,      base_ch)
        self.stage1d = RSU7(base_ch*2,   base_ch,      base_ch)

        # ---------------- Output heads ----------------
        if self.deep_supervision:
            self.side1 = nn.Conv2d(base_ch,    output_ch, 3, padding=1)
            self.side2 = nn.Conv2d(base_ch,    output_ch, 3, padding=1)
            self.side3 = nn.Conv2d(base_ch*2,  output_ch, 3, padding=1)
            self.side4 = nn.Conv2d(base_ch*4,  output_ch, 3, padding=1)
            self.side5 = nn.Conv2d(base_ch*8,  output_ch, 3, padding=1)
            self.side6 = nn.Conv2d(base_ch*8,  output_ch, 3, padding=1)
            self.outconv = nn.Conv2d(6*output_ch, output_ch, 1)
        else:
            self.outconv = nn.Conv2d(base_ch, output_ch, 1)

        # ---------------- Logit CA (post-decoder) ----------------
        self.logit_ca = LogitCARefiner(
            steps=logit_ca_steps, hidden=logit_ca_hidden, groups=logit_ca_groups,
            rand_steps=logit_ca_rand, min_steps=logit_ca_min_steps, max_steps=logit_ca_max_steps
        ) if use_logit_ca else None

    # ---- Helper to compute head logits and (optionally) side logits ----
    def _decode_and_heads(self, x):
        # Encoder
        hx1 = self.stage1(x);  hx = self.pool12(hx1)
        hx2 = self.stage2(hx); hx = self.pool23(hx2)
        hx3 = self.stage3(hx); hx = self.pool34(hx3)
        hx4 = self.stage4(hx); hx = self.pool45(hx4)
        hx5 = self.stage5(hx); hx = self.pool56(hx5)
        hx6 = self.stage6(hx)

        # CA bottleneck
        hx6 = self.ca(hx6)

        # Decoder
        hx6up = _upsample_like(hx6, hx5)
        hx5d  = self.stage5d(torch.cat((hx6up, hx5), 1))

        hx5dup = _upsample_like(hx5d, hx4)
        hx4d   = self.stage4d(torch.cat((hx5dup, hx4), 1))

        hx4dup = _upsample_like(hx4d, hx3)
        hx3d   = self.stage3d(torch.cat((hx4dup, hx3), 1))

        hx3dup = _upsample_like(hx3d, hx2)
        hx2d   = self.stage2d(torch.cat((hx3dup, hx2), 1))

        hx2dup = _upsample_like(hx2d, hx1)
        hx1d   = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # Final logits at input size
        if self.deep_supervision:
            d1 = self.side1(hx1d)
            d2 = _upsample_like(self.side2(hx2d), x)
            d3 = _upsample_like(self.side3(hx3d), x)
            d4 = _upsample_like(self.side4(hx4d), x)
            d5 = _upsample_like(self.side5(hx5d), x)
            d6 = _upsample_like(self.side6(hx6),  x)
            d1u = _upsample_like(d1, x)
            d0  = self.outconv(torch.cat((d1u, d2, d3, d4, d5, d6), 1))
            return d0, (d1u, d2, d3, d4, d5, d6)
        else:
            d0 = self.outconv(hx1d)
            d0 = _upsample_like(d0, x)
            return d0, None

    # ---- Public API: unchanged shape/return ----
    def forward(self, x):
        d0, _ = self._decode_and_heads(x)
        # Logit CA refiner (edits final logits), preserves shape
        if self.use_logit_ca and self.logit_ca is not None:
            d0 = self.logit_ca(d0)
        return d0

    # ---- Optional: get side logits too (for deep supervision) ----
    @torch.no_grad()
    def forward_with_sides(self, x):
        """
        Returns (final_logits, side_logits_tuple or None).
        Does not change the default forward() API.
        """
        d0, sides = self._decode_and_heads(x)
        if self.use_logit_ca and self.logit_ca is not None:
            d0 = self.logit_ca(d0)
        return d0, sides
