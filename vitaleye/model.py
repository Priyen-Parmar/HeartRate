"""VitalNet - rPPG network for VitalEye v2.

Design (drawn from the literature):
  * DiffNormalized motion stream + Standardized appearance stream
    (DeepPhys, Chen & McDuff ECCV'18; TS-CAN, Liu et al. NeurIPS'20; rPPG-Toolbox NeurIPS'23)
  * soft spatial attention masks computed from appearance, applied to the motion stream
    -> learns where the skin/pulse is; replaces the old hard YCbCr mask (mask flicker = noise)
  * 3D convolutions (PhysNet, Yu et al. BMVC'19) for spatio-temporal features
  * dilated temporal residual head with full temporal resolution (no pooling in time)
  * input normalisation lives INSIDE forward(): training and live inference cannot diverge.
    DiffNormalized (x_t+1 - x_t)/(x_t+1 + x_t) is invariant to global gain -> low-light robust.

Input : [B, 3, T, H, W] RGB float in [0, 1], uniform 30 Hz, 72x72 face crop
Output: [B, T] rPPG waveform (arbitrary scale)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def temporal_smooth(x: torch.Tensor) -> torch.Tensor:
    """Fixed [1,2,1]/4 temporal low-pass (replicate padded). Gain cos^2(pi f / 30):
    0.90 at 3 Hz (180 BPM), 0.97 at 1.5 Hz, 0 at 15 Hz -> removes most per-frame sensor noise
    (which dominates in low light) while keeping the pulse band."""
    xp = torch.cat([x[:, :, :1], x, x[:, :, -1:]], dim=2)
    return 0.25 * xp[:, :, :-2] + 0.5 * xp[:, :, 1:-1] + 0.25 * xp[:, :, 2:]


def normalize_inputs(x: torch.Tensor, prefilter: bool = False) -> torch.Tensor:
    """x: [B,3,T,H,W] in [0,1] -> [B,6,T,H,W] (diff-normalized motion, standardized appearance)."""
    x = x.float()
    eps = 2.0 / 255.0
    xm = temporal_smooth(x) if prefilter else x
    d = (xm[:, :, 1:] - xm[:, :, :-1]) / (xm[:, :, 1:] + xm[:, :, :-1] + eps)
    d = torch.cat([torch.zeros_like(d[:, :, :1]), d], dim=2)
    d = d / (d.flatten(1).std(dim=1).view(-1, 1, 1, 1, 1) + 1e-6)
    d = d.clamp(-6.0, 6.0)
    m = x.mean(dim=(2, 3, 4), keepdim=True)
    s = x.std(dim=(2, 3, 4), keepdim=True)
    a = (x - m) / (s + 1e-4)
    return torch.cat([d, a], dim=1)


class ConvBlock3d(nn.Module):
    def __init__(self, cin, cout, k=(3, 3, 3), dil_t=1):
        super().__init__()
        pad = (dil_t * (k[0] // 2), k[1] // 2, k[2] // 2)
        self.conv = nn.Conv3d(cin, cout, k, padding=pad, dilation=(dil_t, 1, 1), bias=False)
        self.bn = nn.BatchNorm3d(cout)

    def forward(self, x):
        return F.elu(self.bn(self.conv(x)))


class Attention(nn.Module):
    """DeepPhys-style soft mask: sigmoid(1x1 conv) normalised to unit mean (x0.5)."""

    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv3d(c, 1, 1)

    def forward(self, a):
        m = torch.sigmoid(self.conv(a))
        H, W = m.shape[-2:]
        norm = m.sum(dim=(3, 4), keepdim=True) + 1e-6
        return m * (H * W) / (2.0 * norm)


class TemporalRes(nn.Module):
    def __init__(self, c, dil, drop):
        super().__init__()
        self.c1 = nn.Conv1d(c, c, 5, padding=2 * dil, dilation=dil, bias=False)
        self.b1 = nn.BatchNorm1d(c)
        self.c2 = nn.Conv1d(c, c, 1, bias=False)
        self.b2 = nn.BatchNorm1d(c)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        y = F.elu(self.b1(self.c1(x)))
        y = self.b2(self.c2(self.drop(y)))
        return F.elu(x + y)


class RegionBranch(nn.Module):
    """Linear-first region integration with learned maximal-ratio combining (new in version 4).

    Problem it solves: in low light the per-pixel pulse is ~40 dB below sensor noise. The pixel path
    applies BN+ELU non-linearities at full resolution, i.e. BEFORE enough pixels are averaged, so the
    weak pulse is partly destroyed. Here pixels are first averaged LINEARLY (noise / sqrt(N)) inside
    soft-skin-weighted regions on 6x6, 3x3 and 1x1 grids (multi-scale, cf. STMaps of RhythmNet,
    Niu et al. TIP'20), using the appearance attention of the pixel path as the skin weight.
    Each region trace is diff-normalised (gain invariant), whitened by its own noise scale, processed
    by a shared dilated 1-D CNN, and the regions are fused with time-varying softmax reliability
    weights conditioned on each region's relative noise level - a learned version of the
    maximal-ratio combining of DistancePPG (Kumar et al. BOE'15).
    All averaging runs in float32 (bf16 has 0.4 % precision, larger than the pulse itself)."""

    GRIDS = (6, 3, 1)

    def __init__(self, c: int = 32):
        super().__init__()
        self.tconv = nn.Sequential(
            nn.Conv1d(4, c, 5, padding=2, bias=False), nn.BatchNorm1d(c), nn.ELU(),
            nn.Conv1d(c, c, 5, padding=4, dilation=2, bias=False), nn.BatchNorm1d(c), nn.ELU(),
            nn.Conv1d(c, c, 5, padding=8, dilation=4, bias=False), nn.BatchNorm1d(c), nn.ELU())
        self.score = nn.Conv1d(c, 1, 9, padding=4)

    def forward(self, x, mask):
        """x: [B,3,T,H,W] in [0,1]; mask: [B,1,T,H/2,W/2] soft skin weight (constant on 2x2 pixel blocks)."""
        B, C, T, H, W = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float(); mask = mask.float()
            h, w = mask.shape[-2:]
            # 2x2 pixel means (exactly equivalent to a nearest-upsampled mask), then block sums by reshape
            xs = x.view(B, C, T, h, 2, w, 2).mean((4, 6))
            g0 = self.GRIDS[0]
            num = (xs * mask).view(B, C, T, g0, h // g0, g0, w // g0).sum((4, 6))     # [B,3,T,6,6]
            den = mask.view(B, 1, T, g0, h // g0, g0, w // g0).sum((4, 6))
            tr = []
            for g in self.GRIDS:
                k = g0 // g
                n_ = num.view(B, C, T, g, k, g, k).sum((4, 6)); d_ = den.view(B, 1, T, g, k, g, k).sum((4, 6))
                tr.append((n_ / (d_ + 1e-6)).flatten(3))
            r = torch.cat(tr, -1)                                         # [B,3,T,R]
            d = (r[:, :, 1:] - r[:, :, :-1]) / (r[:, :, 1:] + r[:, :, :-1] + 1e-3)
            d = torch.cat([torch.zeros_like(d[:, :, :1]), d], 2)
            sc = d.std(dim=(1, 2), keepdim=True)                         # [B,1,1,R] noise scale
            lvl = torch.log(sc + 1e-7)
            dn = (d / (sc + 1e-7)).clamp(-6, 6)
            rel = (lvl - lvl.mean(-1, keepdim=True)).expand(B, 1, T, r.shape[-1])
            z = torch.cat([dn, rel], 1)                                  # [B,4,T,R]
        R = z.shape[-1]
        z = z.permute(0, 3, 1, 2).reshape(B * R, 4, T)
        f = self.tconv(z)                                                # [B*R,c,T]
        w = torch.softmax(self.score(f).view(B, R, T).float(), dim=1)    # reliability over regions
        f = (f.view(B, R, -1, T).float() * w.unsqueeze(2)).sum(1)        # [B,c,T]
        return f, lvl.mean(dim=(1, 2, 3))                                # pooled, mean log-noise


def pixel_noise_level(x: torch.Tensor) -> torch.Tensor:
    """log relative per-pixel temporal noise: std of 2nd temporal difference / mean intensity. [B]"""
    x = x.float()
    d2 = x[:, :, 2:] - 2 * x[:, :, 1:-1] + x[:, :, :-2]
    return torch.log(d2.flatten(1).std(1) / (x.flatten(1).mean(1) + 1e-4) + 1e-7)


class VitalNet(nn.Module):
    """version 1: spatial-only motion stem.  version 3: temporal [1,2,1] pre-filter before
    differencing + 3x5x5 motion stem.  version 4: version 1 + RegionBranch + noise-gated fusion
    (the network measures its own input noise level and decides how much to trust the
    pixel path vs the linear-first region path)."""

    def __init__(self, c: int = 32, drop: float = 0.2, version: int = 1):
        super().__init__()
        self.version = version
        self.prefilter = version == 3
        kt = 3 if version == 3 else 1
        # stems (stride 2 in space: 72 -> 36)
        self.m_stem = nn.Sequential(nn.Conv3d(3, c // 2, (kt, 5, 5), stride=(1, 2, 2), padding=(kt // 2, 2, 2), bias=False),
                                    nn.BatchNorm3d(c // 2), nn.ELU())
        self.a_stem = nn.Sequential(nn.Conv3d(3, c // 2, (1, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2), bias=False),
                                    nn.BatchNorm3d(c // 2), nn.ELU())
        # stage 1 @36x36
        self.m1 = nn.Sequential(ConvBlock3d(c // 2, c), ConvBlock3d(c, c))
        self.a1 = nn.Sequential(ConvBlock3d(c // 2, c, k=(1, 3, 3)), ConvBlock3d(c, c, k=(1, 3, 3)))
        self.att1 = Attention(c)
        # stage 2 @18x18
        self.m2 = nn.Sequential(ConvBlock3d(c, 2 * c), ConvBlock3d(2 * c, 2 * c))
        self.a2 = nn.Sequential(ConvBlock3d(c, 2 * c, k=(1, 3, 3)), ConvBlock3d(2 * c, 2 * c, k=(1, 3, 3)))
        self.att2 = Attention(2 * c)
        # stage 3 @9x9
        self.m3 = nn.Sequential(ConvBlock3d(2 * c, 2 * c, dil_t=2), ConvBlock3d(2 * c, 2 * c, dil_t=4))
        self.pool = nn.AvgPool3d((1, 2, 2))
        self.drop3d = nn.Dropout3d(drop * 0.5)
        # temporal head
        ch = 2 * c
        if version >= 4:
            self.region = RegionBranch(c)
            self.gate = nn.Sequential(nn.Linear(2, 16), nn.ELU(), nn.Linear(16, 2))
            self.fuse = nn.Sequential(nn.Conv1d(ch + c, ch, 1, bias=False), nn.BatchNorm1d(ch), nn.ELU())
        self.head =nn.Sequential(TemporalRes(ch, 1, drop), TemporalRes(ch, 2, drop),
                                  TemporalRes(ch, 4, drop), TemporalRes(ch, 8, drop))
        self.out = nn.Conv1d(ch, 1, 1)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        z = normalize_inputs(x, self.prefilter)
        m, a = z[:, :3], z[:, 3:]
        m, a = self.m_stem(m), self.a_stem(a)
        m, a = self.m1(m), self.a1(a)
        g1 = self.att1(a)
        m = self.pool(m * g1); a = self.pool(a)
        m, a = self.m2(m), self.a2(a)
        g2 = self.att2(a)
        m = self.pool(m * g2)
        m = self.drop3d(self.m3(m))
        f = m.mean(dim=(3, 4))  # [B, C, T]
        if self.version >= 4:
            fr, n_reg = self.region(x, g1)                                             # g1: 36x36 skin weight
            n_pix = pixel_noise_level(x)
            g = 2 * torch.sigmoid(self.gate(torch.stack([n_pix, n_reg], 1)))         # [B,2], 1 at init-ish
            f = self.fuse(torch.cat([f * g[:, :1, None], fr.to(f.dtype) * g[:, 1:, None]], 1))
        y = self.out(self.head(f)).squeeze(1)
        y = y - y.mean(dim=1, keepdim=True)
        if return_attention:
            return y, g1
        return y


# ----------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------
def neg_pearson(pred, target):
    p = pred - pred.mean(dim=1, keepdim=True)
    t = target - target.mean(dim=1, keepdim=True)
    r = (p * t).sum(1) / (p.norm(dim=1) * t.norm(dim=1) + 1e-8)
    return (1 - r).mean()


class SpectralKL(nn.Module):
    """KL(label spectrum || predicted spectrum) on a fine 1-BPM grid (39-192 BPM) computed with
    an explicit DFT matrix (resolution independent of clip length). Inspired by the frequency
    cross-entropy of PhysFormer (Yu et al. CVPR'22) but uses the full label spectrum, so HR
    changes inside a clip are handled gracefully."""

    def __init__(self, fs=30.0, lo_bpm=39, hi_bpm=192):
        super().__init__()
        self.fs = fs
        self.register_buffer("bpm", torch.arange(lo_bpm, hi_bpm + 1, dtype=torch.float32))
        self._cache = {}

    def _basis(self, T, device):
        key = (T, str(device))
        if key not in self._cache:
            t = torch.arange(T, device=device, dtype=torch.float32) / self.fs
            f = self.bpm.to(device) / 60.0
            ang = 2 * torch.pi * t[:, None] * f[None, :]
            win = torch.hann_window(T, periodic=False, device=device)[:, None]
            self._cache[key] = (torch.cos(ang) * win, torch.sin(ang) * win)
        return self._cache[key]

    def psd(self, x):
        c, s = self._basis(x.shape[1], x.device)
        x = x.float()
        return (x @ c) ** 2 + (x @ s) ** 2

    def forward(self, pred, target):
        pp = self.psd(pred)
        tp = self.psd(target)
        p = pp / (pp.sum(1, keepdim=True) + 1e-8)
        q = tp / (tp.sum(1, keepdim=True) + 1e-8)
        return (q * (torch.log(q + 1e-8) - torch.log(p + 1e-8))).sum(1).mean()


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
