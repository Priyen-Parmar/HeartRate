"""GPU data augmentation that simulates real webcam conditions.

Every augmentation exists for a concrete failure mode of the old model:
  heart-rate resampling   -> label HR spread uniformly over 45-180 BPM (UBFC is mostly 60-120);
                             prevents the network from learning a "typical HR" prior, so a person
                             coming back from a run at 130+ BPM is NOT pulled down to ~80
  low light               -> gain down to 6%, gamma, shot + read noise, 8-bit re-quantisation
  colour / side lighting  -> per-channel gains, spatial illumination gradient
  illumination drift      -> slow (<0.5 Hz) multiplicative drift + sudden exposure steps
  motion                  -> per-frame smooth random-walk translation / rotation / scale + jumps
  low/variable webcam fps -> sample at 10-25 fps and linearly re-interpolate to 30 Hz exactly
                             like the live app does
  blur                    -> defocus / denoising
"""

import math
import random
import torch
import torch.nn.functional as F

# augmentation profile. Deployed = run-1 profile below. (Run 2 tried a "hard low light" profile:
# prob 0.7, gain 0.04-0.6, read<=0.015, shot<=0.03, high-ISO 0.2, motion<=7% -> no low-light gain,
# worse motion robustness on the development fold, so it was dropped.)
LOWLIGHT_PROB = 0.6
LOWLIGHT_GAIN = (0.06, 0.8)
READ_NOISE_MAX = 0.012
SHOT_NOISE_MAX = 0.02
HIGH_ISO_PROB = 0.0
MOTION_AMP_MAX = 0.05
MOTION_PROB = 0.7


def _smooth_noise(B, T, device, cutoff_frames):
    """Low-pass random walk in [-1,1], shape [B,T]."""
    n = torch.randn(B, T + 2 * cutoff_frames, device=device)
    k = torch.ones(1, 1, cutoff_frames, device=device) / cutoff_frames
    s = F.conv1d(n.unsqueeze(1), k).squeeze(1)
    s = F.conv1d(s.unsqueeze(1), k).squeeze(1)[:, :T]
    s = s - s.mean(1, keepdim=True)
    return s / (s.abs().amax(1, keepdim=True) + 1e-6)


def spatial_augment(x, out_size, strength=1.0):
    """x: [B,3,T,H,W] -> [B,3,T,out,out]; random zoom/shift/rotation (static + per-frame motion)."""
    B, C, T, H, W = x.shape
    dev = x.device
    scale = torch.empty(B, device=dev).uniform_(0.86, 1.0)               # crop fraction of 1.6x box
    tx = torch.empty(B, device=dev).uniform_(-0.06, 0.06)
    ty = torch.empty(B, device=dev).uniform_(-0.06, 0.06)
    rot = torch.empty(B, device=dev).uniform_(-0.12, 0.12)
    flip = (torch.rand(B, device=dev) < 0.5).float() * -2 + 1

    # per-frame motion trajectory
    amp = torch.empty(B, 1, device=dev).uniform_(0, MOTION_AMP_MAX) * strength * (torch.rand(B, 1, device=dev) < MOTION_PROB)
    cut = random.choice([6, 10, 20])
    mx = _smooth_noise(B, T, dev, cut) * amp
    my = _smooth_noise(B, T, dev, cut) * amp
    mr = _smooth_noise(B, T, dev, cut) * amp * 1.5
    ms = 1 + _smooth_noise(B, T, dev, 30) * amp * 0.8
    # occasional abrupt jump (detector re-lock / sudden head move)
    jump = (torch.rand(B, 1, device=dev) < 0.25 * strength).float()
    jt = torch.randint(10, T - 10, (B, 1), device=dev)
    step = (torch.arange(T, device=dev)[None] >= jt).float() * jump
    mx = mx + step * torch.empty(B, 1, device=dev).uniform_(-0.04, 0.04)

    s = (scale[:, None] * ms)                                           # [B,T]
    r = rot[:, None] + mr
    cos, sin = torch.cos(r) * s, torch.sin(r) * s
    theta = torch.zeros(B, T, 2, 3, device=dev)
    theta[..., 0, 0] = cos * flip[:, None]
    theta[..., 0, 1] = -sin
    theta[..., 1, 0] = sin * flip[:, None]
    theta[..., 1, 1] = cos
    theta[..., 0, 2] = tx[:, None] + mx
    theta[..., 1, 2] = ty[:, None] + my
    frames = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    grid = F.affine_grid(theta.view(B * T, 2, 3), (B * T, C, out_size, out_size), align_corners=False)
    out = F.grid_sample(frames, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return out.view(B, T, C, out_size, out_size).permute(0, 2, 1, 3, 4)


def photometric_augment(x, strength=1.0, force_lowlight=False):
    """x in [0,1], [B,3,T,H,W]. force_lowlight: every clip gets the low-light gain (paired views)."""
    B, C, T, H, W = x.shape
    dev = x.device
    # colour cast
    cast = torch.empty(B, 3, 1, 1, 1, device=dev).uniform_(0.85, 1.15)
    x = x * cast
    # spatial illumination gradient (side light)
    g = torch.empty(B, 1, 1, 1, 1, device=dev).uniform_(-0.5, 0.5) * (torch.rand(B, 1, 1, 1, 1, device=dev) < 0.5)
    ramp = torch.linspace(-0.5, 0.5, W, device=dev).view(1, 1, 1, 1, W)
    x = x * (1 + g * ramp)
    # slow illumination drift + exposure steps
    drift_amp = torch.empty(B, 1, device=dev).uniform_(0, 0.08) * (torch.rand(B, 1, device=dev) < 0.6)
    drift = 1 + _smooth_noise(B, T, dev, 45) * drift_amp
    stepm = (torch.rand(B, 1, device=dev) < 0.15).float()
    st = torch.randint(5, T - 5, (B, 1), device=dev)
    drift = drift * (1 + stepm * (torch.arange(T, device=dev)[None] >= st).float()
                     * torch.empty(B, 1, device=dev).uniform_(-0.15, 0.15))
    x = x * drift.view(B, 1, T, 1, 1)
    # low light: gain (log-uniform LOWLIGHT_GAIN) + gamma
    lowl = (torch.rand(B, device=dev) < (1.0 if force_lowlight else LOWLIGHT_PROB * strength))
    gain = torch.where(lowl, torch.exp(torch.empty(B, device=dev).uniform_(math.log(LOWLIGHT_GAIN[0]), math.log(LOWLIGHT_GAIN[1]))),
                       torch.empty(B, device=dev).uniform_(0.8, 1.3)).view(B, 1, 1, 1, 1)
    gamma = torch.empty(B, 1, 1, 1, 1, device=dev).uniform_(0.8, 1.25)
    x = (x.clamp(1e-5, 1) ** gamma) * gain
    # sensor noise (shot + read), stronger when dark
    read = torch.empty(B, 1, 1, 1, 1, device=dev).uniform_(0, READ_NOISE_MAX) * strength
    shot = torch.empty(B, 1, 1, 1, 1, device=dev).uniform_(0, SHOT_NOISE_MAX) * strength
    # "high ISO": a webcam that compensates darkness with analogue gain -> normal brightness, heavy noise
    iso = (torch.rand(B, 1, 1, 1, 1, device=dev) < HIGH_ISO_PROB).float()
    read = read + iso * torch.empty(B, 1, 1, 1, 1, device=dev).uniform_(0.01, 0.03)
    x = x + torch.randn_like(x) * (read + shot * x.clamp(min=0).sqrt())
    # blur
    if random.random() < 0.3:
        k = torch.tensor([0.25, 0.5, 0.25], device=dev)
        k2 = (k[:, None] * k[None, :]).view(1, 1, 3, 3).repeat(C, 1, 1, 1)
        fr = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        fr = F.conv2d(F.pad(fr, (1, 1, 1, 1), mode="replicate"), k2, groups=C)
        x = fr.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)
    # 8-bit sensor quantisation (dark scenes lose levels)
    x = torch.round(x.clamp(0, 1) * 255) / 255
    return x


def fps_drop(x, prob=0.3):
    """Simulate a webcam delivering 10-25 fps, re-interpolated to 30 Hz (as the live app does)."""
    B, C, T, H, W = x.shape
    out = x.clone()
    for b in range(B):
        if random.random() > prob:
            continue
        f_low = random.uniform(10, 25)
        jitter = random.uniform(0, 0.15)
        n = int(T / 30.0 * f_low)
        t_cap = torch.arange(n, device=x.device, dtype=torch.float32) / f_low * 30.0
        t_cap = (t_cap + torch.randn(n, device=x.device) * jitter * 30.0 / f_low).clamp(0, T - 1)
        t_cap, _ = torch.sort(t_cap)
        src = x[b][:, t_cap.round().long()]                            # captured frames
        t_out = torch.arange(T, device=x.device, dtype=torch.float32)
        idx = torch.searchsorted(t_cap, t_out, right=True) - 1
        idx = idx.clamp(0, n - 2)
        w = ((t_out - t_cap[idx]) / (t_cap[idx + 1] - t_cap[idx]).clamp(min=1e-3)).clamp(0, 1)
        w = w.view(1, T, 1, 1)
        out[b] = src[:, idx] * (1 - w) + src[:, idx + 1] * w
    return out
