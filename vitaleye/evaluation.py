"""Window-level evaluation of VitalNet and classical baselines, incl. low-light / motion stress tests."""

import math
import numpy as np
import torch
import torch.nn.functional as F

from . import INPUT_SIZE, TARGET_FPS
from .data import to_model_input, EVAL_SCALE
from .dsp import hr_from_signal, hr_from_waves, pos_signal, chrom_signal, green_signal

WIN = 300   # 10 s windows (same as the live app)


def stress_transform(x80: torch.Tensor, cond: str, seed: int = 0) -> torch.Tensor:
    """x80: [1,3,T,80,80] in [0,1] -> [1,3,T,72,72] model view under a deterministic condition."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    B, C, T, H, W = x80.shape
    dev = x80.device
    if "motion" in cond:
        # head sway: 0.2-1.0 Hz quasi-periodic translation of +-6% + rotation +-0.08 rad + abrupt shifts
        t = torch.arange(T, dtype=torch.float32) / TARGET_FPS
        f1, f2 = 0.3 + 0.4 * torch.rand(1, generator=g).item(), 0.15 + 0.3 * torch.rand(1, generator=g).item()
        mx = 0.06 * torch.sin(2 * math.pi * f1 * t) + 0.02 * torch.randn(T, generator=g).cumsum(0) / math.sqrt(T)
        my = 0.04 * torch.sin(2 * math.pi * f2 * t + 1.0)
        rot = 0.08 * torch.sin(2 * math.pi * f2 * t)
        jumps = (torch.rand(T, generator=g) < 1 / 90).float() * (torch.rand(T, generator=g) - 0.5) * 0.08
        mx = mx + jumps.cumsum(0)
        theta = torch.zeros(T, 2, 3)
        theta[:, 0, 0] = torch.cos(rot) * EVAL_SCALE; theta[:, 0, 1] = -torch.sin(rot) * EVAL_SCALE
        theta[:, 1, 0] = torch.sin(rot) * EVAL_SCALE; theta[:, 1, 1] = torch.cos(rot) * EVAL_SCALE
        theta[:, 0, 2] = mx; theta[:, 1, 2] = my
        fr = x80[0].permute(1, 0, 2, 3)
        grid = F.affine_grid(theta.to(dev), (T, C, INPUT_SIZE, INPUT_SIZE), align_corners=False)
        x = F.grid_sample(fr, grid, mode="bilinear", padding_mode="border", align_corners=False)
        x = x.permute(1, 0, 2, 3).unsqueeze(0)
    else:
        x = to_model_input(x80)
    if "lowlight" in cond:
        # ~10x darker scene, sensor noise, 8-bit quantisation
        noise = torch.randn(x.shape, generator=g).to(dev)
        x = x * 0.10
        x = x + noise * (0.004 + 0.015 * x.clamp(min=0).sqrt())
        x = torch.round(x.clamp(0, 1) * 255) / 255
    if "lowfps" in cond:
        f_low = 12.0
        n = int(T / TARGET_FPS * f_low)
        t_cap = torch.arange(n, dtype=torch.float32, device=dev) / f_low * TARGET_FPS
        src = x[0][:, t_cap.round().long().clamp(max=T - 1)]
        t_out = torch.arange(T, device=dev, dtype=torch.float32)
        idx = (torch.searchsorted(t_cap, t_out, right=True) - 1).clamp(0, n - 2)
        w = ((t_out - t_cap[idx]) / (t_cap[idx + 1] - t_cap[idx])).clamp(0, 1).view(1, T, 1, 1)
        x = (src[:, idx] * (1 - w) + src[:, idx + 1] * w).unsqueeze(0)
    return x


@torch.no_grad()
def predict_windows(models, x72: torch.Tensor):
    """x72: [1,3,T,72,72] -> list of (start, [waveform per model]) for non-overlapping 10 s windows."""
    T = x72.shape[2]
    out = []
    for s in range(0, T - WIN + 1, WIN):
        clip = x72[:, :, s:s + WIN]
        waves = []
        for m in models:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x72.is_cuda):
                waves.append(m(clip).float().squeeze(0).cpu().numpy())
        out.append((s, waves))
    return out


def ellipse_rgb(x72: torch.Tensor) -> np.ndarray:
    _, C, T, H, W = x72.shape
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    ell = ((((xx - W / 2) / (W * 0.30)) ** 2 + ((yy - H * 0.5) / (H * 0.36)) ** 2) <= 1).to(x72.device)
    v = x72[0][:, :, ell]                                   # [3,T,N]
    return (v.mean(-1).T * 255).cpu().numpy().astype(np.float64)


def evaluate_video(models, video, device, conditions=("clean",), baselines=True, seed=0, combine="psd"):
    """Return {cond: {method: [(hr_pred, hr_gt, snr_pred), ...]}} for 10 s windows."""
    x80 = torch.from_numpy(video.frames).to(device).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
    res = {}
    for ci, cond in enumerate(conditions):
        x72 = stress_transform(x80, cond, seed=seed * 100 + ci)
        r = {"vitalnet": []}
        if baselines:
            r.update({"POS": [], "CHROM": [], "GREEN": []})
        preds = predict_windows(models, x72) if models else [(s, None) for s in range(0, x72.shape[2] - WIN + 1, WIN)]
        rgb = ellipse_rgb(x72) if baselines else None
        for s, wave in preds:
            hr_gt = hr_from_signal(video.ppg_raw[s:s + WIN], TARGET_FPS)[0]
            if wave is not None:
                h, snr, _ = hr_from_waves(wave, TARGET_FPS, combine)
                r["vitalnet"].append((h, hr_gt, snr))
            if baselines:
                seg = rgb[s:s + WIN]
                for name, fn in (("POS", pos_signal), ("CHROM", chrom_signal), ("GREEN", green_signal)):
                    h, snr = hr_from_signal(fn(seg, TARGET_FPS), TARGET_FPS)
                    r[name].append((h, hr_gt, snr))
        res[cond] = r
    return res


def metrics(pairs):
    a = np.array(pairs, dtype=np.float64)
    if len(a) == 0:
        return {}
    p, g = a[:, 0], a[:, 1]
    e = p - g
    return dict(
        n=int(len(a)),
        MAE=float(np.mean(np.abs(e))),
        RMSE=float(np.sqrt(np.mean(e ** 2))),
        MAPE=float(np.mean(np.abs(e) / g) * 100),
        r=float(np.corrcoef(p, g)[0, 1]) if np.std(p) > 1e-6 and np.std(g) > 1e-6 else float("nan"),
        within5=float(np.mean(np.abs(e) <= 5) * 100),
        bias=float(np.mean(e)),
    )
