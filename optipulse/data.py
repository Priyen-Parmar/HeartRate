"""In-memory UBFC cache + GPU batch builder (HR resampling, spatial & photometric augmentation)."""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F

from . import INPUT_SIZE, TARGET_FPS
from .dsp import preprocess_pulse, hr_from_signal
from .augment import spatial_augment, photometric_augment, fps_drop

EVAL_SCALE = 0.93  # centre of the training zoom range (0.86-1.0)


class Video:
    def __init__(self, path):
        z = np.load(path)
        self.sid = os.path.splitext(os.path.basename(path))[0]
        self.frames = z["frames"]                       # [N,80,80,3] uint8 @30 Hz
        self.ppg_raw = z["ppg"].astype(np.float64)      # contact PPG @30 Hz
        self.label = preprocess_pulse(self.ppg_raw, TARGET_FPS).astype(np.float32)
        self.n = len(self.frames)
        # local HR track (10 s window, 1 s step) for HR-resampling augmentation
        w = int(10 * TARGET_FPS)
        self.hr_track = []
        for a in range(0, max(1, self.n - w + 1), 30):
            self.hr_track.append(hr_from_signal(self.label[a:a + w], TARGET_FPS, already_filtered=True)[0])
        self.hr_track = np.array(self.hr_track)

    def local_hr(self, start):
        return float(self.hr_track[min(len(self.hr_track) - 1, start // 30)])


def load_videos(proc_dir, sids):
    return [Video(os.path.join(proc_dir, f"{s}.npz")) for s in sids]


def to_model_input(x80: torch.Tensor, scale: float = EVAL_SCALE) -> torch.Tensor:
    """Deterministic 80->72 view used for evaluation AND live inference. x80: [B,3,T,80,80] float."""
    B, C, T, H, W = x80.shape
    theta = torch.tensor([[scale, 0, 0], [0, scale, 0]], device=x80.device, dtype=x80.dtype)
    theta = theta.unsqueeze(0).expand(B * T, 2, 3)
    fr = x80.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    grid = F.affine_grid(theta, (B * T, C, INPUT_SIZE, INPUT_SIZE), align_corners=False)
    out = F.grid_sample(fr, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return out.view(B, T, C, INPUT_SIZE, INPUT_SIZE).permute(0, 2, 1, 3, 4)


class BatchBuilder:
    def __init__(self, videos, T=180, device="cuda", hr_aug_prob=0.6, hr_range=(45, 180), aug_strength=1.0):
        self.videos = videos
        self.T = T
        self.device = device
        self.hr_aug_prob = hr_aug_prob
        self.hr_range = hr_range
        self.strength = aug_strength
        lens = np.array([v.n for v in videos], dtype=np.float64)
        self.p = lens / lens.sum()

    def sample(self, B, paired=False):
        """paired=True: B//2 clips, each returned twice with the SAME geometry / motion but two
        independent photometric draws, the second forced into low light (x[:B//2] <-> x[B//2:])."""
        if paired:
            B = B // 2
        T = self.T
        segs, labs, speeds = [], [], []
        for _ in range(B):
            v = self.videos[np.random.choice(len(self.videos), p=self.p)]
            r = random.random()
            if r < self.hr_aug_prob:
                probe = random.randint(0, max(0, v.n - 301))
                base = v.local_hr(probe)
                tgt = random.uniform(*self.hr_range)
                s = float(np.clip(tgt / max(base, 30.0), 0.55, 1.8))
            elif r < self.hr_aug_prob + 0.2:
                s = random.uniform(0.9, 1.1)
            else:
                s = 1.0
            L = int(np.ceil((T - 1) * s)) + 2
            if L > v.n:
                s = (v.n - 2) / (T - 1)
                L = v.n
            st = random.randint(0, v.n - L)
            segs.append(v.frames[st:st + L]); labs.append(v.label[st:st + L]); speeds.append(s)
        Lmax = max(len(s) for s in segs)
        fr = np.zeros((B, Lmax, 80, 80, 3), np.uint8)
        lb = np.zeros((B, Lmax), np.float32)
        for i, (s, l) in enumerate(zip(segs, labs)):
            fr[i, :len(s)] = s; lb[i, :len(l)] = l
        x = torch.from_numpy(fr).to(self.device, non_blocking=True).permute(0, 4, 1, 2, 3).float() / 255.0
        y = torch.from_numpy(lb).to(self.device)
        # temporal resampling (HR augmentation) with linear interpolation
        sp = torch.tensor(speeds, device=self.device).view(B, 1)
        pos = torch.arange(T, device=self.device).view(1, T) * sp           # [B,T]
        i0 = pos.floor().long().clamp(max=Lmax - 2)
        w = (pos - i0.float()).clamp(0, 1)
        idx = i0.view(B, 1, T, 1, 1).expand(B, 3, T, 80, 80)
        x0 = torch.gather(x, 2, idx); x1 = torch.gather(x, 2, idx + 1)
        wv = w.view(B, 1, T, 1, 1)
        x = x0 * (1 - wv) + x1 * wv
        y = torch.gather(y, 1, i0) * (1 - w) + torch.gather(y, 1, i0 + 1) * w
        # spatial + photometric + fps augmentation
        x = spatial_augment(x, INPUT_SIZE, self.strength)
        if paired:
            x = torch.cat([photometric_augment(x, self.strength), photometric_augment(x, self.strength, force_lowlight=True)], 0)
            y = torch.cat([y, y], 0)
        else:
            x = photometric_augment(x, self.strength)
        x = fps_drop(x, prob=0.3)
        y = (y - y.mean(1, keepdim=True)) / (y.std(1, keepdim=True) + 1e-6)
        return x, y
