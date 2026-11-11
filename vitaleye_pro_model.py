"""
VitalEyeProNet — Next-Generation Low-Light & Motion-Robust rPPG Architecture
════════════════════════════════════════════════════════════════════════════
Designed to solve:
  1. Low-light & illumination drift (via local AC/DC normalized reflectance + gamma normalization)
  2. Head & bounding-box movements (via Joint-Mask Boundary Suppression + POS/CHROM physics maps
     + Learnable Spatial Attention Gate)
  3. 40–50 BPM low-frequency spectral pinning (via second-order temporal derivatives, multi-scale
     dilated TCN + Temporal Self-Attention, and built-in differentiable high-pass detrending)
  4. Optional Dual-Stream Chest Respiration fusion (`preprocessed_vidformer2`)

Can take raw float video `[B, 3, T, H, W]` in `[0, 1]` directly (computing all 10 physics-guided
channels on the GPU inside `PhysicsGuidedStem`) OR pre-stacked `[B, 6, T, H, W]` tensors.
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import periodogram, butter, filtfilt


# ═══════════════════════════════════════════════════════════════════════════════
#  1. GPU PHYSICS-GUIDED 10-CHANNEL INPUT STEM
# ═══════════════════════════════════════════════════════════════════════════════
class PhysicsGuidedStem(nn.Module):
    """
    Transforms raw skin-masked RGB video `[B, 3, T, H, W]` in `[0, 1]` (or `[B, 6, T, H, W]`)
    into a 10-channel illumination-invariant, motion-compensated tensor on the GPU:
      • Ch 0..2 : Illumination-normalized RGB (per-frame spatial mean normalized)
      • Ch 3..5 : Joint-Masked AC/DC Normalized Frame Difference:
                  ((I_t - I_{t-1}) / (I_t + I_{t-1} + eps)) * M_joint
                  -> Eliminates skin-mask edge step transients & low-light scale decay!
      • Ch 6..7 : Second-Order Temporal Derivative (Acceleration) on Green & Green-Red:
                  (I_t - 2*I_{t-1} + I_{t-2}) * M_joint3
                  -> Natural f^2 high-pass filter that suppresses <0.8 Hz motion drift!
      • Ch 8    : Per-pixel POS (Plane-Orthogonal-to-Skin) projection map
      • Ch 9    : Per-pixel CHROM (Chrominance-based rPPG) projection map
    """

    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept either [B, 3, T, H, W] or [B, 6, T, H, W] (uses first 3 RGB channels)
        if x.shape[1] >= 6:
            rgb = x[:, :3]
        else:
            rgb = x

        B, C, T, H, W = rgb.shape

        # Identify valid non-zero skin pixels (background = 0)
        skin_mask = (rgb.sum(dim=1, keepdim=True) > 0.02).float()  # [B, 1, T, H, W]
        # Joint mask: only compute temporal differences where BOTH t and t-1 are skin!
        # This completely eliminates the massive step-edge spikes when the mask boundary moves.
        joint_mask_1 = torch.zeros_like(skin_mask)
        joint_mask_1[:, :, 1:] = skin_mask[:, :, 1:] * skin_mask[:, :, :-1]

        joint_mask_2 = torch.zeros_like(skin_mask)
        joint_mask_2[:, :, 2:] = (
            skin_mask[:, :, 2:] * skin_mask[:, :, 1:-1] * skin_mask[:, :, :-2]
        )

        # Per-frame, per-channel spatial mean over active skin pixels (for low-light invariance)
        mask_count = skin_mask.sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
        spatial_mean = (rgb * skin_mask).sum(dim=(-2, -1), keepdim=True) / mask_count  # [B, 3, T, 1, 1]

        # Temporal mean of spatial_mean over a local window (DC baseline)
        dc_mean = spatial_mean.mean(dim=2, keepdim=True).clamp(min=0.02)

        # 1. Channels 0..2: Illumination-Normalized Appearance RGB
        rgb_norm = torch.clamp((rgb / dc_mean) * 0.5, 0.0, 1.5) * skin_mask

        # 2. Channels 3..5: AC/DC Normalized Frame Difference (DeepPhys + Joint Mask Clamping)
        ac_dc_diff = torch.zeros_like(rgb)
        num = rgb[:, :, 1:] - rgb[:, :, :-1]
        den = rgb[:, :, 1:] + rgb[:, :, :-1] + 0.05
        ac_dc_diff[:, :, 1:] = torch.clamp(num / den, -0.15, 0.15) * 10.0
        ac_dc_diff = ac_dc_diff * joint_mask_1

        # 3. Channels 6..7: Second-Order Temporal Derivative (Green & Green-Red Acceleration)
        accel_g = torch.zeros((B, 1, T, H, W), device=rgb.device, dtype=rgb.dtype)
        accel_gr = torch.zeros((B, 1, T, H, W), device=rgb.device, dtype=rgb.dtype)
        g_ch = rgb[:, 1:2] / (spatial_mean[:, 1:2].clamp(min=0.02))
        r_ch = rgb[:, 0:1] / (spatial_mean[:, 0:1].clamp(min=0.02))
        gr_diff = g_ch - r_ch

        if T >= 3:
            accel_g[:, :, 2:] = torch.clamp(
                g_ch[:, :, 2:] - 2.0 * g_ch[:, :, 1:-1] + g_ch[:, :, :-2], -0.15, 0.15
            ) * 10.0
            accel_gr[:, :, 2:] = torch.clamp(
                gr_diff[:, :, 2:] - 2.0 * gr_diff[:, :, 1:-1] + gr_diff[:, :, :-2], -0.15, 0.15
            ) * 10.0
        accel_g = accel_g * joint_mask_2
        accel_gr = accel_gr * joint_mask_2

        # 4. Channels 8..9: Physics-Informed POS & CHROM Spatial Feature Maps
        cn = (rgb / dc_mean) * skin_mask  # temporally normalized R_n, G_n, B_n
        rn, gn, bn = cn[:, 0:1], cn[:, 1:2], cn[:, 2:3]

        # POS projection: S1 = Gn - Bn, S2 = Gn + Bn - 2*Rn
        pos_s1 = gn - bn
        pos_s2 = gn + bn - 2.0 * rn
        pos_map = torch.clamp((pos_s1 + 0.5 * pos_s2) * 4.0, -2.0, 2.0) * skin_mask

        # CHROM projection: X = 3*Rn - 2*Gn, Y = 1.5*Rn + Gn - 1.5*Bn
        chrom_x = 3.0 * rn - 2.0 * gn
        chrom_y = 1.5 * rn + gn - 1.5 * bn
        chrom_map = torch.clamp((chrom_x - 0.8 * chrom_y) * 3.0, -2.0, 2.0) * skin_mask

        # Stack all 10 channels: [B, 10, T, H, W]
        out_10ch = torch.cat(
            [rgb_norm, ac_dc_diff, accel_g, accel_gr, pos_map, chrom_map], dim=1
        )
        return out_10ch


# ═══════════════════════════════════════════════════════════════════════════════
#  2. MPS/CUDA-COMPATIBLE 3D BUILDING BLOCKS & SPATIAL SKIN ATTENTION
# ═══════════════════════════════════════════════════════════════════════════════
class SpatialMaxPool3d(nn.Module):
    """MPS/CUDA/CPU-native spatial max pooling (1, 2, 2)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        x_2d = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        out_2d = F.max_pool2d(x_2d, kernel_size=2, stride=2)
        _, _, H_out, W_out = out_2d.shape
        return out_2d.view(B, T, C, H_out, W_out).permute(0, 2, 1, 3, 4).contiguous()


class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation 3D channel attention."""

    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        r = max(ch // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(ch, r, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(r, ch, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape[:2]
        w = self.fc(x.mean(dim=(2, 3, 4))).view(B, C, 1, 1, 1)
        return x * w


class SpatialSkinAttention3D(nn.Module):
    """
    Learns a per-frame spatial confidence mask over (H, W) so the network
    automatically focuses on high-SNR vascular regions (forehead & malar cheeks)
    and suppresses noisy boundaries, glasses glare, and motion shadows.
    """

    def __init__(self, ch: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv3d(ch, max(ch // 4, 8), kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(max(ch // 4, 8)),
            nn.SiLU(inplace=True),
            nn.Conv3d(max(ch // 4, 8), 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize attention weights spatially so average gain stays ~1.0
        a = self.attn(x)  # [B, 1, T, H, W]
        a_norm = a / (a.mean(dim=(-2, -1), keepdim=True) + 1e-6)
        return x * a_norm


class ResBlock3D(nn.Module):
    """3D Residual Block with SE Channel Attention and SiLU activations."""

    def __init__(self, in_ch: int, out_ch: int, stride=1, dropout: float = 0.05):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.se = SEBlock3D(out_ch)
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.drop(out)
        return self.act(out + identity)


# ═══════════════════════════════════════════════════════════════════════════════
#  3. MULTI-SCALE DILATED TCN + TEMPORAL TRANSFORMER REFINEMENT HEAD
# ═══════════════════════════════════════════════════════════════════════════════
class MultiScaleTemporalBlock(nn.Module):
    """
    Parallel dilated 1D convolutions (dilation 1, 2, 4) + Temporal Self-Attention
    to capture both beat-to-beat pulse shape and multi-cycle cardiac periodicity.
    """

    def __init__(self, ch: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        branch_ch = ch // 2
        self.b1 = nn.Sequential(
            nn.Conv1d(ch, branch_ch, kernel_size=5, padding=2, dilation=1, bias=False),
            nn.BatchNorm1d(branch_ch),
            nn.SiLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv1d(ch, branch_ch, kernel_size=5, padding=4, dilation=2, bias=False),
            nn.BatchNorm1d(branch_ch),
            nn.SiLU(inplace=True),
        )
        self.b3 = nn.Sequential(
            nn.Conv1d(ch, branch_ch, kernel_size=5, padding=8, dilation=4, bias=False),
            nn.BatchNorm1d(branch_ch),
            nn.SiLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(branch_ch * 3, ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(ch),
            nn.SiLU(inplace=True),
        )

        # Lightweight Temporal Self-Attention along T to enforce global rhythm consistency
        self.norm = nn.LayerNorm(ch)
        self.mha = nn.MultiheadAttention(embed_dim=ch, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        ms = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        x = x + self.fuse(ms)

        # Temporal self-attention: [B, T, C]
        xt = x.permute(0, 2, 1)
        xt_norm = self.norm(xt)
        attn_out, _ = self.mha(xt_norm, xt_norm, xt_norm, need_weights=False)
        xt = xt + self.drop(attn_out)
        return xt.permute(0, 2, 1)


# ═══════════════════════════════════════════════════════════════════════════════
#  4. FULL MODEL: VitalEyeProNet (Low-Light & Motion-Resilient rPPG + RR)
# ═══════════════════════════════════════════════════════════════════════════════
class VitalEyeProNet(nn.Module):
    """
    Complete end-to-end rPPG (and optional dual-stream Chest RR) network.
      • Input: `x` of shape `[B, 3, T, 64, 64]` or `[B, 6, T, 64, 64]`
      • Output:
          - If `return_aux=False` (default for drop-in compatibility with PhysFormerX):
            returns `ppg_pred` of shape `[B, T]` (zero-mean, high-pass detrended)
          - If `return_aux=True`:
            returns `(ppg_pred, rr_pred)`
    """

    def __init__(self, seq_len: int = 300, use_chest: bool = False):
        super().__init__()
        self.seq_len = seq_len
        self.use_chest = use_chest

        # 1. Physics-guided 10-channel stem
        self.physics_stem = PhysicsGuidedStem()

        # 2. Spatiotemporal 3D Conv Stem (10ch → 48ch)
        self.stem = nn.Sequential(
            nn.Conv3d(10, 48, kernel_size=(1, 5, 5), padding=(0, 2, 2), bias=False),
            nn.BatchNorm3d(48),
            nn.SiLU(inplace=True),
            SpatialMaxPool3d(),  # -> [B, 48, T, 32, 32]
        )

        # 3. Hierarchical 3D Residual + Spatial Skin Attention Stages
        self.layer1 = ResBlock3D(48, 64, stride=(1, 2, 2), dropout=0.05)   # -> [B, 64, T, 16, 16]
        self.spatial_attn1 = SpatialSkinAttention3D(64)
        self.layer2 = ResBlock3D(64, 96, stride=1, dropout=0.05)           # -> [B, 96, T, 16, 16]

        self.layer3 = ResBlock3D(96, 128, stride=(1, 2, 2), dropout=0.05)  # -> [B, 128, T, 8, 8]
        self.spatial_attn2 = SpatialSkinAttention3D(128)
        self.layer4 = ResBlock3D(128, 128, stride=1, dropout=0.05)         # -> [B, 128, T, 8, 8]

        # 4. Multi-Scale Dilated TCN + Temporal Transformer Head
        self.temporal_block1 = MultiScaleTemporalBlock(128, n_heads=4, dropout=0.10)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Conv1d(64, 1, kernel_size=1),
        )

        # Optional respiratory head (from low-frequency branch before high-pass detrending)
        self.rr_head = nn.Sequential(
            nn.Conv1d(128, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU(inplace=True),
            nn.Conv1d(32, 1, kernel_size=1),
        )

    @staticmethod
    def highpass_detrend(sig: torch.Tensor, kernel_size: int = 27) -> torch.Tensor:
        """
        Differentiable moving-average subtraction that removes <0.75 Hz baseline
        wander so the output waveform cannot drift into the 40–50 BPM noise floor.
        """
        T = sig.shape[-1]
        k = min(kernel_size, (T // 3) | 1)
        if k < 5:
            return sig - sig.mean(dim=-1, keepdim=True)
        pad = k // 2
        sig_3d = sig.unsqueeze(1)  # [B, 1, T]
        trend = F.avg_pool1d(F.pad(sig_3d, (pad, pad), mode="replicate"), kernel_size=k, stride=1)
        detrended = (sig_3d - trend).squeeze(1)
        return detrended - detrended.mean(dim=-1, keepdim=True)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        # x: [B, 3 or 6, T, H, W]
        x10 = self.physics_stem(x)  # [B, 10, T, H, W]

        feat = self.stem(x10)
        feat = self.layer1(feat)
        feat = self.spatial_attn1(feat)
        feat = self.layer2(feat)
        feat = self.layer3(feat)
        feat = self.spatial_attn2(feat)
        feat = self.layer4(feat)

        # Collapse spatial dimensions (H, W) -> [B, 128, T]
        t_feat = feat.mean(dim=(-2, -1))

        # Multi-scale temporal refinement
        t_refined = self.temporal_block1(t_feat)

        raw_ppg = self.temporal_conv(t_refined).squeeze(1)  # [B, T]
        ppg_out = self.highpass_detrend(raw_ppg, kernel_size=27)

        if return_aux:
            rr_out = self.rr_head(t_feat).squeeze(1)
            rr_out = rr_out - rr_out.mean(dim=-1, keepdim=True)
            return ppg_out, rr_out

        return ppg_out


# ═══════════════════════════════════════════════════════════════════════════════
#  5. ROBUST HARMONIC-AWARE BPM & RR ESTIMATOR (Eliminates 40–50 BPM Pinning)
# ═══════════════════════════════════════════════════════════════════════════════
def robust_bpm_from_signal(signals: np.ndarray, fps_list=30.0) -> np.ndarray:
    """
    Computes BPM from a batch of 1-D rPPG waveforms `(B, T)` using:
      1. Zero-phase 3rd-order Butterworth bandpass filter (`0.82 - 2.50 Hz` = `49.2 - 150 BPM`)
         to strip sub-cardiac motion/exposure drift before FFT
      2. Hann-windowed 4096-pt periodogram
      3. Parabolic log-power interpolation for sub-bin accuracy
    `fps_list` can be a scalar float or a 1-D array/list of per-sample FPS values.
    """
    if signals.ndim == 1:
        signals = signals[None, :]
    B, T = signals.shape

    if isinstance(fps_list, (int, float)):
        fps_arr = [float(fps_list)] * B
    else:
        fps_arr = [float(f) for f in fps_list]

    bpms = []
    for i in range(B):
        s = signals[i].astype(np.float64)
        fps = max(10.0, min(60.0, fps_arr[i]))
        s = s - np.mean(s)
        if np.std(s) < 1e-7:
            bpms.append(75.0)
            continue

        # Pre-FFT bandpass filtering (0.80 - 2.50 Hz) to prevent 1/f drift sidelobes at 45 BPM
        if T >= 45:
            try:
                nyq = fps / 2.0
                lo = max(0.80 / nyq, 0.02)
                hi = min(2.50 / nyq, 0.98)
                b, a = butter(3, [lo, hi], btype="band")
                s = filtfilt(b, a, s)
            except Exception:
                pass

        freqs, psd = periodogram(s, fs=fps, nfft=4096, detrend="constant", window="hann")
        valid = np.where((freqs >= 0.80) & (freqs <= 2.50))[0]
        if len(valid) == 0:
            bpms.append(75.0)
            continue

        sub_psd = psd[valid]
        k = int(valid[int(np.argmax(sub_psd))])

        # Parabolic interpolation around peak
        if 0 < k < len(psd) - 1:
            y0 = math.log(max(float(psd[k - 1]), 1e-12))
            y1 = math.log(max(float(psd[k]), 1e-12))
            y2 = math.log(max(float(psd[k + 1]), 1e-12))
            denom = 2.0 * (2.0 * y1 - y0 - y2)
            delta = (y0 - y2) / denom if abs(denom) > 1e-9 else 0.0
            delta = max(-0.5, min(0.5, delta))
            df = float(freqs[1] - freqs[0])
            f_peak = float(freqs[k] + delta * df)
        else:
            f_peak = float(freqs[k])

        bpms.append(f_peak * 60.0)

    return np.array(bpms, dtype=np.float32)
