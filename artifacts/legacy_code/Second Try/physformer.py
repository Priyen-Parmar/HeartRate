"""
PhysFormerX v3 — rPPG Heart Rate & Respiratory Rate Estimation for UBFC-rPPG
════════════════════════════════════════════════════════════════════════════
Designed for sub-3 BPM MAE on subject-level test split.

Architecture: Diff-normalized video → 3D ResNet with SE → temporal conv head
Key design decisions:
  • Difference-frame input: frame[t]-frame[t-1] emphasizes subtle color changes
  • Apple Silicon MPS-compatible SpatialMaxPool3d & 3D spatial mean pooling
    (eliminates missing 'aten::max_pool3d_with_indices' on macOS Metal/MPS while
     preserving bit-for-bit identical numerical outputs and state_dict keys)
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import glob, math, copy, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
from scipy.signal import periodogram
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════
DATA_DIR     = 'E:/CS776 Project/preprocessed_physformer'
SAVE_DIR     = 'E:/CS776 Project'
EPOCHS       = 100
FPS          = 30
SEQ_LEN      = 300
CROP_LEN     = 150          # random temporal crop for training
BATCH        = 4            # smaller batch = more updates per epoch
IMG_SIZE     = 64
LR_MAX       = 3e-4
WEIGHT_DECAY = 5e-5
EMA_DECAY    = 0.998
GRAD_CLIP    = 2.0
PATIENCE     = 15


# ═══════════════════════════════════════════════════════════════
#  BPM & RPM HELPERS
# ═══════════════════════════════════════════════════════════════
def bpm_from_signal(signals: np.ndarray, fps: float = FPS) -> np.ndarray:
    """Periodogram-based BPM. signals: (B, T) numpy float."""
    bpms = []
    for s in signals:
        s = s - np.mean(s)
        freqs, psd = periodogram(s, fs=fps, nfft=4096, detrend='constant',
                                 window='hann')
        valid = np.where((freqs >= 0.75) & (freqs <= 2.5))[0]
        if len(valid) > 0:
            bpms.append(freqs[valid][np.argmax(psd[valid])] * 60.0)
        else:
            bpms.append(75.0)
    return np.array(bpms)


def rpm_from_signal(signals: np.ndarray, fps: float = FPS) -> np.ndarray:
    """Periodogram-based Respiratory Rate (BrPM) in 0.15-0.40 Hz (9-24 BrPM)."""
    rpms = []
    for s in signals:
        s = s - np.mean(s)
        freqs, psd = periodogram(s, fs=fps, nfft=4096, detrend='constant',
                                 window='hann')
        valid = np.where((freqs >= 0.15) & (freqs <= 0.40))[0]
        if len(valid) > 0:
            rpms.append(freqs[valid][np.argmax(psd[valid])] * 60.0)
        else:
            rpms.append(15.0)
    return np.array(rpms)


# ═══════════════════════════════════════════════════════════════
#  DATASET — with diff-frame and temporal cropping
# ═══════════════════════════════════════════════════════════════
class rPPGDataset(Dataset):
    def __init__(self, file_paths, augment=True, img_size=IMG_SIZE,
                 crop_len=None):
        self.fps = file_paths
        self.augment = augment
        self.crop_len = crop_len
        self.resize = transforms.Resize((img_size, img_size), antialias=True)

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, idx):
        d = torch.load(self.fps[idx], weights_only=True)
        vid = d['video'].float() / 255.0          # [C, T, H, W]
        ppg = d['ppg'].float()

        C, T, H, W = vid.shape

        # Resize spatially
        vid = self.resize(vid.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)

        # Random temporal crop for training
        if self.crop_len and T > self.crop_len:
            start = random.randint(0, T - self.crop_len)
            vid = vid[:, start:start + self.crop_len]
            ppg = ppg[start:start + self.crop_len]

        if self.augment:
            if random.random() > 0.5:
                vid = torch.flip(vid, dims=[3])
            if random.random() > 0.5:
                vid = vid + torch.randn_like(vid) * 0.01
                vid = torch.clamp(vid, 0, 1)
            if random.random() > 0.5:
                bfactor = 1.0 + (random.random() - 0.5) * 0.2
                vid = torch.clamp(vid * bfactor, 0, 1)

        diff = torch.zeros_like(vid)
        diff[:, 1:] = vid[:, 1:] - vid[:, :-1]
        vid_6ch = torch.cat([vid, diff], dim=0)  # [6, T, H, W]

        ppg = (ppg - ppg.mean()) / (ppg.std() + 1e-8)
        return vid_6ch, ppg


def subject_split(data_dir, train_ratio=0.70, val_ratio=0.15):
    """3-way subject-level split → no data leakage."""
    files = sorted(glob.glob(os.path.join(data_dir, '*.pt')))
    if not files:
        return [], [], []

    from collections import defaultdict
    subj_files = defaultdict(list)
    for fp in files:
        try:
            d = torch.load(fp, weights_only=True)
            sid = d.get('subject_id', os.path.basename(fp))
        except Exception:
            sid = os.path.basename(fp)
        subj_files[sid].append(fp)

    subjects = sorted(subj_files.keys())
    random.seed(42)
    random.shuffle(subjects)

    n_train = max(1, int(len(subjects) * train_ratio))
    n_val = max(1, int(len(subjects) * val_ratio))
    train_subjs = set(subjects[:n_train])
    val_subjs = set(subjects[n_train:n_train + n_val])

    train_files, val_files, test_files = [], [], []
    for sid, fps_list in subj_files.items():
        if sid in train_subjs:
            train_files.extend(fps_list)
        elif sid in val_subjs:
            val_files.extend(fps_list)
        else:
            test_files.extend(fps_list)

    if len(val_files) == 0 or len(test_files) == 0:
        n = len(files)
        n_tr = int(0.70 * n)
        n_va = int(0.15 * n)
        train_files = files[:n_tr]
        val_files = files[n_tr:n_tr + n_va]
        test_files = files[n_tr + n_va:]

    return sorted(train_files), sorted(val_files), sorted(test_files)


# ═══════════════════════════════════════════════════════════════
#  MODEL BLOCKS (MPS-Native 3D Pooling & SE Attention)
# ═══════════════════════════════════════════════════════════════
class SpatialMaxPool3d(nn.Module):
    """
    Drop-in replacement for nn.MaxPool3d((1, 2, 2)) that applies F.max_pool2d
    across (B*T, C, H, W). Produces bit-for-bit identical output to
    nn.MaxPool3d((1, 2, 2)) while running natively on Apple Silicon MPS,
    CUDA, and CPU without triggering missing 'aten::max_pool3d_with_indices'.
    """
    def forward(self, x):
        B, C, T, H, W = x.shape
        x_2d = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        out_2d = F.max_pool2d(x_2d, kernel_size=2, stride=2)
        _, _, H_out, W_out = out_2d.shape
        return out_2d.view(B, T, C, H_out, W_out).permute(0, 2, 1, 3, 4).contiguous()


class SEBlock3D(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        r = max(ch // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(ch, r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(r, ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C = x.shape[:2]
        # x.mean(dim=(2, 3, 4)) is mathematically identical to AdaptiveAvgPool3d(1)
        # and runs natively on Apple Silicon MPS
        w = self.fc(x.mean(dim=(2, 3, 4))).view(B, C, 1, 1, 1)
        return x * w


class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.se = SEBlock3D(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(0.05)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.dropout(out)
        return self.relu(out + identity)


# ═══════════════════════════════════════════════════════════════
#  FULL MODEL — PhysFormerX v3
# ═══════════════════════════════════════════════════════════════
class PhysFormerX(nn.Module):
    """
    6-channel input (RGB + diff) → 3D ResNet → temporal conv → PPG signal
    ~2.02M params
    """
    def __init__(self, seq_len=SEQ_LEN):
        super().__init__()
        # Stem: 6ch → 32 (SpatialMaxPool3d is state_dict-identical to MaxPool3d((1,2,2)) and MPS-native)
        self.stem = nn.Sequential(
            nn.Conv3d(6, 32, kernel_size=(1, 5, 5), padding=(0, 2, 2),
                      bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            SpatialMaxPool3d()  # spatial /2
        )

        # Progressive feature extraction
        self.layer1 = ResBlock3D(32, 64, stride=(1, 2, 2))   # spatial /2
        self.layer2 = ResBlock3D(64, 64, stride=1)
        self.layer3 = ResBlock3D(64, 128, stride=(1, 2, 2))  # spatial /2
        self.layer4 = ResBlock3D(128, 128, stride=1)

        # Spatial collapse
        self.gap = nn.AdaptiveAvgPool3d((None, 1, 1))

        # Temporal processing: conv1d stack for smooth PPG extraction
        self.temporal = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # Final regression
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv1d(64, 1, kernel_size=1),
        )

    def forward(self, x):
        # x: [B, 6, T, H, W]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Spatial collapse over (H, W) — MPS-native equivalent of AdaptiveAvgPool3d((None, 1, 1))
        x = x.mean(dim=(-2, -1))  # [B, 128, T]

        x = self.temporal(x)   # [B, 64, T]
        x = self.head(x).squeeze(1)  # [B, T]

        return x
