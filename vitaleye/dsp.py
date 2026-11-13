"""Signal processing shared by training, evaluation and the live monitor.

The SAME heart-rate estimator is applied to ground-truth PPG and to predicted
rPPG, so label and prediction are measured identically. There is no
physiological prior (no "pull towards 70-80 BPM"): the HR is the dominant
frequency inside a wide 39-192 BPM band.
"""

import numpy as np
from scipy.signal import butter, filtfilt, periodogram
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve

from . import HR_LO_HZ, HR_HI_HZ


def detrend(sig: np.ndarray, lam: float = 100.0) -> np.ndarray:
    """Smoothness-priors detrending (Tarvainen et al. 2002), as used by rPPG-Toolbox."""
    x = np.asarray(sig, dtype=np.float64)
    n = len(x)
    if n < 4:
        return x - x.mean()
    d2 = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
    trend = spsolve((eye(n) + lam**2 * (d2.T @ d2)).tocsc(), x)
    return x - trend


def bandpass(sig: np.ndarray, fs: float, lo: float = HR_LO_HZ, hi: float = HR_HI_HZ, order: int = 2) -> np.ndarray:
    x = np.asarray(sig, dtype=np.float64)
    nyq = fs / 2.0
    b, a = butter(order, [lo / nyq, min(hi / nyq, 0.99)], btype="band")
    if len(x) <= 3 * max(len(a), len(b)):
        return x - x.mean()
    return filtfilt(b, a, x)


def preprocess_pulse(sig: np.ndarray, fs: float) -> np.ndarray:
    """detrend -> bandpass -> z-score. Used for labels, predictions and classical methods."""
    x = bandpass(detrend(sig), fs)
    return (x - x.mean()) / (x.std() + 1e-8)


def spectrum(sig: np.ndarray, fs: float, nfft_min: int = 4096):
    n = len(sig)
    nfft = max(nfft_min, 1 << int(np.ceil(np.log2(max(n, 2)))))
    f, p = periodogram(sig, fs=fs, nfft=nfft, detrend=False, window="hann")
    return f, p


def hr_from_signal(sig: np.ndarray, fs: float, lo: float = HR_LO_HZ, hi: float = HR_HI_HZ, already_filtered: bool = False):
    """Return (hr_bpm, snr_db). Dominant spectral peak in [lo, hi] with parabolic refinement.

    SNR follows de Haan & Jeanne (2013): power within +-0.1 Hz of the fundamental and
    its first harmonic versus the remaining power in the band.
    """
    x = sig if already_filtered else preprocess_pulse(sig, fs)
    f, p = spectrum(x, fs)
    band = np.where((f >= lo) & (f <= hi))[0]
    if len(band) < 3:
        return float("nan"), float("nan")
    k = band[np.argmax(p[band])]
    fk = f[k]
    if 0 < k < len(p) - 1:
        y0, y1, y2 = np.log(p[k - 1] + 1e-20), np.log(p[k] + 1e-20), np.log(p[k + 1] + 1e-20)
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-12:
            fk = fk + np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5) * (f[1] - f[0])
    sig_mask = (np.abs(f - fk) <= 0.1) | (np.abs(f - 2 * fk) <= 0.2)
    in_band = (f >= lo) & (f <= max(hi, 2 * fk + 0.2))
    s = p[sig_mask & in_band].sum()
    n = p[in_band & ~sig_mask].sum() + 1e-20
    return float(fk * 60.0), float(10 * np.log10(s / n + 1e-20))


def hr_from_waves(waves, fs: float, combine: str = "psd", lo: float = HR_LO_HZ, hi: float = HR_HI_HZ):
    """Combine several model outputs for the same clip -> (hr_bpm, snr_db, fused_wave).
    combine='wave'  : average z-scored waveforms, then estimate
            'psd'   : average each model's band-normalised power spectrum (phase-insensitive)
            'median': median of the per-model HR estimates (SNR = median SNR)"""
    filt = [preprocess_pulse(w, fs) for w in waves]
    fused = np.mean(filt, axis=0)
    if combine == "wave" or len(waves) == 1:
        h, s = hr_from_signal(fused, fs, lo, hi, already_filtered=True)
        return h, s, fused
    if combine == "median":
        est = [hr_from_signal(w, fs, lo, hi, already_filtered=True) for w in filt]
        return float(np.median([e[0] for e in est])), float(np.median([e[1] for e in est])), fused
    # psd fusion
    f, p0 = spectrum(filt[0], fs)
    band = (f >= lo) & (f <= hi)
    P = np.zeros_like(p0)
    for w in filt:
        _, p = spectrum(w, fs)
        P += p / (p[band].sum() + 1e-20)
    idx = np.where(band)[0]
    k = idx[np.argmax(P[idx])]
    fk = f[k]
    if 0 < k < len(P) - 1:
        y0, y1, y2 = np.log(P[k - 1] + 1e-20), np.log(P[k] + 1e-20), np.log(P[k + 1] + 1e-20)
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-12:
            fk = fk + np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5) * (f[1] - f[0])
    sig = (np.abs(f - fk) <= 0.1) | (np.abs(f - 2 * fk) <= 0.2)
    inb = (f >= lo) & (f <= max(hi, 2 * fk + 0.2))
    snr = 10 * np.log10(P[sig & inb].sum() / (P[inb & ~sig].sum() + 1e-20) + 1e-20)
    return float(fk * 60.0), float(snr), fused


def resample_uniform(t_src: np.ndarray, x_src: np.ndarray, fs_out: float):
    """Linear resampling of x_src (time on axis 0) from timestamps t_src to a uniform grid."""
    t_src = np.asarray(t_src, dtype=np.float64)
    t_out = np.arange(t_src[0], t_src[-1] + 1e-9, 1.0 / fs_out)
    idx = np.searchsorted(t_src, t_out, side="right") - 1
    idx = np.clip(idx, 0, len(t_src) - 2)
    w = (t_out - t_src[idx]) / np.maximum(t_src[idx + 1] - t_src[idx], 1e-9)
    w = np.clip(w, 0.0, 1.0)
    shape = (-1,) + (1,) * (np.ndim(x_src) - 1)
    w = w.reshape(shape)
    x0 = x_src[idx].astype(np.float32)
    x1 = x_src[idx + 1].astype(np.float32)
    return t_out, (1 - w) * x0 + w * x1


# ----------------------------------------------------------------------------
# Classical (unsupervised) rPPG baselines, operating on per-frame mean RGB
# ----------------------------------------------------------------------------
def skin_rgb_means(frames: np.ndarray) -> np.ndarray:
    """frames: [T,H,W,3] uint8 RGB face crops -> [T,3] mean RGB over a central face ellipse
    restricted to plausible skin chroma (simple and flicker-free)."""
    T, H, W, _ = frames.shape
    yy, xx = np.mgrid[0:H, 0:W]
    ell = (((xx - W / 2) / (W * 0.30)) ** 2 + ((yy - H * 0.50) / (H * 0.36)) ** 2) <= 1.0
    sub = frames[:, ell, :].astype(np.float32)  # [T,N,3]
    return sub.mean(axis=1)


def pos_signal(rgb: np.ndarray, fs: float) -> np.ndarray:
    """POS - Wang et al., 'Algorithmic Principles of Remote PPG', IEEE TBME 2017."""
    T = len(rgb)
    win = int(round(1.6 * fs))
    h = np.zeros(T)
    for s in range(0, T - win + 1):
        c = rgb[s:s + win]
        cn = c / (c.mean(axis=0, keepdims=True) + 1e-9)
        x1 = cn[:, 1] - cn[:, 2]
        x2 = cn[:, 1] + cn[:, 2] - 2 * cn[:, 0]
        hh = x1 + (x1.std() / (x2.std() + 1e-9)) * x2
        h[s:s + win] += hh - hh.mean()
    return h


def chrom_signal(rgb: np.ndarray, fs: float) -> np.ndarray:
    """CHROM - de Haan & Jeanne, IEEE TBME 2013."""
    cn = rgb / (rgb.mean(axis=0, keepdims=True) + 1e-9)
    xs = bandpass(3 * cn[:, 0] - 2 * cn[:, 1], fs)
    ys = bandpass(1.5 * cn[:, 0] + cn[:, 1] - 1.5 * cn[:, 2], fs)
    return xs - (xs.std() / (ys.std() + 1e-9)) * ys


def green_signal(rgb: np.ndarray, fs: float) -> np.ndarray:
    return rgb[:, 1].copy()
