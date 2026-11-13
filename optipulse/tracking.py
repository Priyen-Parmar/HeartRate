"""Causal spectral HR tracking (forward HMM filtering over a 1-BPM grid).

Why: in low light ~35 % of 10-s windows have a noise peak that beats the pulse peak (errors > 10 BPM,
non-harmonic, random). A single window cannot tell them apart, but the TRUE pulse peak is consistent
over consecutive (overlapping) windows while noise peaks jump around.

What it is (and is not):
  * state  = heart rate on a 1-BPM grid spanning the full 39-192 BPM band
  * prior  = UNIFORM over the band (no pull towards 70-80 BPM, no population prior)
  * motion = Gaussian random walk of sigma_bpm per second (continuity only: HR cannot jump 40 BPM in
             one second, but it can drift anywhere; 4 BPM/s comfortably follows post-exercise recovery)
  * evidence = the ensemble's band-normalised power spectrum of the current 10-s window, sharpened
             by `alpha`, with a uniform floor so that a wrong past never locks the tracker
  * strictly CAUSAL: the estimate at time t only uses windows ending <= t (usable live)

Inspired by spectral peak tracking for wearable PPG (TROIKA, Zhang et al. TBME'15; HMM/Viterbi HR
trackers) - here applied to the posterior-like spectrum of a deep rPPG ensemble.
"""

import numpy as np

from . import HR_LO_HZ, HR_HI_HZ
from .dsp import preprocess_pulse, spectrum

BPM_GRID = np.arange(round(HR_LO_HZ * 60), round(HR_HI_HZ * 60) + 1, 1.0)


def band_psd(waves, fs: float) -> np.ndarray:
    """Average band-normalised PSD of several model waveforms, sampled on BPM_GRID (sums to 1)."""
    acc = np.zeros(len(BPM_GRID))
    for w in waves:
        f, p = spectrum(preprocess_pulse(w, fs), fs)
        q = np.interp(BPM_GRID / 60.0, f, p)
        acc += q / (q.sum() + 1e-20)
    return acc / max(len(waves), 1)


class HRTracker:
    def __init__(self, sigma_bpm_per_s: float = 4.0, alpha: float = 2.0, floor: float = 0.02, ref_hop_s: float = 1.0):
        self.sigma = sigma_bpm_per_s
        self.alpha = alpha
        self.floor = floor
        # evidence tempering: consecutive 10-s windows overlap, so the same data is seen again at every
        # update. The tracker was specified (and evaluated) for one update per ref_hop_s; faster updates
        # (live app: every 0.5 s) get proportionally less weight so the posterior is NOT over-confident.
        self.ref_hop = ref_hop_s
        self.post = None
        self.t_last = None

    def reset(self):
        self.post, self.t_last = None, None

    def _transition(self, dt):
        s = max(self.sigma * max(dt, 1e-3), 0.5)
        d = BPM_GRID[:, None] - BPM_GRID[None, :]
        k = np.exp(-0.5 * (d / s) ** 2)
        return k / k.sum(0, keepdims=True)

    def update(self, psd: np.ndarray, t: float):
        """psd on BPM_GRID (any scale). Returns (hr_bpm, confidence in [0,1])."""
        wgt = 1.0 if self.t_last is None else float(np.clip((t - self.t_last) / self.ref_hop, 0.05, 1.0))
        like = psd / (psd.max() + 1e-20)
        like = like ** self.alpha
        like = like / like.sum()
        like = ((1 - self.floor) * like + self.floor / len(like)) ** wgt
        if self.post is None:
            prior = np.full(len(BPM_GRID), 1.0 / len(BPM_GRID))
        else:
            prior = self._transition(t - self.t_last) @ self.post
        post = prior * like
        self.post = post / post.sum()
        self.t_last = t
        k = int(np.argmax(self.post))
        # sub-BPM refinement: posterior mean in +-3 BPM around the MAP
        lo, hi = max(0, k - 3), min(len(BPM_GRID), k + 4)
        w = self.post[lo:hi]
        hr = float((BPM_GRID[lo:hi] * w).sum() / w.sum())
        conf = float(self.post[max(0, k - 3):k + 4].sum())
        return hr, conf
