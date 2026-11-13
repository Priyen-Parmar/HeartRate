# OptiPulse: noise-aware remote heart-rate estimation with linear-first region combining, cross-illumination consistency and causal spectral tracking

*Draft write-up of the method. Results: `results/RESULTS.md`.*

## Problem

Remote photoplethysmography (rPPG) reads the heart rate from tiny colour changes of facial skin.
On UBFC-rPPG the pulse is ~0.2-1 % of pixel intensity, and the video is MPEG-4 compressed at ~0.8 Mbit/s.
Classical methods (GREEN, CHROM, POS) fail on this copy (MAE 15-20 BPM). Deep models do well in good light,
but they degrade sharply when the face is dark and noisy. Our run-1 baseline gave **1.4 BPM in normal light
but 10.9 BPM in low light**. The error analysis (`results/RESULTS.md`, fold-0 diagnosis) showed that:

* darkening alone (x0.1 + 8-bit quantisation) costs little (1.06 -> 1.59 BPM);
* sensor noise alone costs little (-> 2.5 BPM);
* **both together** -> 8.8 BPM: 35 % of 10-s windows have a noise peak that beats the pulse peak. These
  errors are random and non-harmonic, and they come with negative spectral SNR;
* test-time spatial blurring makes it *worse* (the network relies on fine structure), so there is no
  quick fix at inference.

## Contributions (what is new, what is borrowed)

We combine ideas from several lines of work. None of the parts below is copied as-is from one paper.

1. **Linear-first region branch with learned maximal-ratio combining (MRC)** (`optipulse/model.py: RegionBranch`).
   *Inspiration:* DistancePPG (Kumar et al. 2015) combines facial regions with weights equal to their
   SNR (maximal-ratio combining). Multi-region spatio-temporal maps come from RhythmNet (Niu et al. 2020).
   *What we change:* the regions are **soft-skin-weighted**, using the attention mask learned by the pixel
   CNN (DeepPhys/TS-CAN-style attention), at three scales (6x6, 3x3, 1x1). Pixels are averaged **linearly
   and in float32 before any non-linearity**, so sensor noise drops by sqrt(N) while the pulse survives.
   A deep pixel path applies BN+ELU at full resolution and partly destroys a pulse that sits 40 dB below the
   per-pixel noise. Each region trace is diff-normalised (gain-invariant) and whitened by its own noise scale.
   A shared dilated 1-D CNN processes every region, and the regions are fused with *time-varying* softmax
   reliability weights. Those weights are conditioned on each region's relative noise level, so this is a
   learned, time-varying MRC.
2. **Noise-gated two-path fusion.** The network measures the noise level of its own input: per-pixel
   2nd-order temporal noise relative to intensity, and the region noise scale. A small MLP turns these into
   gates that weigh the pixel path against the region path. The model can lean on the region path in the
   dark and on the pixel path in good light. We know of no rPPG network that conditions its fusion on a
   measured noise level.
3. **Cross-illumination self-consistency.** Each training clip is shown twice, with *identical geometry and
   motion* but two independent photometric draws; the second is always forced into low light. On top of
   the supervised loss (neg-Pearson + spectral KL on the contact PPG), the dark view must reproduce the
   waveform the model itself extracts from the well-lit view (stop-gradient teacher).
   *Inspiration:* self-supervised rPPG (Contrast-Phys, Sun & Li 2022; SiNC, Speth et al. 2023) and
   consistency regularisation (Mean-Teacher / FixMatch style). *What is new:* the target is the model's
   own *facial* pulse, which shares the phase and shape of the skin signal. The contact-PPG label has a
   finger-to-face transit delay and a different morphology. The pairing isolates illumination as the
   only nuisance.
4. **Heterogeneous ensemble with spectral fusion.** We ensemble the pixel model (v1) with the
   region-MRC model (v4). They fail in different windows, so averaging their band-normalised power
   spectra (not their waveforms, which can differ in phase) beats either one alone.
5. **Causal spectral HMM tracking** (`optipulse/tracking.py`). The state is the HR on a 1-BPM grid over
   39-192 BPM with a **uniform prior**, so there is no pull towards 70-80 BPM. The only constraint is
   continuity: a Gaussian random walk of 4 BPM/s, which still follows post-exercise recovery. The
   evidence is the sharpened ensemble spectrum with a uniform floor, so the tracker cannot lock forever.
   The decoder is strictly causal: it runs live and uses only past windows.
   *Inspiration:* spectral peak tracking for wrist PPG (TROIKA, Zhang et al. 2015; HMM HR trackers).
   *New here:* it is applied to the spectrum of a deep rPPG ensemble. The tracking parameters were set
   a priori (physiology), not tuned on test data.
6. **Training recipe for unbiased HR** (from run 1):
   * HR-resampling augmentation: clips are time-stretched so the label HR is spread uniformly over
     45-180 BPM. This prevents the "typical HR" bias that made the old model report ~70-100 for everyone.
   * Webcam-realistic photometric and geometric augmentation: low light, shot and read noise, 8-bit
     requantisation, colour cast, side light, illumination drift and exposure steps, per-frame motion
     and jumps, 10-25 fps capture re-interpolated to 30 Hz.
   * A ground-truth audit that keeps subjects with broken contact PPG out of training.

## Evaluation protocol (why the numbers are honest)

* Subject-wise 5-fold CV. Every subject is scored only by models that never saw that person, and the
  deployed ensemble is additionally tested on 7 people **no** model was ever trained on.
* Fixed training schedule (30 epochs, last epoch). There is no early stopping or checkpoint picking on
  test folds.
* The same HR estimator is applied to the prediction and to the contact-PPG label, with no HR prior.
* Stress tests are deterministic and identical for all methods: ~10x darker + sensor noise +
  quantisation, head sway/rotation/jumps, their combination, and a 12 fps webcam.
* Design decisions were made on fold 0 only (development ablation). Fold-0 differences of about 1 BPM
  are within run-to-run noise (identical re-runs moved by that much), so only large, consistent effects
  were acted on.

## Limitations (stated plainly)

* **MAE cannot be 0.** The reference itself (finger contact PPG, 10-s spectral HR) has about +-1 BPM
  uncertainty, and the face-to-finger pulse transit and HR changes inside a window add more. About 1 BPM
  in good light is at the reference's noise floor.
* UBFC-rPPG is small (42-50 subjects) and nearly all lighter-skinned, indoors and seated. Low light and
  motion are **simulated**. Real low-light webcams (auto-exposure, real sensor noise before compression)
  can behave differently, so the live CSV logs should be checked against a reference device.
* Low light remains the hardest condition. The full system reduces the error substantially there, but
  it is not at the normal-light level.

## Results in brief (5-fold CV, 43 subjects with reliable ground truth, MAE in BPM)

| Method | normal | low light | motion | low light + motion | 12 fps |
|---|---|---|---|---|---|
| POS (classical) | 20.56 | 25.97 | 29.17 | 27.64 | 22.74 |
| Legacy PhysFormerX monitor (old project, normal video only) | 6.81 | - | - | - | - |
| VitalNet-Pix (single) | 1.42 | 10.91 | 4.06 | 11.47 | 1.50 |
| VitalNet-MRC (single) | 1.89 | 10.09 | 4.42 | 11.20 | 1.85 |
| Pix + MRC spectral fusion | 1.46 | 9.77 | 3.59 | 10.72 | 1.43 |
| **Full system (+ causal tracking)** | **1.45** | **7.55** | **3.12** | **8.23** | 1.59 |

What carried the gain (honest attribution):
* **Tracking** is the largest single effect in hard conditions (Pix: low light 10.9 -> 8.0).
* **The MRC model alone is not better than the pixel model** in normal light (1.89 vs 1.42). It is slightly
  better in low light (10.09 vs 10.91). Its value is **complementarity**: fusing the two lowers motion error
  by 12 % and low-light error by 10 % *before* tracking.
* Combined: low light **-31 %**, motion **-23 %**, low light + motion **-28 %**, normal light unchanged (~1.4 BPM).
* Fold-0 development numbers were more optimistic (e.g. low light 5.9). This shows why the 5-fold result is
  the number to quote.

Publishable framing (if extended): "noise-aware two-path rPPG with learned region MRC and spectral-posterior
tracking". Before a paper, it would need to be validated on a second dataset (PURE, cross-dataset), on real
(not simulated) low-light recordings, and with darker skin tones.
