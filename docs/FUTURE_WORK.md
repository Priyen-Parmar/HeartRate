# Status and future work

Where the project stands, what is still unsolved, and what the next steps should be.

---

## 1. Status

**Works and is measured**
* On UBFC-rPPG, on unseen people, the error is **1.45 BPM** in normal light.
* The full live app reaches 1.59 BPM end to end on the raw dataset videos.
* Readings that pass the honesty gate reach 1.17 BPM.
* The live app runs on NVIDIA GPUs, Apple silicon and plain CPUs. It uses real frame timestamps, so a
  camera with an uneven frame rate is handled correctly.
* Training, evaluation and reporting are reproducible from `pipeline/` (see the root README).

**Open**
* Real-world accuracy on ordinary webcams is **unmeasured**.
* In every real session recorded so far, the pulse signal was weak (see §2).

---

## 2. The open problem: a weak pulse on real webcams

| Source | Signal SNR (dB) |
|---|---|
| UBFC-rPPG videos (median) | +3.75 |
| UBFC-rPPG, worst video | −1.4 |
| Laptop webcam sessions (3) | −2.0 to −3.1 |
| Phone camera via DroidCam | −2.7 |
| MacBook (M4) FaceTime camera | −2.1 |

At these SNRs the honesty gate correctly hides most readings. In the MacBook session, a patch of wall
with no pulse scored the same SNR as the face. The recording therefore held no measurable pulse, and
the failure was in what the camera captured, not in the HR estimate. Backlight with auto-exposure was
the dominant measured cause in that session.

Details are in [DEVELOPMENT_LOG.md §8–13](DEVELOPMENT_LOG.md).

**Hypotheses, not yet separated**

1. **Domain shift of the model.**
   * The training data is one dataset, one camera, mostly lighter skin and bright, even light.
   * Darker skin returns a weaker pulse. Published studies report much larger errors for existing
     methods on darker skin tones.
2. **Camera pipeline.**
   * Compression and duplicated frames from streaming apps.
   * Phone image processing: noise reduction, beauty filters, HDR, stabilisation.
   * Webcam MJPG compression, and automatic exposure and white balance.
3. **Geometry.** The face may be too close, so the crop runs off the frame, or too far.
4. **Motion and talking** during the sessions.

**The deciding experiment**

1. Record a session following [USAGE.md §6](USAGE.md#6-testing-against-a-reference-device): reference
   readings every 15 s, the face at about one third of the frame height, light in front, camera effects
   off, `--save_raw auto`.
2. Ideally, also record one minute with the phone's own camera app and replay it with `--video`.
3. Analyse the recording offline. For every 10-s window, compare the network's HR and SNR with the
   POS / CHROM / GREEN baselines (`optipulse.dsp`), and with the same pipeline run on a crop of
   background.
4. Interpret the result:
   * If a classical method finds the reference HR with a decent SNR but the network does not, it is a
     **model or domain problem**.
   * If nothing finds it, it is a **capture problem**.

---

## 3. Next steps, in priority order

1. **Collect reference-labelled real sessions** (§2). Without them, no change to the live system can be
   evaluated.
2. If it is a capture problem:
   * lock exposure and white balance where the camera allows it;
   * record with the phone's native camera;
   * give users clearer framing and lighting guidance in the UI.
3. If it is a model problem:
   1. **Fuse a classical candidate** (POS/CHROM spectrum) into the tracker evidence. This is cheap.
      Validate it on UBFC first so it does not hurt there.
   2. **Unsupervised test-time or personal adaptation** on unlabelled real recordings, using SiNC-style
      losses (bandwidth, sparsity, variance) or Contrast-Phys. Evaluate it against reference readings
      that the adaptation never saw.
   3. **Multi-dataset training** (§4). Then test across datasets: train on some, test on another.
4. **Validate on real low-light and real motion videos**, not only on simulated ones.

**Smaller engineering items**
* Fix the stream-URL camera bug ([EXPLAIN.md §13](EXPLAIN.md#13-limitations-and-known-issues)).
* Let `train_cv.py` and `eval_tracking.py` run without CUDA, at least for evaluation.
* Make the `train_cv.py` defaults match a deployed family, or require `--version`.
* Add a small automated regression test (for example, one UBFC subject, or a short synthetic clip with a
  known frequency).

---

## 4. Candidate datasets

None of these have been downloaded. Most need a signed request, usually through a supervising faculty
member.

| Dataset | Content | Why useful |
|---|---|---|
| **MMPD** (EMBC 2023) | 33 people, 11 h, phone camera, Fitzpatrick skin types 3–6, varied light and motion | closest to real phone and webcam use; diverse skin tones |
| **VitalVideos** | ~900 people, all six skin types, PPG and blood pressure | largest and most diverse |
| **UBFC-Phys** | 56 people, talking and stress tasks | realistic motion |
| **PURE** | 10 people, controlled head motion | the standard cross-dataset test set |
| **iBVP** | RGB + thermal, PPG quality labels | clean labels |
| **VIPL-HR** | 107 people, 1,150 videos, several devices including phones | multi-device |
| **SCAMPS** | 2,800 synthetic videos (public) | pre-training |
| **UCLA-rPPG / Diverse-rPPG** | ~50 people with diverse skin tones | skin-tone balance |

Most of these have loaders in [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox).

A suggested plan is to train on UBFC-rPPG, MMPD and UBFC-Phys, then test on PURE and on the new real
recordings.

---

## 5. Research directions

The current contributions are listed in [METHOD.md](METHOD.md). Before any publication, three things
are needed:
* validation on a second dataset (cross-dataset);
* real, not simulated, low-light recordings;
* a range of skin tones.

A stronger story would combine **cross-dataset and cross-device generalisation** with **unsupervised
domain or personal adaptation** on unlabelled video. Adaptation uses no reference labels, so it cannot
overfit to the readings used for checking.
