"""Dataset audit for UBFC-rPPG + the legacy preprocessed windows.

Checks
  A. raw ground truth: length/duration vs video, timestamp gaps, flat-lines, clipping,
     PPG spectral SNR, PPG-derived HR vs oximeter HR (10 s windows)
  B. video <-> label consistency: HR of the classical POS signal from our face crops
     vs HR of the contact PPG (same windows) + best cross-correlation lag
  C. legacy windows (preprocessed_vidformer*): what does the stored 'ppg' label really contain?
Writes reports/dataset_audit.json, reports/dataset_audit.md and reports/figures/audit_*.png
"""

import os
import sys
import json
import glob
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vitaleye.ubfc import list_videos, video_info, load_ground_truth
from vitaleye.dsp import hr_from_signal, preprocess_pulse, skin_rgb_means, pos_signal

WIN_S, STEP_S, FS = 10.0, 5.0, 30.0


def windows(n, fs, win_s=WIN_S, step_s=STEP_S):
    w, s = int(win_s * fs), int(step_s * fs)
    return [(i, i + w) for i in range(0, n - w + 1, s)]


def longest_flat_run(x, fs):
    d = np.abs(np.diff(x)) < 1e-9
    best = cur = 0
    for v in d:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best / fs


def audit_raw(raw_dir, proc_dir):
    rows = []
    for sid, vpath, gpath in list_videos(raw_dir):
        info = video_info(vpath)
        gt = load_ground_truth(gpath, info["n_frames"], info["fps"])
        ppg = gt["ppg"]
        r = dict(sid=sid, kind=gt["kind"], n_frames=info["n_frames"], container_fps=info["fps"],
                 video_dur=info["n_frames"] / info["fps"], gt_dur=gt["gt_duration"])
        # raw timestamp gaps (ds2 only, raw file)
        if gt["kind"] == "ds2":
            t_raw = np.loadtxt(gpath)[2]
            dt = np.diff(t_raw)
            r["true_fps"] = (len(t_raw) - 1) / (t_raw[-1] - t_raw[0])
            r["max_ts_gap_s"] = float(dt.max())
            r["n_gt_samples"] = len(t_raw)
        else:
            r["true_fps"] = info["fps"]
            r["max_ts_gap_s"] = float(np.max(np.diff(np.loadtxt(gpath, delimiter=",")[:, 0]))) / 1000
            r["n_gt_samples"] = int(np.loadtxt(gpath, delimiter=",").shape[0])
        fs = r["true_fps"]
        r["flat_run_s"] = longest_flat_run(ppg, fs)
        lo, hi = np.percentile(ppg, [0.1, 99.9])
        r["clip_frac"] = float(np.mean((ppg <= ppg.min() + 1e-9) | (ppg >= ppg.max() - 1e-9)))
        hr_ppg, snr, hr_sensor = [], [], []
        for a, b in windows(len(ppg), fs):
            h, s = hr_from_signal(ppg[a:b], fs)
            hr_ppg.append(h); snr.append(s)
            hr_sensor.append(float(np.mean(gt["sensor_hr"][a:b])))
        hr_ppg, snr, hr_sensor = map(np.array, (hr_ppg, snr, hr_sensor))
        r["hr_ppg_mean"] = float(hr_ppg.mean()); r["hr_ppg_min"] = float(hr_ppg.min()); r["hr_ppg_max"] = float(hr_ppg.max())
        r["hr_sensor_mean"] = float(hr_sensor.mean())
        r["sensor_vs_ppg_mae"] = float(np.mean(np.abs(hr_ppg - hr_sensor)))
        r["frac_win_disagree10"] = float(np.mean(np.abs(hr_ppg - hr_sensor) > 10))
        r["gt_snr_db_median"] = float(np.median(snr))
        r["frac_win_low_snr"] = float(np.mean(snr < 0.0))
        r["max_hr_jump"] = float(np.max(np.abs(np.diff(hr_ppg)))) if len(hr_ppg) > 1 else 0.0
        r["_series"] = dict(hr_ppg=hr_ppg.tolist(), hr_sensor=hr_sensor.tolist(), snr=snr.tolist())

        # B: video/label consistency from the processed (30 Hz) crops
        npz = os.path.join(proc_dir, f"{sid}.npz")
        if os.path.exists(npz):
            z = np.load(npz)
            frames, ppg30 = z["frames"], z["ppg"]
            r["face_det_rate"] = float(z["det_rate"])
            rgb = skin_rgb_means(frames)
            pos = pos_signal(rgb, FS)
            hp, hv = [], []
            for a, b in windows(len(ppg30), FS):
                hp.append(hr_from_signal(pos[a:b], FS)[0]); hv.append(hr_from_signal(ppg30[a:b], FS)[0])
            hp, hv = np.array(hp), np.array(hv)
            r["pos_vs_gt_mae"] = float(np.mean(np.abs(hp - hv)))
            r["pos_agree_frac_5bpm"] = float(np.mean(np.abs(hp - hv) <= 5))
            # global lag between POS and contact PPG (positive = video leads)
            x, y = preprocess_pulse(pos, FS), preprocess_pulse(ppg30, FS)
            max_lag = int(2 * FS)
            cc = [np.corrcoef(x[max(0, -L):len(x) - max(0, L)], y[max(0, L):len(y) - max(0, -L)])[0, 1]
                  for L in range(-max_lag, max_lag + 1)]
            k = int(np.nanargmax(np.abs(cc)))
            r["pos_ppg_best_lag_s"] = (k - max_lag) / FS
            r["pos_ppg_best_corr"] = float(cc[k])
            r["_series"]["hr_pos"] = hp.tolist()
        rows.append(r)
        print(f"{sid:22s} fps={fs:5.2f} dur v/gt={r['video_dur']:.1f}/{r['gt_dur']:.1f}s  HR(ppg)={r['hr_ppg_mean']:.1f} "
              f"[{r['hr_ppg_min']:.0f}-{r['hr_ppg_max']:.0f}]  sensorMAE={r['sensor_vs_ppg_mae']:.1f}  "
              f"SNR={r['gt_snr_db_median']:.1f}dB  POS-MAE={r.get('pos_vs_gt_mae', float('nan')):.1f}  "
              f"lag={r.get('pos_ppg_best_lag_s', float('nan')):+.2f}s", flush=True)
    return rows


def audit_legacy(legacy_root, raw_rows, raw_dir, max_files=None):
    """Compare the 'ppg' label stored in legacy windows with (a) the contact PPG of the
    same subject and (b) the window's OWN video green-channel trace."""
    import torch
    gt_cache = {}
    for sid, vpath, gpath in list_videos(raw_dir):
        info = video_info(vpath)
        gt_cache[sid] = (load_ground_truth(gpath, info["n_frames"], info["fps"]), info)
    out = {}
    for name in ("preprocessed_vidformer1", "preprocessed_vidformer2"):
        d = os.path.join(legacy_root, name)
        files = sorted(glob.glob(os.path.join(d, "window_*.pt")), key=lambda p: int(p.split("_")[-1][:-3]))
        if max_files:
            files = files[:max_files]
        per_sub_count = {}
        res = []
        for fp in files:
            w = torch.load(fp, map_location="cpu", weights_only=False)
            sid = w["subject_id"]
            sid_n = sid if not sid.startswith("ds1_") else sid
            k = per_sub_count.get(sid, 0); per_sub_count[sid] = k + 1
            lab = w["ppg"].numpy().astype(np.float64)
            fps = float(w["fps"])
            vid = w["video"]
            g = vid[1].float()
            m = (vid.sum(0) > 0).float()
            green = ((g * m).sum((1, 2)) / m.sum((1, 2)).clamp(min=1)).numpy()
            entry = dict(file=os.path.basename(fp), sid=sid, fps=fps)
            entry["label_vs_own_green_corr"] = float(abs(np.corrcoef(preprocess_pulse(green, fps), preprocess_pulse(lab, fps))[0, 1]))
            gt, info = gt_cache.get(sid, (None, None))
            if gt is not None:
                a = k * 150
                seg = gt["ppg"][a:a + 300]
                if len(seg) == 300:
                    best = 0.0
                    for L in range(-60, 61):
                        aa = a + L
                        if aa < 0 or aa + 300 > len(gt["ppg"]):
                            continue
                        c = abs(np.corrcoef(preprocess_pulse(gt["ppg"][aa:aa + 300], fps), preprocess_pulse(lab, fps))[0, 1])
                        best = max(best, c)
                    entry["label_vs_contact_ppg_corr"] = float(best)
                    entry["hr_label"] = hr_from_signal(lab, fps)[0]
                    entry["hr_contact"] = hr_from_signal(seg, info["fps"] if gt["kind"] == "ds1" else len(gt["frame_t"]) / gt["gt_duration"])[0]
            if "hr_bpm" in w:
                entry["stored_hr_bpm"] = float(w["hr_bpm"])
            res.append(entry)
        out[name] = res
        lc = np.array([e.get("label_vs_contact_ppg_corr", np.nan) for e in res])
        gc = np.array([e["label_vs_own_green_corr"] for e in res])
        he = np.array([abs(e.get("hr_label", np.nan) - e.get("hr_contact", np.nan)) for e in res])
        print(f"[legacy] {name}: {len(res)} windows | median |corr(label, contact PPG)|={np.nanmedian(lc):.3f} "
              f"| median |corr(label, own video green)|={np.nanmedian(gc):.3f} | label-HR vs contact-HR MAE={np.nanmean(he):.2f} BPM "
              f"| windows with label-HR error >5 BPM: {np.nanmean(he > 5)*100:.1f}%")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=os.path.join(ROOT, "data", "raw", "UBFC-rPPG"))
    ap.add_argument("--proc_dir", default=os.path.join(ROOT, "data", "processed", "ubfc_v2"))
    ap.add_argument("--legacy_root", default=os.path.join(ROOT, "data", "legacy"))
    ap.add_argument("--skip_legacy", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "reports", "figures"), exist_ok=True)

    rows = audit_raw(a.raw_dir, a.proc_dir)
    legacy = {} if a.skip_legacy else audit_legacy(a.legacy_root, rows, a.raw_dir)

    # figure: GT HR timelines
    n = len(rows); cols = 5; rr = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rr, cols, figsize=(cols * 3.6, rr * 2.4), squeeze=False)
    for ax, r in zip(axs.flat, rows):
        s = r["_series"]; x = np.arange(len(s["hr_ppg"])) * STEP_S
        ax.plot(x, s["hr_ppg"], "k-", lw=1.4, label="contact PPG")
        ax.plot(x, s["hr_sensor"], "c--", lw=1, label="oximeter HR")
        if "hr_pos" in s:
            ax.plot(x, s["hr_pos"], "r:", lw=1, label="POS (video)")
        ax.set_title(r["sid"], fontsize=8); ax.tick_params(labelsize=6); ax.set_ylim(40, 180)
    for ax in list(axs.flat)[n:]:
        ax.axis("off")
    axs[0, 0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "reports", "figures", "audit_hr_timelines.png"), dpi=110); plt.close(fig)

    clean = [{k: v for k, v in r.items() if k != "_series"} for r in rows]
    with open(os.path.join(ROOT, "reports", "dataset_audit.json"), "w") as f:
        json.dump(dict(subjects=clean, legacy=legacy), f, indent=1)
    print("saved reports/dataset_audit.json")


if __name__ == "__main__":
    main()
