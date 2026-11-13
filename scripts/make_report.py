"""Build reports/RESULTS.md + figures from the evaluation JSONs.

inputs (reports/):
  cv_results_pix.json, cv_results_mrc.json      per-window CV results of the two model families (+ baselines)
  tracking_{pix,mrc,system}.json                CV results with / without causal tracking (eval_tracking.py)
  tracking_system_never_trained.json            full deployed ensemble on the 7 never-trained subjects
  cv_results_*fold0.json, tracking_*fold0.json  development ablations on fold 0 (optional)
  pipeline_comparison.json, dataset_audit.json  (optional)
"""

import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vitaleye.evaluation import metrics
from vitaleye.ubfc import audit_flagged

REP = os.path.join(ROOT, "reports")
FIG = os.path.join(REP, "figures")
CONDS = ["clean", "lowlight", "motion", "lowlight+motion", "lowfps"]
SHORT = {"clean": "normal", "lowlight": "low light", "motion": "motion", "lowlight+motion": "low light + motion", "lowfps": "12 fps"}
FOLD0 = ['ds1_12gt', 'ds1_6gt', 'sub42', 'sub1', 'sub12', 'sub27', 'sub48', 'sub41', 'sub9', 'sub46']


def load(name):
    p = os.path.join(REP, name)
    return json.load(open(p)) if os.path.exists(p) else None


def cv_pairs(pv, subs, cond, meth):
    return [p for s in subs if s in pv for p in pv[s][cond][meth]]


def trk_pairs(tj, subs, cond, which):
    """which: 0 = plain, 1 = tracked. rows are (plain, tracked, gt, snr)."""
    ps = tj["per_subject"]
    return [(r[which], r[2], r[3]) for s in subs if s in ps for r in ps[s][cond]]


def row(name, ms, key="MAE", d=2):
    return f"| {name} | " + " | ".join(f"{m.get(key, float('nan')):.{d}f}" if m else "-" for m in ms) + " |"


def main():
    os.makedirs(FIG, exist_ok=True)
    flagged = set(audit_flagged(os.path.join(REP, "dataset_audit.json")))
    pix, mrc = load("cv_results_pix.json"), load("cv_results_mrc.json")
    t_pix, t_mrc, t_sys = load("tracking_pix.json"), load("tracking_mrc.json"), load("tracking_system.json")
    t_nt = load("tracking_system_never_trained.json")
    sids = sorted(pix["per_video"])
    rel = [s for s in sids if s not in flagged]
    L = ["# VitalEye - Results\n",
         "**Protocol.** Subject-wise 5-fold cross-validation on UBFC-rPPG (50 videos): every number below comes from "
         "models that never saw that person. Heart rate per non-overlapping 10-s window, same spectral estimator for "
         "prediction and contact-PPG ground truth (39-192 BPM, no HR prior). Fixed training schedule, last epoch, no "
         "checkpoint selection on test data. Stress conditions are deterministic (same seed for every method). "
         f"Main tables: the {len(rel)} subjects whose contact-PPG passed the audit; flagged subjects "
         f"({', '.join(sorted(flagged))}) are reported separately.\n",
         "**Tracked** = causal HMM decoding (uses only the current and past 10-s windows; usable live).\n"]

    # ---------------------------------------------------------------- main table
    L.append(f"\n## 1. Main result - MAE (BPM) on unseen subjects, reliable ground truth (n={len(rel)})\n")
    L.append("| Method | " + " | ".join(SHORT[c] for c in CONDS) + " |")
    L.append("|---|" + "---|" * len(CONDS))
    for meth in ("GREEN", "CHROM", "POS"):
        L.append(row(meth + " (classical)", [metrics(cv_pairs(pix["per_video"], rel, c, meth)) for c in CONDS]))
    L.append(row("VitalNet-Pix (v1, single model)", [metrics(trk_pairs(t_pix, rel, c, 0)) for c in CONDS]))
    L.append(row("VitalNet-MRC (v4 + consistency, single model)", [metrics(trk_pairs(t_mrc, rel, c, 0)) for c in CONDS]))
    L.append(row("Pix + tracking", [metrics(trk_pairs(t_pix, rel, c, 1)) for c in CONDS]))
    L.append(row("MRC + tracking", [metrics(trk_pairs(t_mrc, rel, c, 1)) for c in CONDS]))
    L.append(row("Pix + MRC spectral fusion (no tracking)", [metrics(trk_pairs(t_sys, rel, c, 0)) for c in CONDS]))
    L.append(row("**VitalEye full system** (Pix + MRC fusion + tracking)", [metrics(trk_pairs(t_sys, rel, c, 1)) for c in CONDS]))
    L.append("\nFull system, all metrics:\n")
    L.append("| Condition | MAE | RMSE | MAPE % | Pearson r | within 5 BPM | within 10 BPM | windows |")
    L.append("|---|---|---|---|---|---|---|---|")
    for c in CONDS:
        pr = trk_pairs(t_sys, rel, c, 1); m = metrics(pr)
        w10 = 100 * np.mean([abs(a - b) <= 10 for a, b, _ in pr])
        L.append(f"| {SHORT[c]} | {m['MAE']:.2f} | {m['RMSE']:.2f} | {m['MAPE']:.1f} | {m['r']:.3f} | {m['within5']:.1f}% | {w10:.1f}% | {m['n']} |")

    # ------------------------------------------------------------- flagged / never trained
    flag_s = [s for s in sids if s in flagged]
    L.append(f"\n## 2. Audit-flagged subjects (n={len(flag_s)}, CV held-out models, full system)\n")
    L.append("| Subject | " + " | ".join(SHORT[c] for c in CONDS) + " |")
    L.append("|---|" + "---|" * len(CONDS))
    for s in flag_s:
        L.append(row(s, [metrics(trk_pairs(t_sys, [s], c, 1)) for c in CONDS]))
    if t_nt:
        L.append("\n### Deployed 10-model ensemble on the 7 people no model was ever trained on\n")
        L.append("| Subjects | " + " | ".join(SHORT[c] for c in CONDS) + " |")
        L.append("|---|" + "---|" * len(CONDS))
        for name, ss in (("sub11/18/20/24 (PPG verified)", ["sub11", "sub18", "sub20", "sub24"]),
                         ("sub3/4/14 (noisy contact PPG)", ["sub3", "sub4", "sub14"])):
            L.append(row(name + " plain", [metrics(trk_pairs(t_nt, ss, c, 0)) for c in CONDS]))
            L.append(row(name + " tracked", [metrics(trk_pairs(t_nt, ss, c, 1)) for c in CONDS]))
        L.append("\nOnly 7 people, so this is a small-sample check: in low light tracking *hurt* the 4 verified "
                 "subjects here (the tracker can follow a wrong peak for a few windows, e.g. sub11), although it "
                 "helped clearly on the 43-subject CV (Section 1). The live app has `--no_track` to switch it off.")

    # ------------------------------------------------------------------ fold-0 ablation
    dev = [("run 1 = Pix v1", "cv_results_pix.json", "tracking_pix.json"),
           ("run 2: harder augmentation, 40 ep", "cv_results_run2_partial.json", None),
           ("v3: temporal pre-filter stem", "cv_results_v3_fold0.json", None),
           ("v4: + region-MRC branch + noise gate", "cv_results_v4_fold0.json", "tracking_v4_fold0.json"),
           ("v4 + low-light consistency (= MRC)", "cv_results_mrc.json", "tracking_v4pair_fold0.json"),
           ("ens Pix + v4", None, "tracking_ens_r1_v4_fold0.json"),
           ("ens Pix + MRC", None, "tracking_ens_r1_v4p_fold0.json"),
           ("ens Pix + v4 + MRC", None, "tracking_ens_r1_v4_v4p_fold0.json")]
    L.append(f"\n## 3. Development ablation (fold 0 only, {len(FOLD0)} unseen subjects, MAE BPM)\n")
    L.append("Used to choose the design; single-fold numbers move by +-1 BPM between identical re-runs, so only "
             "large effects were trusted, and the chosen system was then evaluated on all 5 folds (Section 1).\n")
    L.append("| Variant | " + " | ".join(SHORT[c] for c in CONDS) + " | " + " | ".join(SHORT[c] + " tracked" for c in CONDS[:4]) + " |")
    L.append("|---|" + "---|" * (len(CONDS) + 4))
    for name, cvf, tf in dev:
        cvj, tj = load(cvf) if cvf else None, load(tf) if tf else None
        if cvj:
            plain = [metrics(cv_pairs(cvj["per_video"], FOLD0, c, "vitalnet")) for c in CONDS]
        elif tj:
            plain = [metrics(trk_pairs(tj, FOLD0, c, 0)) for c in CONDS]
        else:
            continue
        tr = [metrics(trk_pairs(tj, FOLD0, c, 1)) for c in CONDS[:4]] if tj else [None] * 4
        L.append(row(name, plain + tr))

    # ------------------------------------------------------------------ per subject
    L.append("\n## 4. Per-subject MAE, full system (CV)\n")
    L.append("| Subject | GT reliable | mean GT HR | " + " | ".join(SHORT[c] for c in CONDS) + " |")
    L.append("|---|---|---|" + "---|" * len(CONDS))
    for s in sids:
        g = np.mean([r[2] for r in t_sys["per_subject"][s]["clean"]])
        L.append(f"| {s} | {'no' if s in flagged else 'yes'} | {g:.0f} | " +
                 " | ".join(f"{metrics(trk_pairs(t_sys, [s], c, 1))['MAE']:.1f}" for c in CONDS) + " |")

    # ------------------------------------------------------------------ SNR calibration
    a = np.array([r for s in rel for c in CONDS for r in t_sys["per_subject"][s][c]])
    err = np.abs(a[:, 0] - a[:, 2]); snr = a[:, 3]
    L.append("\n## 5. Signal quality (ensemble SNR) vs per-window error (no tracking)\n")
    L.append("| SNR | windows | MAE | within 5 BPM |")
    L.append("|---|---|---|---|")
    for lo, hi in zip([-99, -6, -3, 0, 3, 6], [-6, -3, 0, 3, 6, 99]):
        k = (snr >= lo) & (snr < hi)
        if k.sum():
            L.append(f"| {lo} .. {hi} dB | {k.sum()} | {err[k].mean():.2f} | {100*np.mean(err[k]<=5):.1f}% |")

    pc = load("pipeline_comparison.json")
    if pc:
        L.append("\n## 6. Full live pipeline, frame by frame on the raw videos (what the app displays)\n")
        L.append("YOLO face detection -> crops -> HR engine updated every 0.5 s; each displayed HR vs contact-PPG HR "
                 "of the preceding 10 s. vitaleye = held-out pix_+mrc_ fold models with tracking; "
                 "legacy_physformerx = the old project monitor.\n")
        L.append("| Pipeline | subjects | MAE | RMSE | within 5 BPM | updates |")
        L.append("|---|---|---|---|---|---|")
        for k, v in pc.get("summary", {}).items():
            for sub in ("clean_gt", "all"):
                if sub in v:
                    m = v[sub]
                    L.append(f"| {k} | {'reliable GT' if sub == 'clean_gt' else 'all 50'} | {m['MAE']:.2f} | {m['RMSE']:.2f} | "
                             f"{m['within5']:.1f}% | {m['n_updates']} |")

    # ------------------------------------------------------------------ figures
    b = np.array(trk_pairs(t_sys, rel, "clean", 1))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].scatter(b[:, 1], b[:, 0], s=12, alpha=0.6)
    lo, hi = 40, max(150, b[:, :2].max() + 5)
    ax[0].plot([lo, hi], [lo, hi], "k--", lw=1)
    ax[0].set(xlabel="Contact-PPG HR (BPM)", ylabel="VitalEye HR (BPM)", title="10-s windows, unseen subjects", xlim=(lo, hi), ylim=(lo, hi))
    mean = (b[:, 0] + b[:, 1]) / 2; diff = b[:, 0] - b[:, 1]
    md, sd = diff.mean(), diff.std()
    ax[1].scatter(mean, diff, s=12, alpha=0.6)
    for y, ls in ((md, "-"), (md + 1.96 * sd, "--"), (md - 1.96 * sd, "--")):
        ax[1].axhline(y, color="k", ls=ls, lw=1)
    ax[1].set(xlabel="Mean HR (BPM)", ylabel="VitalEye - GT (BPM)", title=f"Bland-Altman: bias {md:+.2f}, LoA +-{1.96*sd:.2f}")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "cv_scatter_bland_altman.png"), dpi=130); plt.close(fig)

    series = [("POS", [metrics(cv_pairs(pix["per_video"], rel, c, "POS"))["MAE"] for c in CONDS]),
              ("VitalNet-Pix", [metrics(trk_pairs(t_pix, rel, c, 0))["MAE"] for c in CONDS]),
              ("VitalNet-MRC", [metrics(trk_pairs(t_mrc, rel, c, 0))["MAE"] for c in CONDS]),
              ("Full system", [metrics(trk_pairs(t_sys, rel, c, 1))["MAE"] for c in CONDS])]
    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.2
    for i, (n, vals) in enumerate(series):
        ax.bar(np.arange(len(CONDS)) + (i - 1.5) * w, vals, w, label=n)
    ax.set_xticks(np.arange(len(CONDS))); ax.set_xticklabels([SHORT[c] for c in CONDS])
    ax.set_ylabel("MAE (BPM), lower is better"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Robustness on unseen subjects (5-fold CV, reliable ground truth)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "robustness_mae.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    for tag, cvj, ls in (("Pix", pix, "-"), ("MRC", mrc, "--")):
        for k, h in sorted(cvj.get("history", {}).items()):
            ax.plot([e["epoch"] for e in h], [e["MAE"] for e in h], ls, marker="o", ms=3, label=f"{tag} fold {k}")
    ax.set(xlabel="epoch", ylabel="held-out MAE (BPM)", title="Held-out MAE during training (monitoring only)")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "training_curves.png"), dpi=130); plt.close(fig)

    L.append("\n## Figures\n")
    for f in ("cv_scatter_bland_altman.png", "robustness_mae.png", "training_curves.png", "audit_hr_timelines.png"):
        L.append(f"![{f}](figures/{f})")
    open(os.path.join(REP, "RESULTS.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:30]))


if __name__ == "__main__":
    main()
