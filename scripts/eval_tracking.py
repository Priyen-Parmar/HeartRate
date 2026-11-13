"""Cross-validated evaluation of a set of fold models WITH and WITHOUT causal HR tracking.

Every subject is evaluated only by the fold model that never saw it (checkpoint 'test_subjects').
Sliding 10-s windows with 1-s hop are inferred; the plain estimate of window [s, s+10 s) is its own
spectral peak, the tracked estimate is the causal HMM output after that same window (uses only
windows ending <= s+10 s). Scored on the same non-overlapping 10-s windows as train_cv.py.

usage: python scripts/eval_tracking.py --models artifacts/trial_models/vitalnet_pix --tag pix
       --models dirA,dirB  -> per subject, ensemble of every checkpoint (from all dirs) that held it out;
                              only subjects held out in EVERY dir are scored (fair comparison)
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vitaleye import TARGET_FPS
from vitaleye.live import load_ensemble
from vitaleye.data import load_videos
from vitaleye.evaluation import stress_transform, metrics, WIN
from vitaleye.dsp import hr_from_signal, hr_from_waves
from vitaleye.tracking import HRTracker, band_psd, BPM_GRID
from vitaleye.ubfc import audit_flagged

CONDS = ("clean", "lowlight", "motion", "lowlight+motion", "lowfps")
HOP = 30


@torch.no_grad()
def sliding_waves(model, x72, bs=16):
    T = x72.shape[2]
    starts = list(range(0, T - WIN + 1, HOP))
    out = []
    for i in range(0, len(starts), bs):
        clip = torch.cat([x72[:, :, s:s + WIN] for s in starts[i:i + bs]], 0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out += list(model(clip).float().cpu().numpy())
    return starts, out


def ckpts(entry):
    """'dir' -> dir/*fold*.pt ; a glob such as 'models/vitalnet/mrc_*.pt' is used as given."""
    pat = entry if ("*" in entry or entry.endswith(".pt")) else os.path.join(entry, "*fold*.pt")
    return sorted(glob.glob(os.path.join(ROOT, pat)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--never_trained", default="",
                    help="comma list of subjects excluded from training in EVERY checkpoint: they are scored by the "
                         "FULL deployed ensemble (all checkpoints of all dirs) instead of the held-out fold models")
    args = ap.parse_args()
    conds = args.conds.split(",")
    dev = torch.device("cuda")
    dirs = args.models.split(",")
    by_sub = {}
    for d in dirs:
        for p in ckpts(d):
            ck = torch.load(p, map_location="cpu", weights_only=False)
            for sid in ck["test_subjects"]:
                by_sub.setdefault(sid, {})[d] = p
    flagged = set(audit_flagged(os.path.join(ROOT, "reports", "dataset_audit.json")))
    res = {c: {"plain": [], "tracked": []} for c in conds}
    per_sub = {}
    groups = {}
    nt = [x for x in args.never_trained.split(",") if x]
    if nt:
        allp = [p for d in dirs for p in ckpts(d)]
        for p in allp:
            assert not set(nt) & set(torch.load(p, map_location="cpu", weights_only=False)["train_subjects"]), p
        groups[tuple(allp)] = nt
        flagged = set()                     # score them explicitly
    else:
        for sid, m in by_sub.items():
            if len(m) == len(dirs):
                groups.setdefault(tuple(m[d] for d in dirs), []).append(sid)
    for paths, sids in sorted(groups.items()):
        models = load_ensemble(None, dev, weights=list(paths))[0]
        for v in load_videos(os.path.join(ROOT, "data", "processed", "ubfc_v2"), sorted(sids)):
            x80 = torch.from_numpy(v.frames).to(dev).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
            per_sub[v.sid] = {}
            for ci, c in enumerate(CONDS):
                if c not in conds:
                    continue
                x72 = stress_transform(x80, c, seed=sum(map(ord, v.sid)) * 100 + ci)
                outs = [sliding_waves(m, x72) for m in models]
                starts = outs[0][0]
                waves = list(zip(*[o[1] for o in outs]))          # per window: tuple of model waves
                trk = HRTracker(args.sigma, args.alpha)
                tracked = {}
                for s, w in zip(starts, waves):
                    tracked[s] = trk.update(band_psd(list(w), TARGET_FPS), (s + WIN) / TARGET_FPS)[0]
                rows = []
                for s, w in zip(starts, waves):
                    if s % WIN:
                        continue
                    gt = hr_from_signal(v.ppg_raw[s:s + WIN], TARGET_FPS)[0]
                    h, snr, _ = hr_from_waves(list(w), TARGET_FPS, "psd")
                    rows.append((h, tracked[s], gt, snr))
                per_sub[v.sid][c] = rows
                if v.sid not in flagged:
                    res[c]["plain"] += [(r[0], r[2], r[3]) for r in rows]
                    res[c]["tracked"] += [(r[1], r[2], r[3]) for r in rows]
            print(f"{v.sid:20s} " + "  ".join(
                f"{c}: {np.mean([abs(r[0]-r[2]) for r in per_sub[v.sid][c]]):5.1f}->{np.mean([abs(r[1]-r[2]) for r in per_sub[v.sid][c]]):5.1f}"
                for c in conds), flush=True)
        del models
    summary = {c: {k: metrics(res[c][k]) for k in res[c]} for c in conds}
    print(f"\n== {args.tag}: reliable-GT subjects (CV, unseen), MAE plain -> tracked")
    for c in conds:
        print(f"{c:16s} {summary[c]['plain']['MAE']:6.2f} -> {summary[c]['tracked']['MAE']:6.2f}   "
              f"within5 {summary[c]['plain']['within5']:5.1f}% -> {summary[c]['tracked']['within5']:5.1f}%")
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    json.dump(dict(args=vars(args), summary=summary, per_subject=per_sub),
              open(os.path.join(ROOT, "reports", f"tracking_{args.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
