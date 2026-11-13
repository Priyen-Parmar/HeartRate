"""End-to-end comparison of the FULL live pipelines on the UBFC videos (frame by frame, exactly as
they run on a webcam): new VitalEye v2 vs the legacy PhysFormerX monitor.

* VitalEye: every subject is scored with the two fold models (pix_ + mrc_) that NEVER saw that
            subject, with causal tracking - i.e. the deployed system minus the other folds (honest).
* legacy    : artifacts/legacy_code/live_hr_monitor.py with its own models (it was trained on an
              unknown split of these same videos, so its numbers are, if anything, optimistic).
Each displayed HR is compared with the contact-PPG HR over the preceding 10 s.
Output: reports/pipeline_comparison.json
"""

import os
import sys
import json
import time
import argparse
import types
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(ROOT, ".cache", "ultralytics"))

from vitaleye.ubfc import list_videos, video_info, load_ground_truth
from vitaleye.dsp import hr_from_signal


def run_new(sids, folds):
    import live_monitor as LM
    out = {}
    for sid in sids:
        k = next(i for i, f in enumerate(folds) if sid in f)
        a = types.SimpleNamespace(camera=0, video="", ubfc=sid, weights=[os.path.join(ROOT, "models", "vitalnet", f"{p}fold{k}.pt") for p in ("pix_", "mrc_")],
                                  ensemble=2, no_track=False, yolo=LM.YOLO_W, window=10.0, min_window=6.0, update=0.5, snr_min=0.0,
                                  det_every=1, headless=True, record="", save_raw="", csv=os.path.join(ROOT, "outputs", "pipeline_eval", f"new_{sid}.csv"),
                                  cpu=False)
        os.makedirs(os.path.dirname(a.csv), exist_ok=True)
        r = LM.run(a)
        out[sid] = [float(e) for e, _ in r["errors"]]
    return out


def run_legacy(sids, raw_dir):
    legacy_dir = os.path.join(ROOT, "artifacts", "legacy_code")
    sys.path.insert(0, legacy_dir)
    import live_hr_monitor as L
    vids = {s: (v, g) for s, v, g in list_videos(raw_dir)}
    out = {}
    for sid in sids:
        v, g = vids[sid]
        info = video_info(v)
        gt = load_ground_truth(g, info["n_frames"], info["fps"])
        rec = []
        orig = L.draw_vitals_panel

        def recorder(frame, bpm, *a, **kw):
            rec.append(bpm)
            return orig(frame, bpm, *a, **kw)
        L.draw_vitals_panel = recorder
        args = types.SimpleNamespace(yolo_weights=L.DEFAULT_YOLO, model_weights=L.DEFAULT_MODEL, camera=0, video=v,
                                     record="", fps=30, yolo_interval=2, slide_step=15, headless=True, self_test=False)
        L.run(args)
        L.draw_vitals_panel = orig
        fps = info["fps"]
        errs = []
        for i in range(int(10 * fps), len(rec), 15):
            if rec[i] > 0:
                gh = hr_from_signal(gt["ppg"][max(0, i + 1 - int(10 * fps)):i + 1], fps)[0]
                errs.append(abs(rec[i] - gh))
        out[sid] = errs
        print(f"[legacy] {sid}: MAE={np.mean(errs):.2f} (n={len(errs)})", flush=True)
    return out


def summarize(d, excl):
    def agg(keys):
        e = np.concatenate([np.array(d[k]) for k in keys if d.get(k)])
        per = [np.mean(d[k]) for k in keys if d.get(k)]
        return dict(n_updates=int(len(e)), MAE=float(e.mean()), RMSE=float(np.sqrt((e ** 2).mean())),
                    within5=float(np.mean(e <= 5) * 100), mean_subject_MAE=float(np.mean(per)))
    keys = sorted(d)
    return dict(all=agg(keys), clean_gt=agg([k for k in keys if k not in excl]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="both", choices=["new", "legacy", "both"])
    a = ap.parse_args()
    from vitaleye.ubfc import audit_flagged
    cv = json.load(open(os.path.join(ROOT, "reports", "cv_results_pix.json")))
    folds = cv["folds"]
    excl = set(audit_flagged(os.path.join(ROOT, "reports", "dataset_audit.json")))
    sids = sorted(s for f in folds for s in f)
    path = os.path.join(ROOT, "reports", "pipeline_comparison.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    if a.which in ("new", "both"):
        t0 = time.time(); res["vitaleye_v2"] = run_new(sids, folds); res["time_new_min"] = (time.time() - t0) / 60
        json.dump(res, open(path, "w"), indent=1)
    if a.which in ("legacy", "both"):
        t0 = time.time(); res["legacy_physformerx"] = run_legacy(sids, os.path.join(ROOT, "data", "raw", "UBFC-rPPG"))
        res["time_legacy_min"] = (time.time() - t0) / 60
    res["summary"] = {k: summarize(res[k], excl) for k in ("vitaleye_v2", "legacy_physformerx") if k in res}
    json.dump(res, open(path, "w"), indent=1)
    print(json.dumps(res["summary"], indent=1))


if __name__ == "__main__":
    main()

