import os
import sys
import json
import time
import types
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from optipulse.paths import VITALNET_DIR, OUTPUTS_DIR, METRICS_DIR, DATASET_AUDIT, YOLO_CONFIG_DIR
os.environ.setdefault("YOLO_CONFIG_DIR", YOLO_CONFIG_DIR)

from optipulse.ubfc import audit_flagged


def run_new(sids, folds):
    from optipulse import app as LM
    out = {}
    for sid in sids:
        k = next(i for i, f in enumerate(folds) if sid in f)
        a = types.SimpleNamespace(
            camera=0,
            video="",
            ubfc=sid,
            weights=[os.path.join(VITALNET_DIR, f"{p}fold{k}.pt") for p in ("pix_", "mrc_")],
            ensemble=2,
            no_track=False,
            yolo=LM.YOLO_W,
            window=10.0,
            min_window=6.0,
            update=0.5,
            snr_min=0.0,
            det_every=1,
            headless=True,
            record="",
            save_raw="",
            csv=os.path.join(OUTPUTS_DIR, "pipeline_eval", f"new_{sid}.csv"),
            cpu=False,
        )
        os.makedirs(os.path.dirname(a.csv), exist_ok=True)
        r = LM.run(a)
        out[sid] = [float(e) for e, _ in r["errors"]]
    return out


def summarize(d, excl):
    def agg(keys):
        e = np.concatenate([np.array(d[k]) for k in keys if d.get(k)])
        per = [np.mean(d[k]) for k in keys if d.get(k)]
        return dict(
            n_updates=int(len(e)),
            MAE=float(e.mean()),
            RMSE=float(np.sqrt((e ** 2).mean())),
            within5=float(np.mean(e <= 5) * 100),
            mean_subject_MAE=float(np.mean(per)),
        )
    keys = sorted(d)
    return dict(all=agg(keys), clean_gt=agg([k for k in keys if k not in excl]))


def main():
    cv = json.load(open(os.path.join(METRICS_DIR, "cv_results_pix.json")))
    folds = cv["folds"]
    excl = set(audit_flagged(DATASET_AUDIT))
    sids = sorted(s for f in folds for s in f)
    path = os.path.join(METRICS_DIR, "pipeline_comparison.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    t0 = time.time()
    res["optipulse"] = run_new(sids, folds)
    res["time_new_min"] = (time.time() - t0) / 60
    res["summary"] = {k: summarize(res[k], excl) for k in ("optipulse", "legacy_physformerx") if k in res}
    json.dump(res, open(path, "w"), indent=1)
    print(json.dumps(res["summary"], indent=1))


if __name__ == "__main__":
    main()
