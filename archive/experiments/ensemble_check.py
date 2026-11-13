"""Honest measurement of the ensemble effect: three independently trained models that all held out
the SAME fold-0 subjects (run 1, run 2-partial and v3 dev runs) are evaluated alone and averaged
on those unseen subjects. Output: results/metrics/ensemble_check.json"""

import os
import sys
import json
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from optipulse.paths import TRIAL_MODELS_DIR, DATASET_AUDIT, UBFC_PROCESSED_DIR, METRICS_DIR
from optipulse.live import load_ensemble
from optipulse.data import load_videos
from optipulse.evaluation import evaluate_video, metrics
from optipulse.ubfc import audit_flagged

CONDS = ("clean", "lowlight", "motion", "lowlight+motion", "lowfps")
CKPTS = {
    "run1": os.path.join(TRIAL_MODELS_DIR, "vitalnet_pix", "fold0.pt"),
    "run2": os.path.join(TRIAL_MODELS_DIR, "vitalnet_run2_partial", "fold0.pt"),
    "v3": os.path.join(TRIAL_MODELS_DIR, "vitalnet_v3_dev", "fold0.pt"),
}


def main():
    dev = torch.device("cuda")
    ck = torch.load(CKPTS["run1"], map_location="cpu", weights_only=False)
    flagged = set(audit_flagged(DATASET_AUDIT))
    test = [s for s in ck["test_subjects"] if s not in flagged]
    vids = load_videos(UBFC_PROCESSED_DIR, test)
    models = {k: load_ensemble(None, dev, weights=[p])[0][0] for k, p in CKPTS.items()}
    configs = {"run1": (["run1"], "wave"), "run2": (["run2"], "wave"), "v3": (["v3"], "wave"),
               "ens3-wave": (["run1", "run2", "v3"], "wave"), "ens3-psd": (["run1", "run2", "v3"], "psd"),
               "ens3-median": (["run1", "run2", "v3"], "median")}
    out = {}
    for name, (keys, comb) in configs.items():
        pairs = {c: [] for c in CONDS}
        for v in vids:
            r = evaluate_video([models[k] for k in keys], v, dev, CONDS, baselines=False, seed=sum(map(ord, v.sid)), combine=comb)
            for c in CONDS:
                pairs[c] += r[c]["vitalnet"]
        out[name] = {c: metrics(pairs[c]) for c in CONDS}
        print(f"{name:12s} " + "  ".join(f"{c}={out[name][c]['MAE']:.2f}" for c in CONDS), flush=True)
    json.dump(dict(subjects=test, results=out), open(os.path.join(METRICS_DIR, "ensemble_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
