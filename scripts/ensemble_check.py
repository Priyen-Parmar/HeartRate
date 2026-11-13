"""Honest measurement of the ensemble effect: three independently trained models that all held out
the SAME fold-0 subjects (run 1, run 2-partial and v3 dev runs) are evaluated alone and averaged
on those unseen subjects. Output: reports/ensemble_check.json"""

import os
import sys
import json
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vitaleye.live import load_ensemble
from vitaleye.data import load_videos
from vitaleye.evaluation import evaluate_video, metrics
from vitaleye.ubfc import audit_flagged

CONDS = ("clean", "lowlight", "motion", "lowlight+motion", "lowfps")
CKPTS = {
    "run1": os.path.join(ROOT, "artifacts", "trial_models", "vitalnet_pix", "fold0.pt"),
    "run2": os.path.join(ROOT, "artifacts", "trial_models", "vitalnet_run2_partial", "fold0.pt"),
    "v3": os.path.join(ROOT, "artifacts", "trial_models", "vitalnet_v3_dev", "fold0.pt"),
}


def main():
    dev = torch.device("cuda")
    ck = torch.load(CKPTS["run1"], map_location="cpu", weights_only=False)
    flagged = set(audit_flagged(os.path.join(ROOT, "reports", "dataset_audit.json")))
    test = [s for s in ck["test_subjects"] if s not in flagged]
    vids = load_videos(os.path.join(ROOT, "data", "processed", "ubfc_v2"), test)
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
    json.dump(dict(subjects=test, results=out), open(os.path.join(ROOT, "reports", "ensemble_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
