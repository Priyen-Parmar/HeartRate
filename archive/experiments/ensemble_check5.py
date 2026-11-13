"""Honest test of the DEPLOYED 5-model ensemble: subjects that were excluded from training in every
run-1 fold are unseen by all 5 models. Compares each single fold model with the 5-model ensemble
(wave / psd / median fusion). Output: results/metrics/ensemble_check5.json"""

import os
import sys
import glob
import json
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from optipulse.paths import TRIAL_MODELS_DIR, UBFC_PROCESSED_DIR, METRICS_DIR
from optipulse.live import load_ensemble
from optipulse.data import load_videos
from optipulse.evaluation import evaluate_video, metrics

CONDS = ("clean", "lowlight", "motion", "lowlight+motion", "lowfps")
MODEL_DIR = os.path.join(TRIAL_MODELS_DIR, "vitalnet_pix")
VALID_PPG = ["sub11", "sub18", "sub20", "sub24"]   # oximeter-HR dropouts, PPG verified
NOISY_PPG = ["sub3", "sub4", "sub14"]             # noisy contact PPG


def main():
    dev = torch.device("cuda")
    paths = sorted(glob.glob(os.path.join(MODEL_DIR, "fold*.pt")))
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        assert not set(VALID_PPG + NOISY_PPG) & set(ck["train_subjects"]), p
    models = [load_ensemble(None, dev, weights=[p])[0][0] for p in paths]
    out = {}
    for subset_name, subset in (("verified_ppg", VALID_PPG), ("noisy_ppg", NOISY_PPG)):
        vids = load_videos(UBFC_PROCESSED_DIR, subset)
        configs = {f"fold{i}": ([m], "wave") for i, m in enumerate(models)}
        configs.update({"ens5-wave": (models, "wave"), "ens5-psd": (models, "psd"), "ens5-median": (models, "median")})
        out[subset_name] = {}
        for name, (ms, comb) in configs.items():
            pairs = {c: [] for c in CONDS}
            for v in vids:
                r = evaluate_video(ms, v, dev, CONDS, baselines=False, seed=sum(map(ord, v.sid)), combine=comb)
                for c in CONDS:
                    pairs[c] += r[c]["vitalnet"]
            out[subset_name][name] = {c: metrics(pairs[c]) for c in CONDS}
            print(f"[{subset_name}] {name:12s} " + "  ".join(f"{c}={out[subset_name][name][c]['MAE']:.2f}" for c in CONDS), flush=True)
        singles = [k for k in out[subset_name] if k.startswith("fold")]
        mean_single = {c: float(np.mean([out[subset_name][k][c]["MAE"] for k in singles])) for c in CONDS}
        out[subset_name]["mean_of_single_models"] = mean_single
        print(f"[{subset_name}] mean single   " + "  ".join(f"{c}={mean_single[c]:.2f}" for c in CONDS))
    json.dump(out, open(os.path.join(METRICS_DIR, "ensemble_check5.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
