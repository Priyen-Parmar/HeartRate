"""Subject-wise 5-fold cross-validation training of VitalNet on UBFC-rPPG (RTX GPU).

* every subject is in exactly one test fold -> every reported number is on UNSEEN people
* no early stopping / checkpoint picking on the test fold (fixed schedule, last epoch)
* subjects whose ground truth fails the audit are excluded from TRAINING (still reported)
* the fold models form the deployed ensemble (models/vitalnet/*.pt; final system = 5 x v1 'pix_' +
  5 x v4 'mrc_' models, see docs/METHOD.md)
"""

import os
import sys
import json
import time
import math
import random
import argparse
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from optipulse.paths import UBFC_PROCESSED_DIR, VITALNET_DIR, METRICS_DIR

from optipulse.model import VitalNet, neg_pearson, SpectralKL, count_params
from optipulse.data import load_videos, BatchBuilder
from optipulse.evaluation import evaluate_video, metrics

CONDITIONS = ("clean", "lowlight", "motion", "lowlight+motion", "lowfps")


def make_folds(sids, k, seed):
    rng = random.Random(seed)
    ds1 = sorted([s for s in sids if s.startswith("ds1_")]); ds2 = sorted([s for s in sids if not s.startswith("ds1_")])
    rng.shuffle(ds1); rng.shuffle(ds2)
    folds = [[] for _ in range(k)]
    for i, s in enumerate(ds1 + ds2):
        folds[i % k].append(s)
    return folds


def train_fold(args, fold, train_sids, test_sids, device, log):
    torch.manual_seed(args.seed + fold); np.random.seed(args.seed + fold); random.seed(args.seed + fold)
    tr_videos = load_videos(args.proc_dir, train_sids)
    te_videos = load_videos(args.proc_dir, test_sids)
    bb = BatchBuilder(tr_videos, T=args.clip_len, device=device)
    model = VitalNet(c=args.width, drop=args.dropout, version=args.version).to(device)
    spec = SpectralKL().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total = args.epochs * args.steps
    warm = args.steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, i / total))))
    log(f"[fold {fold}] train={len(train_sids)} subjects, test={test_sids}, params={count_params(model)/1e6:.3f}M")
    hist = []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time(); lp = lk = 0.0
        for it in range(args.steps):
            x, y = bb.sample(args.batch, paired=args.pair_w > 0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(x)
            pred = pred.float()
            l_p = neg_pearson(pred, y)
            l_k = spec(pred, y)
            loss = l_p + args.kl_w * l_k
            if args.pair_w > 0:
                # cross-illumination consistency: the low-light view must reproduce the waveform the
                # model itself extracts from the well-lit view of the same clip (teacher = stop-grad).
                # Unlike the contact-PPG label, this target shares the facial pulse's phase/shape.
                h = pred.shape[0] // 2
                loss = loss + args.pair_w * neg_pearson(pred[h:], pred[:h].detach())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            lp += l_p.item(); lk += l_k.item()
        msg = f"[fold {fold}] ep {ep:3d}  pearson={lp/args.steps:.4f}  specKL={lk/args.steps:.4f}  lr={sched.get_last_lr()[0]:.2e}  {time.time()-t0:.0f}s"
        if ep % args.eval_every == 0 or ep == args.epochs:
            model.eval()
            pairs = []
            for v in te_videos:
                pairs += evaluate_video([model], v, device, ("clean",), baselines=False)["clean"]["vitalnet"]
            m = metrics(pairs)
            msg += f"  | test-fold(monitor only) MAE={m['MAE']:.2f} RMSE={m['RMSE']:.2f}"
            hist.append(dict(epoch=ep, **m))
        log(msg)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{args.prefix}fold{fold}.pt")
    torch.save(dict(arch="VitalNet", width=args.width, dropout=args.dropout, version=args.version, state_dict=model.state_dict(),
                    fold=fold, train_subjects=train_sids, test_subjects=test_sids, args=vars(args)), path)
    # full evaluation with stress tests + baselines
    model.eval()
    per_video = {}
    for v in te_videos:
        per_video[v.sid] = evaluate_video([model], v, device, CONDITIONS, baselines=True, seed=sum(map(ord, v.sid)))
    return per_video, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc_dir", default=UBFC_PROCESSED_DIR)
    ap.add_argument("--out_dir", default=VITALNET_DIR)
    ap.add_argument("--report", default=os.path.join(METRICS_DIR, "cv_results.json"))
    ap.add_argument("--exclude_train", default="", help="comma separated subjects excluded from training")
    ap.add_argument("--prefix", default="", help="checkpoint file prefix, e.g. pix_ -> pix_fold0.pt")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only_fold", type=int, default=-1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--clip_len", type=int, default=180)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--kl_w", type=float, default=0.5)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--version", type=int, default=4,
                    help="VitalNet version (1 = run 1, 3 = temporal pre-filter, 4 = + region branch / noise gate)")
    ap.add_argument("--pair_w", type=float, default=0.0, help="weight of the paired low-light consistency loss (0 = off)")
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=776)
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    log_path = args.report.replace(".json", ".log")
    logf = open(log_path, "a", encoding="utf-8")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    sids = sorted(os.path.splitext(f)[0] for f in os.listdir(args.proc_dir) if f.endswith(".npz"))
    excl = set(s for s in args.exclude_train.split(",") if s)
    folds = make_folds(sids, args.folds, args.seed)
    log(f"=== VitalNet CV  {time.strftime('%Y-%m-%d %H:%M')}  args={vars(args)}")
    log(f"excluded from training (ground-truth audit): {sorted(excl)}")
    results = dict(args=vars(args), folds=folds, excluded=sorted(excl), per_video={}, history={})
    if os.path.exists(args.report) and args.only_fold >= 0:
        results = json.load(open(args.report))
    for k in range(args.folds):
        if args.only_fold >= 0 and k != args.only_fold:
            continue
        test = folds[k]
        train = [s for s in sids if s not in test and s not in excl]
        t0 = time.time()
        per_video, hist = train_fold(args, k, train, test, device, log)
        results["per_video"].update(per_video)
        results["history"][str(k)] = hist
        log(f"[fold {k}] done in {(time.time()-t0)/60:.1f} min; peak GPU mem {torch.cuda.max_memory_allocated()/2**30:.2f} GB")
        json.dump(results, open(args.report, "w"), indent=1)

    # aggregate
    summary = {}
    for subset_name, subset in (("all_subjects", sids), ("clean_gt_subjects", [s for s in sids if s not in excl])):
        summary[subset_name] = {}
        for cond in CONDITIONS:
            summary[subset_name][cond] = {}
            for meth in ("vitalnet", "POS", "CHROM", "GREEN"):
                pairs = [p for s in subset if s in results["per_video"] for p in results["per_video"][s][cond][meth]]
                summary[subset_name][cond][meth] = metrics(pairs)
    results["summary"] = summary
    json.dump(results, open(args.report, "w"), indent=1)
    for cond in CONDITIONS:
        row = summary["clean_gt_subjects"][cond]
        log(f"{cond:16s} " + "  ".join(f"{m}: MAE={row[m].get('MAE', float('nan')):.2f}" for m in row))
    log(f"peak GPU memory: {torch.cuda.max_memory_allocated()/2**30:.2f} GB")


if __name__ == "__main__":
    main()
