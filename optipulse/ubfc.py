"""Readers for the UBFC-rPPG raw data (Dataset1 + Dataset2, flat folder layout)."""

import os
import re
import glob
import numpy as np
import cv2


def subject_id(path: str) -> str:
    name = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"vid_2_sub(\d+)", name)
    if m:
        return f"sub{int(m.group(1))}"
    m = re.search(r"vid_1_(\w+)", name)
    return f"ds1_{m.group(1)}" if m else name


def audit_flagged(audit_json: str):
    """Subjects whose ground truth fails the audit rule (oximeter-vs-PPG disagreement >10 BPM in
    >30% of 10-s windows, or median contact-PPG SNR < 2 dB). Used as the fixed 'unreliable GT'
    evaluation subset for every method and every run."""
    import json
    a = json.load(open(audit_json))
    return sorted(s["sid"] for s in a["subjects"]
                  if s["frac_win_disagree10"] > 0.30 or s["gt_snr_db_median"] < 2.0)


def list_videos(raw_dir: str):
    vids = sorted(glob.glob(os.path.join(raw_dir, "vid_*.avi")))
    out = []
    for v in vids:
        base = os.path.basename(v)[len("vid_"):-len(".avi")]
        gt = os.path.join(raw_dir, f"ground_truth_{base}.txt")
        if os.path.exists(gt):
            out.append((subject_id(v), v, gt))
    return out


def video_info(path: str):
    cap = cv2.VideoCapture(path)
    info = dict(
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return info


def read_frames(path: str):
    """Yield BGR frames."""
    cap = cv2.VideoCapture(path)
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        yield fr
    cap.release()


def load_ground_truth(gt_path: str, n_frames: int, video_fps: float):
    """Return dict with
        frame_t  : [n_frames] timestamp (s) of every video frame
        ppg      : [n_frames] contact PPG sampled at the frame timestamps
        sensor_hr: [n_frames] oximeter HR at the frame timestamps (NOT used as label)
    Dataset2 files hold one PPG sample per frame (+ timestamp).  Dataset1 files hold a
    ~62.5 Hz stream (ms, HR, SpO2, PPG); it is interpolated onto frame times i/fps.
    """
    with open(gt_path) as f:
        first = f.readline()
    if "," in first:  # Dataset1
        d = np.loadtxt(gt_path, delimiter=",")
        t_gt = (d[:, 0] - d[0, 0]) / 1000.0
        frame_t = np.arange(n_frames) / video_fps
        ppg = np.interp(frame_t, t_gt, d[:, 3])
        hr = np.interp(frame_t, t_gt, d[:, 1])
        gt_duration = float(t_gt[-1])
        kind = "ds1"
    else:  # Dataset2
        d = np.loadtxt(gt_path)
        ppg, hr, t = d[0], d[1], d[2]
        n = min(len(ppg), n_frames)
        ppg, hr, t = ppg[:n], hr[:n], t[:n] - t[0]
        # The logged timestamps jitter in a 11 ms / 44 ms pattern (logging-timer quantisation),
        # while the camera itself captures at a constant rate.  Use a least-squares linear fit
        # of the timestamps: true average frame rate, no artificial jitter.
        idx = np.arange(n)
        slope, icpt = np.polyfit(idx, t, 1)
        frame_t = idx * slope
        gt_duration = float(t[-1])
        kind = "ds2"
    return dict(frame_t=frame_t, ppg=ppg.astype(np.float64), sensor_hr=hr.astype(np.float64),
                gt_duration=gt_duration, kind=kind)
