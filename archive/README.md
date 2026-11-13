# archive/: history, not used by the current system

Nothing in this folder is imported or needed by `optipulse/`, the launchers or the deployed models. It is
kept so that earlier results can be reproduced and compared.

```text
archive/
├── legacy_system/          the PREVIOUS version of the project (before VitalNet)
│   ├── live_hr_monitor.py  old live monitor (PhysFormerX / VitalEyeProNet + YOLO); still runnable, and
│   │                       pipeline/compare_pipelines.py runs it to get the "old system: 6.81 BPM" baseline
│   ├── PhysFormerX.py, Vidformer1.py, vitaleye_pro_model.py, train_vitaleye_pro.py, preprocessor.py, ...
│   ├── older_versions/     even older variants of those scripts + a notebook
│   ├── deploy/             old Gradio / Docker web demo (Hugging Face Space) and the modules the old monitor imports;
│   │   └── models/         face_landmark.tflite (versioned) + vidformer.pt, yolov8/ (local only, git-ignored *.pt)
│   └── docs/               documentation, figures and screenshots of the OLD system, and a reference paper
│                           (Chen et al. 2019, respiratory rate from face videos)
├── experiments/            one-off development experiments of the current system
│   └── ensemble_check*.py  ensemble combinations on fold 0 -> results/metrics/ensemble_check*.json
└── trial_models/           (local only, git-ignored) checkpoints of every VitalNet experiment
                            (run 1, run 2, v3, v4, ...); needed only by experiments/
```

Why the old system was replaced (details in `docs/WORK_REPORT.md`):
* it assumed the camera's frame rate instead of using real timestamps;
* preprocessing differed between training and live use;
* a hard YCbCr skin mask flickered from frame to frame;
* it was biased towards ~70–100 BPM;
* it was a video transformer trained on very little data.

Result: 6.81 BPM end-to-end on the UBFC videos vs 1.59 BPM for the current system.

The old code contains hard-coded paths of the machine it was written on (e.g. `E:/...`). They were left
as they were. Running the old monitor needs its own extra packages (e.g. `tqdm`, see `deploy/requirements.txt`).

> The documents in `legacy_system/docs/` describe the old design. For the current system read
> `docs/EXPLAIN.md` and `docs/METHOD.md`.
