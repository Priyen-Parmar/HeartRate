"""OptiPulse - contactless heart-rate estimation from a camera (remote photoplethysmography).

Package layout
--------------
dsp.py      signal processing shared by training, evaluation and the live app
ubfc.py     UBFC-rPPG raw-data readers (video + contact-PPG ground truth)
face.py     YOLOv8 face detector + temporally smoothed, enlarged face crop
model.py    VitalNet: 3D-CNN rPPG network with soft skin attention
augment.py  GPU augmentations (low light, sensor noise, motion, HR resampling, fps drop)
"""

TARGET_FPS = 30.0      # every clip (training and live) is resampled to a uniform 30 Hz timeline
CROP_SIZE = 80         # stored crop resolution (1.6x enlarged face box)
INPUT_SIZE = 72        # network input resolution
BOX_SCALE = 1.6        # face box enlargement factor
HR_LO_HZ = 0.65        # 39 BPM  -- HR search band, identical for labels and predictions
HR_HI_HZ = 3.2         # 192 BPM
