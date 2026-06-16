import wfdb
import numpy as np
from pathlib import Path
from ecg_preprocessing import bandpass_filter

# ============================================================
# DATASET CONFIGURATION
# ============================================================
# Point this to the MIT-BIH Arrhythmia Database directory
# Download from: https://physionet.org/content/mitdb/1.0.0/

DATASET_DIR = Path("../data/mit-bih")

records = [
    "100","101","102","103","104",
    "105","106","107","108","109",
    "111","112","113","114","115",
    "116","117","118","119","121",
    "122","123","124","200","201",
    "202","203","205","207","208",
    "209","210","212","213","214",
    "215","217","219","220","221",
    "222","223","228","230","231",
    "232","233","234"
]

before = 100
after = 150

X = []
y = []

NORMAL = {"N", "L", "R"}

for record_name in records:

    print("Processing", record_name)

    record = wfdb.rdrecord(f"{DATASET_DIR}/{record_name}")
    ann = wfdb.rdann(f"{DATASET_DIR}/{record_name}", "atr")

    # STEP 1: CLEAN SIGNAL (BANDPASS FILTER)
    signal = bandpass_filter(record.p_signal[:, 0])

    # STEP 2: LOOP THROUGH BEATS
    for sample, label in zip(ann.sample, ann.symbol):

        start = sample - before
        end = sample + after

        if start < 0 or end >= len(signal):
            continue

        beat = signal[start:end]

        if len(beat) != before + after:
            continue

        X.append(beat)

        # binary label
        y.append(0 if label in NORMAL else 1)

X = np.array(X)
y = np.array(y)

print("Final dataset:", X.shape, y.shape)

np.save("../data/X.npy", X)
np.save("../data/y.npy", y)
