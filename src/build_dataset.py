from pathlib import Path

import wfdb
import numpy as np

from ecg_preprocessing import bandpass_filter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "mit-bih-arrhythmia-database-1.0.0"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

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

    record_path = str(RAW_DATA_DIR / record_name)
    record = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, "atr")

    # STEP 1: CLEAN SIGNAL (IMPORTANT CHANGE)
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

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
np.save(PROCESSED_DIR / "X.npy", X)
np.save(PROCESSED_DIR / "y.npy", y)

