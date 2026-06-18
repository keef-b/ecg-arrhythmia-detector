from pathlib import Path

import wfdb
import numpy as np

# ============================================================
# DATASET LOCATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "mit-bih-arrhythmia-database-1.0.0"

# ============================================================
# RECORDS IN MIT-BIH ARRHYTHMIA DATABASE
# ============================================================

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

# ============================================================
# WINDOW SIZE AROUND EACH HEARTBEAT
# ============================================================

before = 100
after = 150

# Total samples per heartbeat
window_size = before + after

# ============================================================
# STORAGE FOR MACHINE LEARNING DATASET
# ============================================================

# X will contain heartbeat waveforms
X = []

# y will contain labels
y = []

# ============================================================
# LOOP THROUGH EVERY ECG RECORD
# ============================================================

for record_name in records:

    print(f"\nLoading record {record_name}")

    # --------------------------------------------------------
    # LOAD ECG SIGNAL
    # --------------------------------------------------------

    record = wfdb.rdrecord(str(DATASET_DIR / record_name))

    # --------------------------------------------------------
    # LOAD ANNOTATIONS
    # --------------------------------------------------------

    ann = wfdb.rdann(str(DATASET_DIR / record_name), "atr")

    # --------------------------------------------------------
    # USE LEAD 1
    # --------------------------------------------------------

    signal = record.p_signal[:, 0]

    print(f"Annotations found: {len(ann.sample)}")

    # --------------------------------------------------------
    # LOOP THROUGH EVERY ANNOTATION
    # --------------------------------------------------------

    for sample, label in zip(
        ann.sample,
        ann.symbol
    ):

        # ----------------------------------------------------
        # KEEP ONLY BEAT ANNOTATIONS
        # ----------------------------------------------------
        #
        # MIT-BIH contains many annotation types:
        #
        # +  paced beat
        # |  rhythm change
        # ~  noise
        # etc.
        #
        # We only want actual heartbeat labels.
        #

        beat_labels = [
            "N",   # Normal
            "L",   # Left bundle branch block
            "R",   # Right bundle branch block
            "A",   # Atrial premature beat
            "a",   # Aberrated atrial premature beat
            "J",   # Nodal premature beat
            "V",   # Ventricular premature beat
            "E",   # Ventricular escape beat
            "F",   # Fusion beat
            "/"    # Paced beat
        ]

        if label not in beat_labels:
            continue

        # ----------------------------------------------------
        # DEFINE WINDOW AROUND HEARTBEAT
        # ----------------------------------------------------

        start = sample - before
        end = sample + after

        # ----------------------------------------------------
        # SKIP HEARTBEATS TOO CLOSE TO EDGES
        # ----------------------------------------------------

        if start < 0:
            continue

        if end >= len(signal):
            continue

        # ----------------------------------------------------
        # EXTRACT HEARTBEAT
        # ----------------------------------------------------

        beat = signal[start:end]

        # ----------------------------------------------------
        # VERIFY WINDOW LENGTH
        # ----------------------------------------------------

        if len(beat) != window_size:
            continue

        # ----------------------------------------------------
        # STORE FOR TRAINING
        # ----------------------------------------------------

        X.append(beat)
        y.append(label)

# ============================================================
# CONVERT TO NUMPY ARRAYS
# ============================================================

X = np.array(X)

y = np.array(y)

# ============================================================
# DATASET SUMMARY
# ============================================================

print("\n========================")
print("DATASET COMPLETE")
print("========================")

print("Number of beats:", len(X))
print("X shape:", X.shape)
print("y shape:", y.shape)

# ============================================================
# SAVE DATASET
# ============================================================

np.save("X_beats.npy", X)
np.save("y_labels.npy", y)

print("\nSaved:")
print("X_beats.npy")
print("y_labels.npy")

'''
| Symbol | Meaning                               |                  |
| ------ | ------------------------------------- | ---------------- |
| N      | Normal beat                           |                  |
| L      | Left bundle branch block beat         |                  |
| R      | Right bundle branch block beat        |                  |
| A      | Atrial premature beat                 |                  |
| a      | Aberrated atrial premature beat       |                  |
| J      | Nodal (junctional) premature beat     |                  |
| S      | Supraventricular premature beat       |                  |
| V      | Premature ventricular contraction     |                  |
| F      | Fusion of ventricular and normal beat |                  |
| /      | Paced beat                            |                  |
| Q      | Unclassifiable beat                   | ([PhysioNet][1]) |

[1]: https://physionet.org/physiotools/wpg/wpg_36.htm?utm_source=chatgpt.com "WFDB Programmer's Guide: 4. Annotation Codes"


'''