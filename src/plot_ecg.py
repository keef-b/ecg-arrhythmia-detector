# Import WFDB library (used specifically for reading MIT-BIH ECG datasets)
import wfdb
from pathlib import Path

# Import matplotlib for plotting graphs (ECG visualization)
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================
# Point this to the MIT-BIH Arrhythmia Database directory
# Download from: https://physionet.org/content/mitdb/1.0.0/

DATASET_DIR = Path("../data/mit-bih")


# ============================================================
# STEP 1: LOAD THE ECG RECORD
# ============================================================
# We give WFDB the "record name", not the file extensions.
# So '100' means:
#   100.hea (header file)
#   100.dat (ECG signal data)
#   100.atr (annotations - loaded separately below)

patient_record = 121  # change to be the ID of the patient you want to inspect 

record = wfdb.rdrecord(
    str(DATASET_DIR / str(patient_record))
)

# ============================================================
# STEP 2: LOAD THE ANNOTATIONS
# ============================================================
# 'atr' = annotation file type used in MIT-BIH database
# This contains:
#   - sample locations (where each heartbeat occurs)
#   - labels (what type of beat it is)

ann = wfdb.rdann(
    str(DATASET_DIR / str(patient_record)),
    'atr'
)


# ============================================================
# STEP 3: EXTRACT ECG SIGNAL
# ============================================================
# record.p_signal is a 2D array:
#   shape = (num_samples, num_leads)
#
# MIT-BIH has 2 ECG leads, so:
#   column 0 = first lead (we usually start here)
#   column 1 = second lead

signal = record.p_signal[:, 0]


# ============================================================
# STEP 4: PLOT A SMALL SECTION OF ECG SIGNAL
# ============================================================
# We only plot the first 2000 samples so we can visually
# see a few heartbeats clearly.

plt.figure(figsize=(12, 4))
plt.plot(signal[:2000])

# Add title and labels for clarity
plt.title(f"MIT-BIH Record {patient_record} (First 2000 samples)")
plt.xlabel("Sample index (time)")
plt.ylabel("ECG amplitude (mV)")


# ============================================================
# STEP 5: ADD ANNOTATION LINES
# ============================================================
# ann.sample contains the exact index positions of heartbeats.
# Example:
#   ann.sample = [77, 370, 662, ...]
#
# Each number corresponds to a heartbeat location in the signal.

for sample in ann.sample:

    # We only plot annotations that appear inside our zoomed window
    # (first 2000 samples), otherwise plot would be too cluttered
    if sample < 2000:

        # Draw a vertical dashed line at each heartbeat location
        plt.axvline(
            x=sample,
            color='red',
            linestyle='--',
            alpha=0.6
        )

# ============================================================
# STEP 6: DISPLAY THE PLOT
# ============================================================
# This opens a window showing:
#   - ECG waveform
#   - red vertical lines = detected heartbeats

plt.show()
