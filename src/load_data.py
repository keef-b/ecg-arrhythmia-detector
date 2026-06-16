import wfdb
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================
# Point this to the MIT-BIH Arrhythmia Database directory
# Download from: https://physionet.org/content/mitdb/1.0.0/

DATASET_DIR = Path("../data/mit-bih")

record = wfdb.rdrecord(str(DATASET_DIR / "100"))
ann = wfdb.rdann(str(DATASET_DIR / "100"), 'atr')

print(record.p_signal.shape)
print(ann.symbol[:20])
