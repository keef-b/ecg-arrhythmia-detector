# ============================================================
# EXPLORE ANNOTATIONS IN MIT-BIH DATABASE
# ============================================================

# WFDB is used to read MIT-BIH ECG files
import wfdb
from pathlib import Path

# Counter counts how many times each label appears
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================
# Point this to the MIT-BIH Arrhythmia Database directory
# Download from: https://physionet.org/content/mitdb/1.0.0/

DATASET_DIR = Path("../data/mit-bih")


# ============================================================
# LOAD ANNOTATIONS
# ============================================================

# Read the annotation file associated with record 100
#
# This loads information such as:
# - heartbeat locations
# - heartbeat labels
#
# Example:
# sample 77  -> N
# sample 370 -> N
# sample 662 -> A
#
ann = wfdb.rdann(
    str(DATASET_DIR / "100"),
    'atr'
)


# ============================================================
# COUNT LABELS
# ============================================================

# ann.symbol contains all annotation codes
#
# Example:
# ['+', 'N', 'N', 'N', 'A', 'N', ...]
#
# Counter tells us how many of each exist
#
counts = Counter(ann.symbol)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\nAnnotation Counts:\n")

for label, count in sorted(counts.items()):
    print(f"{label}: {count}")
