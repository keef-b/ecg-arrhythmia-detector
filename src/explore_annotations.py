from pathlib import Path

# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

# WFDB is used to read MIT-BIH ECG files
import wfdb

# Counter counts how many times each label appears
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "mit-bih-arrhythmia-database-1.0.0"


# ------------------------------------------------------------
# LOAD ANNOTATIONS
# ------------------------------------------------------------

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
    str(RAW_DATA_DIR / '100'),
    'atr'
)


# ------------------------------------------------------------
# COUNT LABELS
# ------------------------------------------------------------

# ann.symbol contains all annotation codes
#
# Example:
# ['+', 'N', 'N', 'N', 'A', 'N', ...]
#
# Counter tells us how many of each exist
#
counts = Counter(ann.symbol)


# ------------------------------------------------------------
# PRINT RESULTS
# ------------------------------------------------------------

print("\nAnnotation Counts:\n")

for label, count in sorted(counts.items()):
    print(f"{label}: {count}")