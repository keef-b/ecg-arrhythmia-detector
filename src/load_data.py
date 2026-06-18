from pathlib import Path

import wfdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "mit-bih-arrhythmia-database-1.0.0"

record_path = str(RAW_DATA_DIR / "100")
record = wfdb.rdrecord(record_path)
ann = wfdb.rdann(record_path, 'atr')

print(record.p_signal.shape)
print(ann.symbol[:20])
