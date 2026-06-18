import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

y = np.load(PROCESSED_DIR / "y.npy")

unique, counts = np.unique(y, return_counts=True)

print("Class Distribution")
print("-" * 30)

for label, count in zip(unique, counts):
    print(f"{label}: {count}")

