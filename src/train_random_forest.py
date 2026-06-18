# ============================================================
# FAST BASELINE MODEL WITH VALIDATION & CONFUSION MATRIX
# ============================================================
#
# Speed optimizations:
# 1. Removed 5-fold cross-validation (use single train/test)
# 2. Removed most visualizations (keep confusion matrix only)
# 3. Optimized model parameters
# 4. Kept overfitting detection and detailed metrics
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score
)

try:
    import seaborn as sns
    SNS_AVAILABLE = True
except ImportError:
    SNS_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"


# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

print("\n" + "="*50)
print("LOADING DATA")
print("="*50)

X = np.load(PROCESSED_DIR / "X.npy")
y = np.load(PROCESSED_DIR / "y.npy")

print(f"Total samples: {X.shape[0]:,}")
print(f"Features per sample: {X.shape[1]}")


# ============================================================
# STEP 2: CHECK CLASS DISTRIBUTION
# ============================================================

unique, counts = np.unique(y, return_counts=True)

print("\n" + "="*50)
print("CLASS DISTRIBUTION")
print("="*50)

for label, count in zip(unique, counts):
    percentage = (count / len(y)) * 100
    if label == 0:
        print(f"Normal (0):   {count:>7,} ({percentage:.1f}%)")
    else:
        print(f"Abnormal (1): {count:>7,} ({percentage:.1f}%)")

imbalance_ratio = counts[0] / counts[1]
print(f"\n⚠️  Class Imbalance: {imbalance_ratio:.2f}:1 (Normal:Abnormal)")


# ============================================================
# STEP 3: NORMALIZATION
# ============================================================

print("\n" + "="*50)
print("PREPROCESSING")
print("="*50)

X = (X - np.mean(X, axis=1, keepdims=True)) / (
    np.std(X, axis=1, keepdims=True) + 1e-8
)

print("✓ Z-score normalization complete")


# ============================================================
# STEP 4: TRAIN/TEST SPLIT
# ============================================================

print("\n" + "="*50)
print("TRAIN/TEST SPLIT")
print("="*50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training set: {X_train.shape[0]:,} samples (80%)")
print(f"Test set:     {X_test.shape[0]:,} samples (20%)")
print("✓ Stratified split (no data leakage)")


# ============================================================
# STEP 5: TRAIN MODEL (FAST VERSION)
# ============================================================

print("\n" + "="*50)
print("TRAINING MODEL")
print("="*50)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

print("Training (this may take 1-2 minutes)...\n")
model.fit(X_train, y_train)
print("\n✓ Model trained")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_DIR / "random_forest_binary.joblib")
print(f"✓ Saved model: {MODEL_DIR / 'random_forest_binary.joblib'}")


# ============================================================
# STEP 6: PREDICTIONS
# ============================================================

print("\n" + "="*50)
print("MAKING PREDICTIONS")
print("="*50)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

y_train_pred_proba = model.predict_proba(X_train)[:, 1]
y_test_pred_proba = model.predict_proba(X_test)[:, 1]

print("✓ Predictions complete")


# ============================================================
# STEP 7: OVERFITTING DETECTION
# ============================================================

print("\n" + "="*50)
print("OVERFITTING ANALYSIS")
print("="*50)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

train_f1 = f1_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)

train_roc = roc_auc_score(y_train, y_train_pred_proba)
test_roc = roc_auc_score(y_test, y_test_pred_proba)

print(f"\n{'Metric':<15} {'Train':<10} {'Test':<10} {'Diff':<10}")
print("-" * 50)
print(f"{'Accuracy':<15} {train_acc:.4f}{'':<5} {test_acc:.4f}{'':<5} {train_acc-test_acc:+.4f}")
print(f"{'F1-Score':<15} {train_f1:.4f}{'':<5} {test_f1:.4f}{'':<5} {train_f1-test_f1:+.4f}")
print(f"{'ROC-AUC':<15} {train_roc:.4f}{'':<5} {test_roc:.4f}{'':<5} {train_roc-test_roc:+.4f}")

overfitting_gap = train_acc - test_acc
if overfitting_gap > 0.1:
    print(f"\n⚠️  OVERFITTING DETECTED (gap > 0.10)")
elif overfitting_gap > 0.05:
    print(f"\n~ Slight overfitting (gap > 0.05)")
else:
    print(f"\n✓ Good generalization (train ≈ test)")


# ============================================================
# STEP 8: TEST SET EVALUATION
# ============================================================

print("\n" + "="*50)
print("TEST SET PERFORMANCE")
print("="*50)

cm = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0

print(f"\n{'Confusion Matrix:':<20}")
print(f"{'':20} Predicted")
print(f"{'':20} Normal  Abnormal")
print(f"Actual Normal       {cm[0,0]:<7} {cm[0,1]:<7}")
print(f"       Abnormal     {cm[1,0]:<7} {cm[1,1]:<7}")

print(f"\n{'CRITICAL METRICS':<30} VALUE")
print("-" * 50)
print(f"{'Sensitivity (catching abnormal)':<30} {sensitivity:.4f} ({sensitivity*100:.1f}%)")
print(f"{'Specificity (identifying normal)':<30} {specificity:.4f} ({specificity*100:.1f}%)")
print(f"{'Precision (abnormal predictions)':<30} {precision:.4f}")
print(f"{'False Negatives (MISSED abnormal):':<30} {fn}")

print("\n" + "="*50)
print("DETAILED CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_test, y_test_pred, 
                          target_names=['Normal', 'Abnormal']))


# ============================================================
# STEP 9: CONFUSION MATRIX VISUALIZATION
# ============================================================

print("\n" + "="*50)
print("GENERATING CONFUSION MATRIX")
print("="*50)

fig, ax = plt.subplots(figsize=(8, 6))

if SNS_AVAILABLE:
    # Use seaborn for better visualization
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Abnormal'],
                yticklabels=['Normal', 'Abnormal'],
                cbar_kws={'label': 'Count'},
                ax=ax)
else:
    # Fallback to matplotlib
    ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Abnormal'])
    ax.set_yticklabels(['Normal', 'Abnormal'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', 
                   color='white', fontsize=16, fontweight='bold')

ax.set_title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12)

plt.tight_layout()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR / 'confusion_matrix.png'}")
plt.show()


# ============================================================
# STEP 10: SUMMARY
# ============================================================

print("\n" + "="*50)
print("RESULTS SUMMARY")
print("="*50)

print(f"""
DATA INTEGRITY:
✓ Stratified train/test split → No data leakage

MODEL PERFORMANCE:
• Test Accuracy:  {test_acc:.4f}
• Test F1-Score:  {test_f1:.4f}
• Test ROC-AUC:   {test_roc:.4f}

CLINICAL METRICS:
• Sensitivity (Recall):  {sensitivity:.4f} ({sensitivity*100:.1f}%)
  → Model catches {int(sensitivity*100)}% of abnormal cases
  → Misses {int((1-sensitivity)*100)}% of abnormal cases ⚠️
  
• Specificity:          {specificity:.4f} ({specificity*100:.1f}%)
  → Correctly identifies {int(specificity*100)}% of normal cases

ISSUES DETECTED:
""")

if sensitivity < 0.90:
    print(f"⚠️  Low Sensitivity ({sensitivity:.1%})")
    print("   → Too many abnormal cases are being missed!")
    print("   → Consider: threshold adjustment, class weighting, or better model")
else:
    print(f"✓ Good Sensitivity ({sensitivity:.1%})")

if overfitting_gap > 0.1:
    print(f"⚠️  Overfitting Detected (gap: {overfitting_gap:.4f})")
    print("   → Model performs better on training than test data")
else:
    print(f"✓ No significant overfitting")

print("\n" + "="*50)
print("Analysis Complete!")
print("="*50)
