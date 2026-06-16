# ============================================================
# TRAIN BASELINE MODEL FOR ECG CLASSIFICATION
# ============================================================
#
# Goal of this script:
#   We are NOT trying to build the best model yet.
#   We are building a "baseline model" to answer:
#
#   → "Can a simple ML model learn anything useful from ECG beats?"
#
# This is the first step in any real ML project.
# ============================================================


# ============================================================
# IMPORTS (libraries we need)
# ============================================================

import numpy as np

# train_test_split:
#   splits your dataset into training data and test data
from sklearn.model_selection import train_test_split

# RandomForestClassifier:
#   a tree-based machine learning model
#   it learns patterns by splitting data into decision rules
from sklearn.ensemble import RandomForestClassifier

# evaluation tools:
#   confusion_matrix → shows correct vs incorrect predictions
#   classification_report → precision, recall, F1 score
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# STEP 1: LOAD YOUR DATASET
# ============================================================

# X = ECG beats (input features)
# shape = (num_samples, 250)
# each row = one heartbeat waveform

X = np.load("../data/X.npy")

# y = labels for each heartbeat
# 0 = normal
# 1 = abnormal

y = np.load("../data/y.npy")

print("\n====================")
print("DATA LOADED")
print("====================")

print("X shape:", X.shape)  # how many beats, how many samples per beat
print("y shape:", y.shape)  # number of labels


# ============================================================
# STEP 2: CHECK CLASS DISTRIBUTION
# ============================================================
#
# VERY IMPORTANT STEP:
# We want to know if the dataset is balanced or not.
# If one class dominates, accuracy can be misleading.

unique, counts = np.unique(y, return_counts=True)

print("\n====================")
print("CLASS DISTRIBUTION")
print("====================")

for label, count in zip(unique, counts):
    if label == 0:
        print("Normal (0):", count)
    else:
        print("Abnormal (1):", count)


# ============================================================
# STEP 3: NORMALIZATION
# ============================================================
#
# WHY DO THIS?
# ECG signals have different amplitudes depending on:
#   - patient
#   - electrode placement
#   - recording conditions
#
# We want to remove amplitude bias so the model learns SHAPE,
# not raw voltage magnitude.

# axis=1 means:
#   normalize EACH heartbeat independently

X = (X - np.mean(X, axis=1, keepdims=True)) / (
    np.std(X, axis=1, keepdims=True) + 1e-8
)

# 1e-8 prevents division by zero


# ============================================================
# STEP 4: TRAIN / TEST SPLIT
# ============================================================
#
# We split data into:
#
# TRAINING SET → model learns patterns from this
# TEST SET     → model is evaluated on unseen data
#
# IMPORTANT:
# stratify=y ensures class balance is preserved in both sets

X_train, X_test, y_train, y_test = train_test_split(
    X,                  # input data
    y,                  # labels
    test_size=0.2,      # 20% test, 80% train
    random_state=42,    # makes results reproducible
    stratify=y          # keeps class distribution balanced
)


# ============================================================
# STEP 5: DEFINE MODEL
# ============================================================
#
# Random Forest = ensemble of decision trees
#
# Each tree:
#   learns rules like:
#   "if value at time 120 > threshold → abnormal"
#
# Many trees together → more stable predictions

model = RandomForestClassifier(
    n_estimators=100,  # number of trees in the forest
    n_jobs=-1,         # use all CPU cores (faster training)
    random_state=42    # reproducibility
)


# ============================================================
# STEP 6: TRAIN MODEL
# ============================================================
#
# This is where the model learns patterns in ECG beats.

model.fit(X_train, y_train)


# ============================================================
# STEP 7: MAKE PREDICTIONS
# ============================================================
#
# Now we test the model on unseen data.

y_pred = model.predict(X_test)


# ============================================================
# STEP 8: EVALUATION (MOST IMPORTANT PART)
# ============================================================
#
# This tells us how well the model performs.
#
# Confusion Matrix:
#   shows:
#     true positives
#     false positives
#     true negatives
#     false negatives
#
# Classification Report:
#   precision → how many predicted positives were correct
#   recall    → how many real positives were found
#   f1-score  → balance of precision and recall

print("\n====================")
print("CONFUSION MATRIX")
print("====================")
print(confusion_matrix(y_test, y_pred))


print("\n====================")
print("CLASSIFICATION REPORT")
print("====================")
print(classification_report(y_test, y_pred))


# ============================================================
# END OF SCRIPT
# ============================================================
#
# At this point you have:
#   - trained a model
#   - evaluated it
#   - established a baseline
#
