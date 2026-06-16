# ECG Arrhythmia Detector

A machine learning project for detecting cardiac arrhythmias from ECG (electrocardiogram) signals using the MIT-BIH Arrhythmia Database.

## Overview

This project builds a baseline machine learning model to classify ECG heartbeats as normal or abnormal (arrhythmia). It includes data preprocessing, feature extraction, model training, and evaluation.

## Dataset

The project uses the **MIT-BIH Arrhythmia Database**, a publicly available dataset of cardiac arrhythmia recordings.

**Download**: https://physionet.org/content/mitdb/1.0.0/

After downloading, extract the data to `data/mit-bih/` in the project root.

### Directory Structure
```
ecg-arrhythmia-detector/
├── data/
│   ├── mit-bih/              # MIT-BIH database files (download required)
│   │   ├── 100.hea
│   │   ├── 100.dat
│   │   ├── 100.atr
│   │   └── ...
│   ├── X.npy                 # Preprocessed ECG beats
│   └── y.npy                 # Labels
├── src/
│   ├── ecg_preprocessing.py  # Signal filtering and preprocessing
│   ├── build_dataset.py      # Extract beats from raw ECG data
│   ├── beats_and_annotations.py
│   ├── train_baseline.py     # Train Random Forest classifier
│   ├── plot_ecg.py           # Visualize ECG signals
│   ├── explore_annotations.py
│   ├── load_data.py
│   └── verify_class_distribution.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/keef-b/ecg-arrhythmia-detector.git
cd ecg-arrhythmia-detector
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
- Download MIT-BIH Arrhythmia Database from: https://physionet.org/content/mitdb/1.0.0/
- Extract to `data/mit-bih/`

## Usage

### Step 1: Build Dataset from Raw ECG Files
```bash
cd src
python build_dataset.py
```
This extracts heartbeat windows from raw ECG signals and saves them as numpy arrays.

### Step 2: Train Baseline Model
```bash
python train_baseline.py
```
This trains a Random Forest classifier and displays performance metrics.

### Step 3: Visualize ECG Signal
```bash
python plot_ecg.py
```
Plots a sample ECG recording with detected heartbeats marked.

### Step 4: Explore Annotations
```bash
python explore_annotations.py
```
Analyzes annotation labels in the database.

## Model Architecture

**Baseline Model**: Random Forest Classifier
- 100 decision trees
- Input features: 250 samples per heartbeat
- Output: Binary classification (Normal vs. Abnormal)

## Preprocessing Steps

1. **Bandpass Filter** (0.5-40 Hz)
   - Removes baseline wander and high-frequency noise
   - Preserves cardiac signal characteristics

2. **Normalization** (Z-score per beat)
   - Centers each heartbeat around zero
   - Normalizes by standard deviation
   - Makes model robust to amplitude variations

3. **Beat Extraction**
   - Extracts 250-sample windows around detected beats
   - 100 samples before beat, 150 samples after

## Results

The baseline model achieves performance metrics displayed in the classification report, including:
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Model Improvements

Potential enhancements to improve model performance:

### 1. Feature Engineering
- Frequency-domain features (FFT)
- Statistical features (peak detection, RMS)
- Wavelet features

### 2. Advanced Models
- Gradient Boosting (XGBoost, LightGBM)
- Support Vector Machines (SVM)
- Neural Networks (1D CNN, LSTM)

### 3. Hyperparameter Tuning
- GridSearchCV or RandomizedSearchCV
- Cross-validation (k-fold)

### 4. Data Augmentation
- Time-shifting
- Amplitude scaling
- Synthetic beat generation

### 5. Class Imbalance Handling
- Class weighting
- Oversampling/undersampling
- SMOTE

## Dependencies

See `requirements.txt` for full list:
- `numpy` - Numerical computing
- `scipy` - Scientific computing (signal processing)
- `scikit-learn` - Machine learning
- `wfdb` - ECG file I/O
- `matplotlib` - Visualization

## References

- MIT-BIH Arrhythmia Database: https://physionet.org/content/mitdb/1.0.0/
- WFDB Software Package: https://github.com/MIT-LCP/wfdb-python

## License

MIT License - See LICENSE file for details

## Contact

For questions or issues, please open an issue on GitHub.
