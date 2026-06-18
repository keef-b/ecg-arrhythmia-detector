# ecg_preprocessing.py

import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, fs=360, lowcut=0.5, highcut=40, order=4):
    """
    Clean ECG signal by removing:
    - baseline wander (<0.5 Hz)
    - high frequency noise (>40 Hz)
    """

    nyquist = 0.5 * fs

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype="band")

    return filtfilt(b, a, signal)