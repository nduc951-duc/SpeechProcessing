"""BiGRU100 preprocessing matched to the trained checkpoint."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

SAMPLE_RATE = 16_000
PRE_EMPHASIS = 0.97
N_MFCC = 40
FEATURE_DIM = N_MFCC * 3
MAX_FRAMES = 200


N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 64


def _pad_or_center_crop(features: np.ndarray) -> np.ndarray:
    """Match the train-time 200-frame center crop and zero padding."""
    if features.shape[0] > MAX_FRAMES:
        start = (features.shape[0] - MAX_FRAMES) // 2
        return features[start : start + MAX_FRAMES]
    if features.shape[0] < MAX_FRAMES:
        return np.pad(features, ((0, MAX_FRAMES - features.shape[0]), (0, 0)), mode="constant")
    return features


def extract_features(audio_path: str | Path) -> np.ndarray:
    """Create the train-compatible ``[200, 120]`` BiGRU100 input."""
    signal, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    if signal.size == 0:
        raise ValueError("The audio file contains no samples.")

    emphasized = np.append(signal[0], signal[1:] - PRE_EMPHASIS * signal[:-1])
    mfcc = librosa.feature.mfcc(
        y=emphasized,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        window="hamming",
    )
    features = np.concatenate(
        (mfcc, librosa.feature.delta(mfcc), librosa.feature.delta(mfcc, order=2)), axis=0
    ).T
    features = (features - features.mean(axis=0, keepdims=True)) / (
        features.std(axis=0, keepdims=True) + 1e-9
    )
    return _pad_or_center_crop(features).astype(np.float32, copy=False)
