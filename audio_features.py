"""Audio preprocessing shared by model training and web inference."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

SAMPLE_RATE = 16_000
PRE_EMPHASIS = 0.97
N_MFCC = 40
FEATURE_DIM = N_MFCC * 3
MAX_FRAMES = 200


def _delta(features: np.ndarray, order: int) -> np.ndarray:
    """Return stable delta features even for very short audio clips."""
    frame_count = features.shape[1]
    width = min(9, frame_count if frame_count % 2 else frame_count - 1)
    if width < 3:
        return np.zeros_like(features)
    return librosa.feature.delta(features, width=width, order=order)


def extract_features(audio_path: str | Path) -> np.ndarray:
    """Create the normalized, padded ``[200, 120]`` feature tensor."""
    signal, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    if signal.size == 0:
        raise ValueError("The audio file contains no samples.")

    emphasized = np.empty_like(signal)
    emphasized[0] = signal[0]
    emphasized[1:] = signal[1:] - PRE_EMPHASIS * signal[:-1]

    mfcc = librosa.feature.mfcc(y=emphasized, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    features = np.concatenate((mfcc, _delta(mfcc, 1), _delta(mfcc, 2)), axis=0).T
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

    padded = np.zeros((MAX_FRAMES, FEATURE_DIM), dtype=np.float32)
    usable_frames = min(MAX_FRAMES, features.shape[0])
    padded[:usable_frames] = features[:usable_frames]
    return padded
