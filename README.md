# BiGRU Attention Speaker Identification

Web demo and training code for **closed-set speaker identification** on 100 speakers from the VCTK corpus. The repository intentionally contains one model only: `best_bigru_attention_100.pth`, a 2-layer bidirectional GRU with temporal attention.

The web application accepts one recording and returns the most likely speaker plus the top five speaker probabilities. It is designed for the 100 speakers stored in the checkpoint; it cannot identify an unseen person as a new class.

## Project layout

```text
SpeechProcessing/
|-- app.py                    # FastAPI server and prediction endpoints
|-- audio_features.py         # Shared preprocessing implementation
|-- model.py                  # BiGRU + Attention definition
|-- train.py                  # Training, validation and test evaluation
|-- models/
|   `-- best_bigru_attention_100.pth
|-- static/                   # Browser interface
|-- requirements.txt
`-- README.md
```

## Requirements

- Python 3.10, 3.11 or 3.12. Python 3.13 is not supported by the current `librosa` feature stack used by this project.
- `pip`
- Optional: NVIDIA GPU with a CUDA-compatible PyTorch install for faster training and inference

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the web demo

The checkpoint is already included at `models/best_bigru_attention_100.pth`.

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Upload WAV, MP3, FLAC, OGG or M4A audio. The server automatically uses CUDA when PyTorch detects it; otherwise it runs on CPU.

API endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Model, device, feature shape and readiness |
| `GET` | `/api/speakers` | 100 speaker IDs encoded by the checkpoint |
| `POST` | `/api/predict` | Predict from multipart field `file` |

Example request:

```bash
curl -X POST -F "file=@recording.wav" http://127.0.0.1:8000/api/predict
```

## Dataset format

Download VCTK and arrange the selected speaker audio in a directory tree like this:

```text
vctk_100/
|-- p225/
|   |-- p225_001.wav
|   `-- ...
|-- p226/
|   |-- p226_001.wav
|   `-- ...
`-- ...
```

`train.py` chooses the first 100 `p###` directories in sorted order. Use the same speaker list when training and serving; the label mapping is saved inside the checkpoint as `speaker_to_label` and `label_to_speaker`.

## Audio preprocessing

The training script and API call the same `extract_features()` function, so inference uses the exact feature contract used during training.

1. **Mono and 16 kHz resampling**: `librosa.load(..., sr=16000, mono=True)` mixes multi-channel recordings to mono and resamples every input to 16 kHz.
2. **Pre-emphasis**: `y[t] = x[t] - 0.97 * x[t - 1]` boosts higher-frequency detail before cepstral analysis.
3. **MFCCs**: extract 40 coefficients for every analysis frame.
4. **Dynamic features**: compute 40 first-order delta coefficients and 40 second-order delta-delta coefficients.
5. **Concatenation**: combine them to a 120-dimensional vector per frame: `40 + 40 + 40 = 120`.
6. **Per-utterance normalization**: normalize each feature dimension to zero mean and unit standard deviation.
7. **Fixed length**: truncate sequences longer than 200 frames and append zero frames to shorter sequences. Final input shape is `[200, 120]`.

## Model architecture

| Stage | Configuration | Output |
| --- | --- | --- |
| Input | normalized MFCC feature sequence | `[batch, 200, 120]` |
| BiGRU | 2 layers, 256 hidden units per direction, bidirectional | `[batch, 200, 512]` |
| Attention | `Linear(512, 1)` + softmax over time | `[batch, 512]` |
| Classifier | `Linear(512, 256)`, ReLU, Dropout `0.4`, `Linear(256, 100)` | `[batch, 100]` |
| Output | softmax at inference | 100 class probabilities |

The trained checkpoint uses the PyTorch parameter names `bigru.*`, `attention.attention.*`, and `classifier.0` / `classifier.3`. `model.py` preserves that layout so it can load `best_bigru_attention_100.pth` directly.

## Train, validate and test

The default configuration follows the project pipeline:

| Parameter | Value |
| --- | --- |
| Speakers | 100 |
| Split | 81% train / 9% validation / 10% test per speaker |
| Batch size | 32 |
| Epochs | 50 |
| Optimizer | AdamW |
| Learning rate | 0.0005 |
| Loss | Cross entropy |

Run training:

```bash
python train.py --data-root /path/to/vctk_100
```

Useful overrides:

```bash
python train.py \
  --data-root /path/to/vctk_100 \
  --output models/best_bigru_attention_100.pth \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 0.0005
```

During each epoch, the script trains on the 81% split and validates on the 9% split. When validation accuracy improves, it overwrites the output checkpoint with:

- `model_state_dict`
- speaker-to-label mappings
- architecture metadata (`num_speakers`, `n_mfcc`, `dropout`)
- best validation accuracy

After the final epoch, it reloads the best checkpoint and evaluates the held-out 10% test split. Metrics are written to `models/test_metrics.json`:

- Accuracy
- Weighted precision
- Weighted recall
- Weighted F1-score

## Notes and limitations

- This is a **closed-set** classifier. Predictions are always one of the 100 speakers known during training.
- Use recordings with speech and limited background noise for sensible results.
- Do not compare scores across a model retrained with a different speaker list or preprocessing configuration.
- The dataset is deliberately excluded from Git. Keep VCTK locally and pass its location with `--data-root`.

## License and data

This repository contains project code and a trained checkpoint. Obtain and use VCTK according to its own license and terms. Do not commit private recordings or a redistributed VCTK corpus to the repository.
