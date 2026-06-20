"""FastAPI inference service for the 100-speaker BiGRU Attention model."""

from __future__ import annotations

import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from audio_features import FEATURE_DIM, MAX_FRAMES, N_MFCC, SAMPLE_RATE, extract_features
from model import BiGRUAttention

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "best_bigru_attention_100.pth"
STATIC_DIR = ROOT_DIR / "static"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


class Predictor:
    def __init__(self, checkpoint_path: Path) -> None:
        self.checkpoint_path = checkpoint_path
        self.model: BiGRUAttention | None = None
        self.label_to_speaker: dict[int, str] = {}

    def load(self) -> None:
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=DEVICE, weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.checkpoint_path, map_location=DEVICE)

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        speaker_count = int(checkpoint.get("num_speakers", 100))
        self.model = BiGRUAttention(
            input_size=FEATURE_DIM,
            hidden_size=256,
            num_layers=2,
            num_speakers=speaker_count,
            dropout=float(checkpoint.get("dropout", 0.4)),
        ).to(DEVICE)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        stored_labels = checkpoint.get("label_to_speaker")
        if stored_labels:
            self.label_to_speaker = {int(label): str(speaker) for label, speaker in stored_labels.items()}
        else:
            speakers = sorted(path.name for path in (ROOT_DIR / "demo_test_set_100").glob("p*") if path.is_dir())
            self.label_to_speaker = dict(enumerate(speakers))

    def predict(self, audio_path: Path, top_k: int = 5) -> list[dict[str, float | int | str]]:
        if self.model is None:
            raise RuntimeError("The model has not been loaded.")

        features = torch.from_numpy(extract_features(audio_path)).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(features), dim=1)[0]

        values, labels = torch.topk(probabilities, k=min(top_k, probabilities.numel()))
        return [
            {
                "rank": rank,
                "label": int(label),
                "speaker": self.label_to_speaker.get(int(label), f"class_{int(label)}"),
                "confidence": round(float(value) * 100, 2),
            }
            for rank, (value, label) in enumerate(zip(values.cpu(), labels.cpu()), start=1)
        ]


predictor = Predictor(MODEL_PATH)


@asynccontextmanager
async def lifespan(_: FastAPI):
    predictor.load()
    yield


app = FastAPI(title="BiGRU Speaker Identification", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "ready" if predictor.model is not None else "loading",
        "model": MODEL_PATH.name,
        "device": str(DEVICE),
        "sample_rate": SAMPLE_RATE,
        "input_shape": [MAX_FRAMES, FEATURE_DIM],
        "speakers": len(predictor.label_to_speaker),
    }


@app.get("/api/speakers")
def speakers() -> dict[str, list[str]]:
    return {"speakers": [predictor.label_to_speaker[key] for key in sorted(predictor.label_to_speaker)]}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Supported formats: WAV, MP3, FLAC, OGG, and M4A.")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
            temporary_path = Path(temporary_file.name)
            shutil.copyfileobj(file.file, temporary_file)

        predictions = predictor.predict(temporary_path)
        return {
            "file_name": file.filename,
            "prediction": predictions[0],
            "top_predictions": predictions,
        }
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        await file.close()
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()
