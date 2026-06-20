"""Train and evaluate the BiGRU Attention speaker identification model."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from audio_features import extract_features
from model import BiGRUAttention

SEED = 42


class SpeakerDataset(Dataset):
    def __init__(self, records: list[tuple[Path, int]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.records[index]
        return torch.from_numpy(extract_features(path)), torch.tensor(label, dtype=torch.long)


def discover_records(data_root: Path, speaker_limit: int) -> tuple[list[tuple[Path, int]], dict[str, int]]:
    speaker_dirs = sorted(path for path in data_root.iterdir() if path.is_dir() and path.name.startswith("p"))[:speaker_limit]
    if len(speaker_dirs) != speaker_limit:
        raise ValueError(f"Expected at least {speaker_limit} speaker directories in {data_root}.")

    speaker_to_label = {directory.name: index for index, directory in enumerate(speaker_dirs)}
    records: list[tuple[Path, int]] = []
    for directory in speaker_dirs:
        records.extend((audio_path, speaker_to_label[directory.name]) for audio_path in sorted(directory.rglob("*.wav")))
    if not records:
        raise ValueError("No WAV files were found below the selected speaker directories.")
    return records, speaker_to_label


def split_per_speaker(records: list[tuple[Path, int]]) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[tuple[Path, int]]]:
    grouped: dict[int, list[tuple[Path, int]]] = defaultdict(list)
    for record in records:
        grouped[record[1]].append(record)

    generator = random.Random(SEED)
    train_records: list[tuple[Path, int]] = []
    validation_records: list[tuple[Path, int]] = []
    test_records: list[tuple[Path, int]] = []
    for speaker_records in grouped.values():
        generator.shuffle(speaker_records)
        count = len(speaker_records)
        train_end = max(1, int(count * 0.81))
        validation_end = max(train_end + 1, int(count * 0.90))
        train_records.extend(speaker_records[:train_end])
        validation_records.extend(speaker_records[train_end:validation_end])
        test_records.extend(speaker_records[validation_end:])
    return train_records, validation_records, test_records


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: AdamW | None, device: torch.device) -> tuple[float, list[int], list[int]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels: list[int] = []
    predictions: list[int] = []

    for features, targets in loader:
        features, targets = features.to(device), targets.to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(features)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * targets.size(0)
        labels.extend(targets.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())

    return total_loss / len(loader.dataset), labels, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BiGRU Attention on 100 VCTK speakers.")
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing p### speaker folders.")
    parser.add_argument("--output", type=Path, default=Path("models/best_bigru_attention_100.pth"))
    parser.add_argument("--speakers", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records, speaker_to_label = discover_records(args.data_root, args.speakers)
    train_records, validation_records, test_records = split_per_speaker(records)
    print(f"Split: train={len(train_records)}, validation={len(validation_records)}, test={len(test_records)}")

    loaders = [
        DataLoader(SpeakerDataset(records_for_split), batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers)
        for records_for_split, shuffle in ((train_records, True), (validation_records, False), (test_records, False))
    ]
    model = BiGRUAttention(num_speakers=args.speakers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    best_validation_accuracy = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_labels, train_predictions = run_epoch(model, loaders[0], criterion, optimizer, device)
        validation_loss, validation_labels, validation_predictions = run_epoch(model, loaders[1], criterion, None, device)
        validation_accuracy = accuracy_score(validation_labels, validation_predictions)
        train_accuracy = accuracy_score(train_labels, train_predictions)
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.4f}, acc {train_accuracy:.4f} | val loss {validation_loss:.4f}, acc {validation_accuracy:.4f}")

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "speaker_to_label": speaker_to_label,
                    "label_to_speaker": {label: speaker for speaker, label in speaker_to_label.items()},
                    "num_speakers": args.speakers,
                    "n_mfcc": 40,
                    "dropout": 0.4,
                    "validation_accuracy": validation_accuracy,
                },
                args.output,
            )

    checkpoint = torch.load(args.output, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_labels, test_predictions = run_epoch(model, loaders[2], criterion, None, device)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_predictions, average="weighted", zero_division=0)
    metrics = {
        "test_loss": test_loss,
        "accuracy": accuracy_score(test_labels, test_predictions),
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
    }
    print(json.dumps(metrics, indent=2))
    (args.output.parent / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
