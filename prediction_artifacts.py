from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def classification_metrics(y_true: Sequence, y_pred: Sequence) -> dict[str, float]:
    truth = np.asarray(y_true).reshape(-1)
    prediction = np.asarray(y_pred).reshape(-1)
    if truth.size == 0:
        raise ValueError("Cannot evaluate an empty prediction array")
    if truth.shape != prediction.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape: {truth.shape} vs {prediction.shape}"
        )
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
    }


def save_prediction_artifact(
    output_dir: str | Path,
    *,
    model: str,
    dataset: str,
    seed: int,
    y_true: Sequence,
    y_pred: Sequence,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    truth = np.asarray(y_true).reshape(-1)
    prediction = np.asarray(y_pred).reshape(-1)
    metrics = classification_metrics(truth, prediction)

    array_path = destination / "final_predictions.npz"
    metadata_path = destination / "prediction_metadata.json"
    np.savez_compressed(array_path, y_true=truth, y_pred=prediction)
    payload: dict[str, Any] = {
        "artifact_schema": "aim-tsformer-classification-predictions-v1",
        "model": str(model),
        "dataset": str(dataset),
        "seed": int(seed),
        "n_samples": int(truth.size),
        **metrics,
        **dict(metadata or {}),
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return array_path, metadata_path


__all__ = ["classification_metrics", "save_prediction_artifact"]
