from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from prediction_artifacts import classification_metrics


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    return value if isinstance(value, dict) else {}


def sidecar_metadata(path: Path) -> dict[str, Any]:
    candidates = [
        path.with_name("prediction_metadata.json"),
        path.with_suffix(".json"),
        path.with_name("best_hparams.json"),
    ]
    metadata: dict[str, Any] = {}
    for candidate in candidates:
        if candidate == path or not candidate.is_file():
            continue
        try:
            metadata.update(read_json(candidate))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
    if path.name == "final_predictions.npz" and "model" not in metadata:
        metadata["model"] = "AIM-TSformer"
    return metadata


def load_artifact(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    metadata = sidecar_metadata(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as values:
            truth = np.asarray(values["y_true"]).reshape(-1)
            prediction = np.asarray(values["y_pred"]).reshape(-1)
            for key in ("model", "dataset", "seed"):
                if key in values.files and key not in metadata:
                    scalar = np.asarray(values[key]).reshape(-1)[0]
                    metadata[key] = scalar.item() if hasattr(scalar, "item") else scalar
        return truth, prediction, metadata
    if suffix == ".csv":
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        if not rows or "y_true" not in rows[0] or "y_pred" not in rows[0]:
            raise ValueError("CSV must contain y_true and y_pred columns")
        truth = np.asarray([row["y_true"] for row in rows])
        prediction = np.asarray([row["y_pred"] for row in rows])
        for key in ("model", "dataset", "seed"):
            values = {row.get(key, "") for row in rows} - {""}
            if len(values) == 1 and key not in metadata:
                metadata[key] = values.pop()
        return truth, prediction, metadata
    if suffix == ".json":
        payload = read_json(path)
        if "y_true" not in payload or "y_pred" not in payload:
            raise ValueError("JSON must contain y_true and y_pred")
        metadata.update({key: payload[key] for key in ("model", "dataset", "seed") if key in payload})
        return (
            np.asarray(payload["y_true"]).reshape(-1),
            np.asarray(payload["y_pred"]).reshape(-1),
            metadata,
        )
    raise ValueError(f"Unsupported prediction format: {path.suffix}")


def discover(roots: list[Path], explicit: list[Path]) -> list[Path]:
    files = list(explicit)
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        files.extend(root.rglob("*.npz"))
        files.extend(
            path for path in root.rglob("*.csv") if "prediction" in path.name.lower()
        )
        files.extend(
            path
            for path in root.rglob("*.json")
            if "prediction" in path.name.lower() and path.name != "prediction_metadata.json"
        )
    return sorted(dict.fromkeys(files), key=lambda path: str(path).lower())


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_fields = [
        "model", "dataset", "seed", "n_samples", "accuracy", "macro_f1",
        "balanced_accuracy", "artifact",
    ]
    write_csv(output_dir / "all_models_metrics_per_seed.csv", rows, per_seed_fields)

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["model"], row["dataset"]), []).append(row)
    summary: list[dict[str, Any]] = []
    for (model, dataset), group in sorted(groups.items()):
        item: dict[str, Any] = {"model": model, "dataset": dataset, "n_seeds": len(group)}
        for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
            values = np.asarray([float(row[metric]) for row in group], dtype=float)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(item)
    fields = [
        "model", "dataset", "n_seeds", "accuracy_mean", "accuracy_std",
        "macro_f1_mean", "macro_f1_std", "balanced_accuracy_mean",
        "balanced_accuracy_std",
    ]
    write_csv(output_dir / "all_models_metrics_mean_std.csv", summary, fields)

    models = sorted({row["model"] for row in summary})
    datasets = sorted({row["dataset"] for row in summary})
    lookup = {(row["model"], row["dataset"]): row for row in summary}
    for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
        table = []
        for dataset in datasets:
            item: dict[str, Any] = {"dataset": dataset}
            for model in models:
                value = lookup.get((model, dataset))
                item[model] = (
                    f"{value[f'{metric}_mean']:.3f} ± {value[f'{metric}_std']:.3f}"
                    if value
                    else ""
                )
            table.append(item)
        write_csv(output_dir / f"table_{metric}_mean_std.csv", table, ["dataset", *models])


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", action="append", type=Path, default=[])
    parser.add_argument("--prediction-file", action="append", type=Path, default=[])
    parser.add_argument("--model", help="Fallback model name for artifacts without metadata")
    parser.add_argument("--dataset", help="Fallback dataset name for artifacts without metadata")
    parser.add_argument("--seed", type=int, help="Fallback seed for artifacts without metadata")
    parser.add_argument("--output-dir", type=Path, default=here / "results_prediction_metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = discover(args.prediction_root, args.prediction_file)
    if not files:
        raise FileNotFoundError("No prediction artifacts were found")
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for path in files:
        try:
            truth, prediction, metadata = load_artifact(path)
            model = str(metadata.get("model") or args.model or "").strip()
            dataset = str(metadata.get("dataset") or args.dataset or "").strip()
            seed = metadata.get("seed", args.seed)
            if not model or not dataset or seed is None:
                raise KeyError("model, dataset, and seed metadata are required")
            metrics = classification_metrics(truth, prediction)
            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "seed": int(seed),
                    "n_samples": int(truth.size),
                    **metrics,
                    "artifact": str(path.resolve()),
                }
            )
        except Exception as error:
            skipped.append({"artifact": str(path), "error": f"{type(error).__name__}: {error}"})
            print(f"[SKIP] {path}: {type(error).__name__}: {error}")
    if not rows:
        raise RuntimeError("No valid prediction artifact was found")
    aggregate(rows, args.output_dir)
    write_csv(args.output_dir / "skipped_prediction_artifacts.csv", skipped, ["artifact", "error"])
    print(f"Evaluated {len(rows)} artifact(s); skipped {len(skipped)}.")


if __name__ == "__main__":
    main()
