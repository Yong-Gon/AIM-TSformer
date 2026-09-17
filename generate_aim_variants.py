from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


VARIANTS = (
    "original",
    "no_exp",
    "no_abs",
    "no_exp_no_abs",
    "angle_only",
    "sign_only",
    "diagonal_only",
    "symmetric",
    "flat_as_positive",
)
VARIANT_ALIASES = {
    "no exp": "no_exp",
    "no ab": "no_abs",
    "no abs": "no_abs",
    "no exp, ab": "no_exp_no_abs",
    "no exp, abs": "no_exp_no_abs",
    "diag_only": "diagonal_only",
}


def normalize_variant(name: str) -> str:
    result = VARIANT_ALIASES.get(name.strip().lower(), name.strip().lower())
    if result not in VARIANTS:
        raise ValueError(f"Unknown AIM variant {name!r}. Available: {VARIANTS}")
    return result


def gradient_sign_numpy(values: np.ndarray, flat_as_positive: bool = False) -> np.ndarray:
    d1 = values[1:-1] - values[:-2]
    d2 = values[2:] - values[1:-1]
    sign = np.sign(d1).astype(np.float32)
    use_second = sign == 0
    sign[use_second] = np.sign(d2[use_second]).astype(np.float32)
    if flat_as_positive:
        sign[sign == 0] = 1.0
    else:
        sign[(d1 == 0) & (d2 == 0)] = 0.0
    return sign


def local_values_numpy(values: np.ndarray, variant: str) -> np.ndarray:
    variant = normalize_variant(variant)
    if len(values) < 3:
        return np.zeros(0, dtype=np.float32)
    d1 = values[1:-1] - values[:-2]
    d2 = values[2:] - values[1:-1]
    cosine = (1.0 + d1 * d2) / (
        np.sqrt(1.0 + d1 * d1) * np.sqrt(1.0 + d2 * d2)
    )
    cosine = np.clip(cosine, -1.0, 1.0).astype(np.float32)
    sign = gradient_sign_numpy(values, flat_as_positive=variant == "flat_as_positive")
    if variant == "angle_only":
        return np.exp(np.abs(cosine)).astype(np.float32)
    if variant == "sign_only":
        return sign
    if variant == "no_exp":
        return (sign * np.abs(cosine)).astype(np.float32)
    if variant == "no_abs":
        return (sign * np.exp(cosine)).astype(np.float32)
    if variant == "no_exp_no_abs":
        return (sign * cosine).astype(np.float32)
    return (sign * np.exp(np.abs(cosine))).astype(np.float32)


def fill_local_band_numpy(matrix: np.ndarray, local: np.ndarray) -> None:
    length = matrix.shape[0]
    padded = np.zeros(length, dtype=np.float32)
    if len(local):
        padded[:len(local)] = local
        padded[len(local):] = local[-1]
    matrix[np.diag_indices(length)] = padded
    if length > 1:
        selected = np.where(
            np.abs(padded[:-1]) >= np.abs(padded[1:]), padded[:-1], padded[1:]
        )
        index = np.arange(length - 1)
        matrix[index, index + 1] = selected
        matrix[index + 1, index] = selected


def aim_numpy_single(values: np.ndarray, variant: str) -> np.ndarray:
    variant = normalize_variant(variant)
    length = len(values)
    local = local_values_numpy(values.astype(np.float32, copy=False), variant)
    matrix = np.zeros((length, length), dtype=np.float32)
    if variant != "diagonal_only" and len(local):
        prefix = np.concatenate(([0.0], np.cumsum(local, dtype=np.float64)))
        row = np.arange(length)[:, None]
        column = np.arange(length)[None, :]
        valid = column - row >= 2
        end = np.clip(column - 1, 0, length - 2)
        start = np.clip(row, 0, length - 2)
        upper = np.zeros_like(matrix)
        sums = prefix[end] - prefix[start]
        counts = np.maximum(column - row - 1, 1)
        upper[valid] = (sums[valid] / counts[valid]).astype(np.float32)
        matrix = upper + upper.T if variant == "symmetric" else upper - upper.T
    fill_local_band_numpy(matrix, local)
    return matrix


def _torch_batch_aim(batch: np.ndarray, variant: str, device: str) -> np.ndarray:
    import torch

    variant = normalize_variant(variant)
    x = torch.as_tensor(batch, dtype=torch.float32, device=device)
    batch_size, length, channels = x.shape
    output = torch.zeros((batch_size, length, length, channels), device=device)
    if length < 3:
        return output.cpu().numpy()

    d1 = x[:, 1:-1] - x[:, :-2]
    d2 = x[:, 2:] - x[:, 1:-1]
    cosine = ((1.0 + d1 * d2) / (
        torch.sqrt(1.0 + d1.square()) * torch.sqrt(1.0 + d2.square())
    )).clamp(-1.0, 1.0)
    sign = torch.sign(d1)
    sign = torch.where(sign == 0, torch.sign(d2), sign)
    if variant == "flat_as_positive":
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    else:
        sign = torch.where((d1 == 0) & (d2 == 0), torch.zeros_like(sign), sign)

    if variant == "angle_only":
        local = torch.exp(cosine.abs())
    elif variant == "sign_only":
        local = sign
    elif variant == "no_exp":
        local = sign * cosine.abs()
    elif variant == "no_abs":
        local = sign * torch.exp(cosine)
    elif variant == "no_exp_no_abs":
        local = sign * cosine
    else:
        local = sign * torch.exp(cosine.abs())

    if variant != "diagonal_only":
        prefix = torch.nn.functional.pad(local, (0, 0, 1, 0)).cumsum(dim=1)
        row = torch.arange(length, device=device)[:, None]
        column = torch.arange(length, device=device)[None, :]
        start = row.clamp(0, length - 2)
        end = (column - 1).clamp(0, length - 2)
        sums = prefix[:, end, :] - prefix[:, start, :]
        counts = (column - row - 1).clamp_min(1).to(x.dtype)
        upper = torch.where(
            (column - row >= 2)[None, :, :, None],
            sums / counts[None, :, :, None],
            torch.zeros_like(sums),
        )
        output = upper + upper.transpose(1, 2) if variant == "symmetric" else upper - upper.transpose(1, 2)

    padded = torch.zeros((batch_size, length, channels), device=device)
    padded[:, :length - 2] = local
    padded[:, length - 2:] = local[:, -1:]
    diagonal = torch.arange(length, device=device)
    output[:, diagonal, diagonal] = padded
    adjacent = torch.where(
        padded[:, :-1].abs() >= padded[:, 1:].abs(), padded[:, :-1], padded[:, 1:]
    )
    index = torch.arange(length - 1, device=device)
    output[:, index, index + 1] = adjacent
    output[:, index + 1, index] = adjacent
    return output.cpu().numpy()


def resample(values: np.ndarray, target_length: int) -> np.ndarray:
    if values.shape[1] == target_length:
        return values.astype(np.float32, copy=False)
    result = np.empty((values.shape[0], target_length, values.shape[2]), dtype=np.float32)
    if target_length < values.shape[1]:
        width = values.shape[1] / target_length
        for index in range(target_length):
            start = int(round(index * width))
            stop = min(int(round((index + 1) * width)), values.shape[1])
            result[:, index] = (
                values[:, start:stop].mean(axis=1) if stop > start else values[:, start]
            )
        return result
    old = np.linspace(0.0, 1.0, values.shape[1])
    new = np.linspace(0.0, 1.0, target_length)
    for sample in range(values.shape[0]):
        for channel in range(values.shape[2]):
            result[sample, :, channel] = np.interp(new, old, values[sample, :, channel])
    return result


def load_preprocessed(dataset_dir: Path, mode: str, split: str) -> np.ndarray:
    name = dataset_dir.name
    nested = dataset_dir / "preprocessed" / mode / f"{name}_{split}_df.pkl"
    legacy = dataset_dir / f"{name}_{split}_df.pkl"
    path = nested if nested.is_file() else legacy if mode == "train_minmax_neg1_pos1" else nested
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run preprocess_time_series.py first."
        )
    with path.open("rb") as stream:
        values = np.asarray(pickle.load(stream), dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"Expected [sample,time,channel] in {path}, got {values.shape}")
    return values


def image_type_name(preprocess_mode: str, variant: str) -> str:
    return f"aim_{preprocess_mode.lower()}_{variant}"


def generate_split(
    values: np.ndarray,
    output_path: Path,
    variant: str,
    engine: str,
    device: str,
    batch_size: int,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        print(f"[SKIP] Exists: {output_path}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shape = (values.shape[0], values.shape[1], values.shape[1], values.shape[2])
    output = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=shape)
    for start in range(0, values.shape[0], batch_size):
        stop = min(start + batch_size, values.shape[0])
        if engine == "torch":
            output[start:stop] = _torch_batch_aim(values[start:stop], variant, device)
        else:
            for sample in range(start, stop):
                for channel in range(values.shape[2]):
                    output[sample, :, :, channel] = aim_numpy_single(
                        values[sample, :, channel], variant
                    )
        output.flush()
        print(f"  {output_path.name}: {stop}/{values.shape[0]}", flush=True)
    del output


def process_dataset(
    dataset_dir: Path,
    preprocess_modes: Iterable[str],
    variants: Iterable[str],
    lengths: Iterable[int] | None,
    args: argparse.Namespace,
) -> None:
    for preprocess_mode in preprocess_modes:
        train = load_preprocessed(dataset_dir, preprocess_mode, "train")
        test = load_preprocessed(dataset_dir, preprocess_mode, "test")
        if train.shape[1:] != test.shape[1:]:
            raise ValueError(f"Train/test dimensions differ for {dataset_dir.name}")
        for length in (list(lengths) if lengths else [train.shape[1]]):
            train_length = resample(train, length)
            test_length = resample(test, length)
            for raw_variant in variants:
                variant = normalize_variant(raw_variant)
                image_type = image_type_name(preprocess_mode, variant)
                train_path = dataset_dir / f"{dataset_dir.name}_{image_type}_train_{length}.npy"
                test_path = dataset_dir / f"{dataset_dir.name}_{image_type}_test_{length}.npy"
                generate_split(train_length, train_path, variant, args.engine, args.device, args.batch_size, args.overwrite)
                generate_split(test_length, test_path, variant, args.engine, args.device, args.batch_size, args.overwrite)
                settings = {
                    "dataset": dataset_dir.name,
                    "image_type": image_type,
                    "preprocessing_mode": preprocess_mode,
                    "variant": variant,
                    "sequence_length": length,
                    "output_layout": "[sample, height, width, channel]",
                    "dtype": "float32",
                    "local_formulation": {
                        "original": "sign * exp(abs(cos(A)))",
                        "no_exp": "sign * abs(cos(A))",
                        "no_abs": "sign * exp(cos(A))",
                        "no_exp_no_abs": "sign * cos(A)",
                    }.get(variant, variant),
                    "interval_rule": "mean local value for j-i>=2",
                    "reverse_rule": "symmetric" if variant == "symmetric" else "antisymmetric",
                    "flat_interval_sign": 1 if variant == "flat_as_positive" else 0,
                }
                settings_path = dataset_dir / f"{dataset_dir.name}_{image_type}_settings_{length}.json"
                if args.overwrite or not settings_path.exists():
                    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
                else:
                    print(f"[SKIP] Exists: {settings_path}")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ.get("AIM_DATASET_ROOT", here / "dataset")),
    )
    parser.add_argument("--ts-type", default="Multivariate_ts")
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument(
        "--preprocess-modes",
        nargs="+",
        default=["original"],
        help="Preprocessed inputs. Public original means unscaled native values.",
    )
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--lengths", nargs="+", type=int)
    parser.add_argument("--engine", choices=("torch", "numpy"), default="torch")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.engine == "torch":
        import torch
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            print("[WARN] CUDA is unavailable; using CPU.")
            args.device = "cpu"
    base = args.dataset_root / args.ts_type
    selected = set(args.datasets or [])
    for dataset_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        if selected and dataset_dir.name not in selected:
            continue
        process_dataset(
            dataset_dir,
            args.preprocess_modes,
            args.variants,
            args.lengths,
            args,
        )


if __name__ == "__main__":
    main()
