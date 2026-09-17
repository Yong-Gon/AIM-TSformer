from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot


METHODS = ("gasf", "gadf", "mtf", "rp")


def load_values(dataset_dir: Path, preprocess_mode: str, split: str) -> np.ndarray:
    name = dataset_dir.name
    nested = dataset_dir / "preprocessed" / preprocess_mode / f"{name}_{split}_df.pkl"
    legacy = dataset_dir / f"{name}_{split}_df.pkl"
    path = nested if nested.is_file() else legacy if preprocess_mode == "train_minmax_neg1_pos1" else nested
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run preprocess_time_series.py first."
        )
    with path.open("rb") as stream:
        values = np.asarray(pickle.load(stream), dtype=np.float32)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError(f"Expected finite [sample,time,channel] values in {path}")
    return values


def resample(values: np.ndarray, target_length: int) -> np.ndarray:
    if values.shape[1] == target_length:
        return values
    if target_length < 1:
        raise ValueError("Sequence length must be positive.")
    if target_length < values.shape[1]:
        width = values.shape[1] / target_length
        bins = []
        for index in range(target_length):
            start = int(round(index * width))
            stop = min(int(round((index + 1) * width)), values.shape[1])
            bins.append(values[:, start:stop].mean(axis=1) if stop > start else values[:, start])
        return np.stack(bins, axis=1).astype(np.float32)
    old_time = np.arange(values.shape[1])
    new_time = np.linspace(0, values.shape[1] - 1, target_length)
    result = np.empty((values.shape[0], target_length, values.shape[2]), dtype=np.float32)
    for sample in range(values.shape[0]):
        for channel in range(values.shape[2]):
            result[sample, :, channel] = np.interp(
                new_time, old_time, values[sample, :, channel]
            )
    return result


def make_transformer(method: str, image_size: int):
    if method == "gasf":
        return GramianAngularField(
            image_size=image_size,
            sample_range=(-1, 1),
            method="summation",
            overlapping=False,
            flatten=False,
        )
    if method == "gadf":
        return GramianAngularField(
            image_size=image_size,
            sample_range=(-1, 1),
            method="difference",
            overlapping=False,
            flatten=False,
        )
    if method == "mtf":
        return MarkovTransitionField(
            image_size=image_size,
            n_bins=8,
            strategy="quantile",
            overlapping=False,
            flatten=False,
        )
    if method == "rp":
        return RecurrencePlot(
            dimension=1,
            time_delay=1,
            threshold=None,
            percentage=10,
            flatten=False,
        )
    raise ValueError(f"Unknown method {method!r}; choose from {METHODS}.")


def method_settings(method: str, image_size: int) -> dict:
    common = {
        "method": method,
        "image_size": image_size,
        "channel_handling": "each channel transformed independently",
        "output_layout": "[sample, height, width, channel]",
        "dtype": "float32",
    }
    if method in {"gasf", "gadf"}:
        common.update(
            pyts_class="GramianAngularField",
            sample_range=[-1, 1],
            sample_range_scope="pyts rescales each sample/channel series independently",
            gaf_method="summation" if method == "gasf" else "difference",
            overlapping=False,
            flatten=False,
        )
    elif method == "mtf":
        common.update(
            pyts_class="MarkovTransitionField",
            n_bins=8,
            strategy="quantile",
            quantile_scope="each sample/channel series",
            overlapping=False,
            flatten=False,
        )
    else:
        common.update(
            pyts_class="RecurrencePlot",
            dimension=1,
            time_delay=1,
            threshold=None,
            percentage=10,
            percentage_effect="unused when threshold is None",
            flatten=False,
        )
    return common


def transform_to_file(
    values: np.ndarray,
    method: str,
    output_path: Path,
    chunk_size: int,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        print(f"[SKIP] Exists: {output_path}")
        return
    length = values.shape[1]
    transformer = make_transformer(method, length)
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(values.shape[0], length, length, values.shape[2]),
    )
    for channel in range(values.shape[2]):
        for start in range(0, values.shape[0], chunk_size):
            stop = min(start + chunk_size, values.shape[0])
            output[start:stop, :, :, channel] = transformer.transform(
                values[start:stop, :, channel]
            ).astype(np.float32, copy=False)
        output.flush()
        print(f"  {output_path.name}: channel {channel + 1}/{values.shape[2]}")
    del output


def process_dataset(
    dataset_dir: Path,
    preprocess_mode: str,
    methods: Iterable[str],
    lengths: Iterable[int] | None,
    args: argparse.Namespace,
) -> None:
    train = load_values(dataset_dir, preprocess_mode, "train")
    test = load_values(dataset_dir, preprocess_mode, "test")
    if train.shape[1:] != test.shape[1:]:
        raise ValueError(f"Train/test dimensions differ for {dataset_dir.name}")
    for length in (list(lengths) if lengths else [train.shape[1]]):
        train_length = resample(train, length)
        test_length = resample(test, length)
        for method in methods:
            method = method.lower()
            train_path = dataset_dir / f"{dataset_dir.name}_{method}_train_{length}.npy"
            test_path = dataset_dir / f"{dataset_dir.name}_{method}_test_{length}.npy"
            transform_to_file(train_length, method, train_path, args.chunk_size, args.overwrite)
            transform_to_file(test_length, method, test_path, args.chunk_size, args.overwrite)
            settings = method_settings(method, length)
            settings.update(
                dataset=dataset_dir.name,
                source_preprocessing_mode=preprocess_mode,
                length_adjustment=(
                    "none" if length == train.shape[1]
                    else "PAA" if length < train.shape[1]
                    else "linear interpolation"
                ),
            )
            settings_path = dataset_dir / f"{dataset_dir.name}_{method}_settings_{length}.json"
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
        "--preprocess-mode",
        default="original",
        help="One preprocessing mode per run keeps the public filenames unambiguous.",
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--lengths", nargs="+", type=int)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = args.dataset_root / args.ts_type
    selected = set(args.datasets or [])
    for dataset_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        if selected and dataset_dir.name not in selected:
            continue
        process_dataset(
            dataset_dir,
            args.preprocess_mode,
            args.methods,
            args.lengths,
            args,
        )


if __name__ == "__main__":
    main()
