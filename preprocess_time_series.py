from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sktime.datasets import load_from_tsfile_to_dataframe


MODES = (
    "original",
    "train_minmax_neg1_pos1",
    "amplitude_0p5",
    "amplitude_2p0",
    "time_0p5T",
    "time_2p0T",
)
ALIASES = {
    "paper_native": "original",
    "existing": "train_minmax_neg1_pos1",
    "paper_native_amp_0p5": "amplitude_0p5",
    "paper_native_amp_2p0": "amplitude_2p0",
    "paper_native_time_0p5T": "time_0p5T",
    "paper_native_time_2p0T": "time_2p0T",
}


def dataframe_to_dense(frame) -> np.ndarray:
    n_samples, n_channels = frame.shape
    lengths = {
        len(np.asarray(frame.iloc[row, channel]))
        for row in range(n_samples)
        for channel in range(n_channels)
    }
    if len(lengths) != 1:
        raise ValueError(
            "AIM requires equal-length series, but the dataset contains "
            f"lengths {sorted(lengths)}. Resample to a common length first."
        )
    length = lengths.pop()
    values = np.empty((n_samples, length, n_channels), dtype=np.float32)
    for row in range(n_samples):
        for channel in range(n_channels):
            values[row, :, channel] = np.asarray(
                frame.iloc[row, channel], dtype=np.float32
            )
    if not np.isfinite(values).all():
        raise ValueError("The input contains NaN or infinite values.")
    return values


def load_native_split(dataset_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    name = dataset_dir.name
    train_path = dataset_dir / f"{name}_TRAIN.ts"
    test_path = dataset_dir / f"{name}_TEST.ts"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(f"Missing TRAIN/TEST .ts files in {dataset_dir}")
    train_frame, _ = load_from_tsfile_to_dataframe(str(train_path))
    test_frame, _ = load_from_tsfile_to_dataframe(str(test_path))
    return dataframe_to_dense(train_frame), dataframe_to_dense(test_frame)


def train_fitted_channel_minmax(
    train: np.ndarray, test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict[str, list]]:
    train_out = np.empty_like(train, dtype=np.float32)
    test_out = np.empty_like(test, dtype=np.float32)
    data_min, data_max = [], []
    for channel in range(train.shape[2]):
        scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
        scaler.fit(train[:, :, channel].reshape(-1, 1))
        train_out[:, :, channel] = scaler.transform(
            train[:, :, channel].reshape(-1, 1)
        ).reshape(train.shape[:2])
        test_out[:, :, channel] = scaler.transform(
            test[:, :, channel].reshape(-1, 1)
        ).reshape(test.shape[:2])
        data_min.append(float(scaler.data_min_[0]))
        data_max.append(float(scaler.data_max_[0]))
    return train_out, test_out, {"train_channel_min": data_min, "train_channel_max": data_max}


def paa(values: np.ndarray, target_length: int) -> np.ndarray:
    if target_length < 1 or target_length > values.shape[1]:
        raise ValueError("PAA target_length must be in [1, native_length].")
    result = np.empty((values.shape[0], target_length, values.shape[2]), dtype=np.float32)
    width = values.shape[1] / target_length
    for index in range(target_length):
        start = int(round(index * width))
        stop = min(int(round((index + 1) * width)), values.shape[1])
        result[:, index] = (
            values[:, start:stop].mean(axis=1) if stop > start else values[:, start]
        )
    return result


def linear_resample(values: np.ndarray, target_length: int) -> np.ndarray:
    native_length = values.shape[1]
    old_time = np.linspace(0.0, 1.0, native_length, dtype=np.float64)
    new_time = np.linspace(0.0, 1.0, target_length, dtype=np.float64)
    result = np.empty((values.shape[0], target_length, values.shape[2]), dtype=np.float32)
    for sample in range(values.shape[0]):
        for channel in range(values.shape[2]):
            result[sample, :, channel] = np.interp(
                new_time, old_time, values[sample, :, channel]
            )
    return result


def transform_mode(
    train_native: np.ndarray, test_native: np.ndarray, mode: str
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    mode = ALIASES.get(mode, mode)
    metadata: Dict[str, object] = {
        "mode": mode,
        "fit_scope": "none",
        "value_domain": "native",
        "temporal_transform": "none",
    }
    if mode == "original":
        train, test = train_native.copy(), test_native.copy()
    elif mode == "train_minmax_neg1_pos1":
        train, test, fitted = train_fitted_channel_minmax(train_native, test_native)
        metadata.update(
            fit_scope="training split only, independently per channel",
            value_domain="MinMaxScaler(feature_range=(-1, 1)); test may exceed the range",
            **fitted,
        )
    elif mode == "amplitude_0p5":
        train, test = train_native * 0.5, test_native * 0.5
        metadata["amplitude_multiplier"] = 0.5
    elif mode == "amplitude_2p0":
        train, test = train_native * 2.0, test_native * 2.0
        metadata["amplitude_multiplier"] = 2.0
    elif mode == "time_0p5T":
        target = max(1, train_native.shape[1] // 2)
        train, test = paa(train_native, target), paa(test_native, target)
        metadata.update(temporal_transform="PAA", temporal_factor=0.5)
    elif mode == "time_2p0T":
        target = train_native.shape[1] * 2
        train = linear_resample(train_native, target)
        test = linear_resample(test_native, target)
        metadata.update(temporal_transform="linear interpolation", temporal_factor=2.0)
    else:
        raise ValueError(f"Unknown mode {mode!r}; choose from {MODES}.")
    return train.astype(np.float32), test.astype(np.float32), metadata


def process_dataset(dataset_dir: Path, modes: Iterable[str], overwrite: bool = False) -> None:
    train_native, test_native = load_native_split(dataset_dir)
    if train_native.shape[1:] != test_native.shape[1:]:
        raise ValueError(
            f"Train/test dimensions differ for {dataset_dir.name}: "
            f"{train_native.shape[1:]} vs {test_native.shape[1:]}"
        )
    for requested_mode in modes:
        mode = ALIASES.get(requested_mode, requested_mode)
        train, test, metadata = transform_mode(train_native, test_native, mode)
        output_dir = dataset_dir / "preprocessed" / mode
        output_dir.mkdir(parents=True, exist_ok=True)
        train_path = output_dir / f"{dataset_dir.name}_train_df.pkl"
        test_path = output_dir / f"{dataset_dir.name}_test_df.pkl"
        metadata_path = output_dir / "preprocessing_settings.json"
        outputs = (train_path, test_path, metadata_path)
        existing = [path for path in outputs if path.exists()]
        if len(existing) == len(outputs) and not overwrite:
            print(f"[SKIP] {dataset_dir.name}/{mode}: output exists (use --overwrite)")
            continue
        if existing and not overwrite:
            raise FileExistsError(
                f"Partial output exists for {dataset_dir.name}/{mode}: {existing}. "
                "Inspect it, then rerun with --overwrite."
            )
        with train_path.open("wb") as stream:
            pickle.dump(train, stream, protocol=pickle.HIGHEST_PROTOCOL)
        with test_path.open("wb") as stream:
            pickle.dump(test, stream, protocol=pickle.HIGHEST_PROTOCOL)
        metadata.update(
            dataset=dataset_dir.name,
            native_train_shape=list(train_native.shape),
            native_test_shape=list(test_native.shape),
            output_train_shape=list(train.shape),
            output_test_shape=list(test.shape),
            output_dtype="float32",
            test_statistics_used_for_fit=False,
        )
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"[OK] {dataset_dir.name}/{mode}: train={train.shape}, test={test.shape}")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ.get("AIM_DATASET_ROOT", here / "dataset")),
        help="Directory containing Multivariate_ts and/or Univariate_ts.",
    )
    parser.add_argument("--ts-types", nargs="+", default=["Multivariate_ts"])
    parser.add_argument("--datasets", nargs="*", help="Dataset folder names; omit for all.")
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=(*MODES, *ALIASES))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for ts_type in args.ts_types:
        base = args.dataset_root / ts_type
        if not base.is_dir():
            print(f"[SKIP] Missing directory: {base}")
            continue
        selected = set(args.datasets or [])
        dataset_dirs = sorted(path for path in base.iterdir() if path.is_dir())
        for dataset_dir in dataset_dirs:
            if selected and dataset_dir.name not in selected:
                continue
            process_dataset(dataset_dir, args.modes, args.overwrite)


if __name__ == "__main__":
    main()
