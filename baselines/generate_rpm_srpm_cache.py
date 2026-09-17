from __future__ import annotations

import argparse
import math
import os
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from numpy.lib.stride_tricks import sliding_window_view


PUBLIC_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = Path(
    os.environ.get("AIM_DATASET_ROOT", str(PUBLIC_ROOT / "dataset"))
) / "Multivariate_ts"
WINDOW = 2


def load_pickle_array(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        array = pickle.load(handle)
    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError(f"Expected (N,L,F), got {array.shape} from {path}")
    return array


def rpm_image(sample: np.ndarray) -> np.ndarray:
    sample = np.asarray(sample, dtype=np.float32)
    return (sample[None, :, :] - sample[:, None, :]).astype(np.float32, copy=False)


def rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    width = 2 * window + 1
    windows = sliding_window_view(np.asarray(values), width, axis=0)
    return np.std(windows, axis=-1)


def srpm_image(sample: np.ndarray, window: int = WINDOW) -> np.ndarray:
    sample = np.asarray(sample, dtype=np.float32)
    trend = rolling_std(rolling_std(sample, window), window)
    half = int(trend.shape[0] // 2)
    if half < 1:
        raise ValueError(
            f"SRPM requires original length > {4 * window + 1}; got {sample.shape[0]}"
        )
    index = np.arange(half, dtype=np.int64)
    sums = index[:, None] + index[None, :]
    return (trend[sums, :] - trend[index, None, :]).astype(np.float32, copy=False)


def output_shape(transform: str, data_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    n, length, channels = map(int, data_shape)
    if transform == "rpm":
        size = length
    elif transform == "srpm":
        size = (length - 4 * WINDOW) // 2
        if size < 1:
            raise ValueError(f"SRPM is undefined for sequence length {length}")
    else:
        raise ValueError(transform)
    return n, size, size, channels


def output_path(dataset_dir: Path, dataset: str, split: str, transform: str, length: int) -> Path:
    if transform == "rpm":
        name = f"{dataset}_rpm_{split}_{length}.npy"
    else:
        name = f"{dataset}_srpm_{split}_win{WINDOW}_len{length}.npy"
    return dataset_dir / name


def stats_path(train_path: Path) -> Path:
    return train_path.with_suffix(".stats.npz")


def array_is_complete(path: Path, expected_shape: tuple[int, ...]) -> bool:
    if not path.exists():
        return False
    try:
        array = np.load(path, mmap_mode="r")
        return tuple(array.shape) == tuple(expected_shape) and array.dtype == np.float32
    except Exception:
        return False


def scan_channel_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    array = np.load(path, mmap_mode="r")
    channels = int(array.shape[-1])
    mins = np.full(channels, np.inf, dtype=np.float32)
    maxs = np.full(channels, -np.inf, dtype=np.float32)
    for index in range(int(array.shape[0])):
        sample = np.asarray(array[index], dtype=np.float32)
        with np.errstate(all="ignore"):
            mins = np.minimum(mins, np.nanmin(sample, axis=(0, 1)).astype(np.float32))
            maxs = np.maximum(maxs, np.nanmax(sample, axis=(0, 1)).astype(np.float32))
    return mins, maxs


def save_stats_atomic(path: Path, mins: np.ndarray, maxs: np.ndarray) -> None:
    temp = path.with_name(path.stem + ".partial.npz")
    np.savez(temp, mins=np.asarray(mins, dtype=np.float32), maxs=np.asarray(maxs, dtype=np.float32))
    os.replace(temp, path)


def generate_split(
    data: np.ndarray,
    target: Path,
    transform: str,
    overwrite: bool = False,
    track_stats: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    expected_shape = output_shape(transform, tuple(data.shape))
    if not overwrite and array_is_complete(target, expected_shape):
        print(f"[CACHE HIT] {target} | shape={expected_shape}", flush=True)
        if track_stats:
            sidecar = stats_path(target)
            if sidecar.exists():
                with np.load(sidecar, allow_pickle=False) as saved:
                    return saved["mins"].astype(np.float32), saved["maxs"].astype(np.float32)
            print(f"[STATS] scanning existing cache {target.name}", flush=True)
            return scan_channel_stats(target)
        return None, None

    required = int(np.prod(expected_shape, dtype=np.int64)) * np.dtype(np.float32).itemsize
    free = shutil.disk_usage(target.parent).free
    reserve = 8 * 1024**3
    if free < required + reserve:
        raise OSError(
            f"Insufficient disk for {target.name}: need {required / 2**30:.2f} GiB "
            f"plus 8 GiB reserve, free={free / 2**30:.2f} GiB"
        )

    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        partial.unlink()
    print(
        f"[GENERATE] {transform.upper()} {target.name} | shape={expected_shape} | "
        f"size={required / 2**30:.2f} GiB",
        flush=True,
    )
    mapped = open_memmap(partial, mode="w+", dtype=np.float32, shape=expected_shape)
    channels = int(expected_shape[-1])
    mins = np.full(channels, np.inf, dtype=np.float32) if track_stats else None
    maxs = np.full(channels, -np.inf, dtype=np.float32) if track_stats else None
    started = time.perf_counter()
    for index in range(int(data.shape[0])):
        image = rpm_image(data[index]) if transform == "rpm" else srpm_image(data[index], WINDOW)
        mapped[index] = image
        if track_stats:
            with np.errstate(all="ignore"):
                mins = np.minimum(mins, np.nanmin(image, axis=(0, 1)).astype(np.float32))
                maxs = np.maximum(maxs, np.nanmax(image, axis=(0, 1)).astype(np.float32))
        if index == 0 or (index + 1) % 100 == 0 or index + 1 == data.shape[0]:
            print(
                f"[PROGRESS] {target.name} {index + 1}/{data.shape[0]} | "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    mapped.flush()
    del mapped
    os.replace(partial, target)
    return mins, maxs


def generate_dataset(dataset_root: Path, dataset: str, transform: str, overwrite: bool = False) -> list[Path]:
    dataset_dir = dataset_root / dataset
    train_path = dataset_dir / f"{dataset}_train_df.pkl"
    test_path = dataset_dir / f"{dataset}_test_df.pkl"
    train = load_pickle_array(train_path)
    test = load_pickle_array(test_path)
    if train.shape[1:] != test.shape[1:]:
        raise ValueError(f"Train/test shape mismatch: {train.shape}, {test.shape}")
    length = int(train.shape[1])
    train_out = output_path(dataset_dir, dataset, "train", transform, length)
    test_out = output_path(dataset_dir, dataset, "test", transform, length)
    mins, maxs = generate_split(train, train_out, transform, overwrite=overwrite, track_stats=True)
    if mins is not None and maxs is not None:
        save_stats_atomic(stats_path(train_out), mins, maxs)
    generate_split(test, test_out, transform, overwrite=overwrite, track_stats=False)
    print(f"[DONE] {dataset} {transform.upper()} cache", flush=True)
    return [train_out, test_out, stats_path(train_out)]


def self_test() -> None:
    rng = np.random.default_rng(20260909)
    sample = rng.normal(size=(17, 3)).astype(np.float32)
    expected_rpm = np.zeros((17, 17, 3), dtype=np.float32)
    for channel in range(3):
        values = sample[:, channel]
        expected_rpm[:, :, channel] = values[None, :] - values[:, None]
    np.testing.assert_array_equal(rpm_image(sample), expected_rpm)

    trends = []
    for channel in range(3):
        values = sample[:, channel]
        first = np.array(
            [np.std(values[i - WINDOW:i + WINDOW + 1]) for i in range(WINDOW, len(values) - WINDOW)]
        )
        second = np.array(
            [np.std(first[i - WINDOW:i + WINDOW + 1]) for i in range(WINDOW, len(first) - WINDOW)]
        )
        trends.append(second)
    trend = np.stack(trends, axis=1)
    half = trend.shape[0] // 2
    expected_srpm = np.zeros((half, half, 3), dtype=np.float32)
    for channel in range(3):
        for i in range(half):
            for j in range(half):
                expected_srpm[i, j, channel] = trend[i + j, channel] - trend[i, channel]
    np.testing.assert_allclose(srpm_image(sample), expected_srpm, rtol=1e-6, atol=1e-7)
    print("[SELF TEST] exact RPM and SRPM formulas verified", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--transform", choices=("rpm", "srpm"), required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    generate_dataset(args.dataset_root, args.dataset, args.transform, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
