from pathlib import Path
import os
import time
import traceback

import numpy as np
from numba import njit, prange
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from rocket_hydra.common_sktime_mtsc_utils import (
    DEFAULT_DATASETS,
    format_seconds,
    get_existing_rows,
    load_uea_numpy3d,
    make_base_result_row,
    remove_same_key,
    row_already_ok,
    save_experiment_tables,
    set_seed,
)

BASE_PARENT = Path(os.environ.get(
    "AIM_DATASET_ROOT", str(Path(__file__).resolve().parent.parent / "dataset")
))
TS_TYPE = "Multivariate_ts"
DATASETS = DEFAULT_DATASETS
SEEDS = [42, 43, 44, 45, 46]
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs" / "DTW_I"

NORMALIZE_BY_TRAIN = True
REUSE_DETERMINISTIC_RESULT_ACROSS_SEEDS = True
SKIP_COMPLETED = True
SKIP_DATASETS = {
    "FaceDetection": "Retained as N/A: full independent DTW is computationally prohibitive.",
}


@njit(cache=False)
def _dtw_squared_1d(a: np.ndarray, b: np.ndarray) -> float:
    m = b.shape[0]
    previous = np.full(m + 1, np.inf, dtype=np.float64)
    current = np.full(m + 1, np.inf, dtype=np.float64)
    previous[0] = 0.0
    for i in range(1, a.shape[0] + 1):
        current[0] = np.inf
        for j in range(1, m + 1):
            delta = float(a[i - 1] - b[j - 1])
            best = previous[j]
            if current[j - 1] < best:
                best = current[j - 1]
            if previous[j - 1] < best:
                best = previous[j - 1]
            current[j] = delta * delta + best
        swap = previous
        previous = current
        current = swap
    return previous[m]


@njit(cache=False, parallel=True)
def _predict_independent_dtw(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
) -> np.ndarray:
    predictions = np.empty(X_test.shape[0], dtype=y_train.dtype)
    for test_index in prange(X_test.shape[0]):
        best_distance = np.inf
        best_label = y_train[0]
        for train_index in range(X_train.shape[0]):
            distance = 0.0
            for channel in range(X_train.shape[1]):
                distance += _dtw_squared_1d(
                    X_test[test_index, channel], X_train[train_index, channel]
                )
                if distance >= best_distance:
                    break
            if distance < best_distance:
                best_distance = distance
                best_label = y_train[train_index]
        predictions[test_index] = best_label
    return predictions


def evaluate_dataset(dataset_name: str) -> tuple[dict, np.ndarray, np.ndarray]:
    set_seed(SEEDS[0])
    X_train, y_train, X_test, y_test, label_map, dpath, loader_name = load_uea_numpy3d(
        BASE_PARENT,
        dataset_name,
        ts_type=TS_TYPE,
        fixed_length=None,
        normalize_by_train=NORMALIZE_BY_TRAIN,
    )
    started = time.perf_counter()
    y_pred = _predict_independent_dtw(
        np.asarray(X_train, dtype=np.float32),
        np.asarray(y_train),
        np.asarray(X_test, dtype=np.float32),
    )
    elapsed = time.perf_counter() - started
    base = make_base_result_row(
        "DTW_I", dataset_name, SEEDS[0], X_train, y_train, X_test, y_test,
        label_map, dpath, loader_name, None, NORMALIZE_BY_TRAIN,
    )
    base.update({
        "final_test_acc": float(accuracy_score(y_test, y_pred)),
        "final_test_macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "final_test_balanced_acc": float(balanced_accuracy_score(y_test, y_pred)),
        "fit_seconds": 0.0,
        "predict_seconds": float(elapsed),
        "total_seconds": float(elapsed),
        "n_neighbors": 1,
        "distance": "independent_dimension_full_window_dtw",
        "epochs": "not_applicable",
        "status": "OK",
        "error": "",
    })
    return base, y_test, y_pred


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = get_existing_rows(OUTPUT_ROOT / "DTW_I_per_seed_results.csv")
    for dataset_name in DATASETS:
        if dataset_name in SKIP_DATASETS:
            for seed in SEEDS:
                if row_already_ok(rows, "DTW_I", dataset_name, seed):
                    continue
                rows = remove_same_key(rows, "DTW_I", dataset_name, seed)
                rows.append({
                    "model": "DTW_I", "dataset": dataset_name, "seed": seed,
                    "epochs": "not_applicable", "status": "SKIPPED",
                    "error": SKIP_DATASETS[dataset_name],
                })
            save_experiment_tables(rows, OUTPUT_ROOT, "DTW_I")
            continue

        if all(row_already_ok(rows, "DTW_I", dataset_name, seed) for seed in SEEDS):
            print(f"[SKIP] DTW_I {dataset_name}: all seed rows exist.", flush=True)
            continue
        print(f"\nDTW_I | dataset={dataset_name} | deterministic single computation", flush=True)
        try:
            base, _, _ = evaluate_dataset(dataset_name)
            seeds_to_write = SEEDS if REUSE_DETERMINISTIC_RESULT_ACROSS_SEEDS else [SEEDS[0]]
            for seed in seeds_to_write:
                row = dict(base)
                row["seed"] = int(seed)
                row["deterministic_result_reused"] = seed != SEEDS[0]
                rows = remove_same_key(rows, "DTW_I", dataset_name, seed)
                rows.append(row)
            print(
                f"[OK] acc={base['final_test_acc']:.6f} | "
                f"macro_f1={base['final_test_macro_f1']:.6f} | "
                f"bacc={base['final_test_balanced_acc']:.6f} | "
                f"time={format_seconds(base['total_seconds'])}",
                flush=True,
            )
        except Exception as error:
            traceback.print_exc()
            for seed in SEEDS:
                rows = remove_same_key(rows, "DTW_I", dataset_name, seed)
                rows.append({
                    "model": "DTW_I", "dataset": dataset_name, "seed": seed,
                    "epochs": "not_applicable", "status": "ERROR", "error": repr(error),
                })
        save_experiment_tables(rows, OUTPUT_ROOT, "DTW_I")


if __name__ == "__main__":
    main()
