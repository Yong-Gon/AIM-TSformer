from pathlib import Path
import os
import time
import traceback

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sktime.classification.interval_based import TimeSeriesForestClassifier

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
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs" / "TSF"

N_ESTIMATORS = 200
N_JOBS = int(os.environ.get("AIM_BASELINE_N_JOBS", "-1"))
NORMALIZE_BY_TRAIN = True
SKIP_COMPLETED = True


def run_one(dataset_name: str, seed: int) -> dict:
    set_seed(seed)
    X_train, y_train, X_test, y_test, label_map, dpath, loader_name = load_uea_numpy3d(
        BASE_PARENT,
        dataset_name,
        ts_type=TS_TYPE,
        fixed_length=None,
        normalize_by_train=NORMALIZE_BY_TRAIN,
    )
    original_channels = int(X_train.shape[1])
    original_length = int(X_train.shape[2])
    X_train_tsf = X_train.reshape(X_train.shape[0], 1, -1)
    X_test_tsf = X_test.reshape(X_test.shape[0], 1, -1)
    classifier = TimeSeriesForestClassifier(
        n_estimators=N_ESTIMATORS,
        n_jobs=N_JOBS,
        random_state=int(seed),
    )
    started = time.perf_counter()
    classifier.fit(X_train_tsf, y_train)
    fitted = time.perf_counter()
    y_pred = classifier.predict(X_test_tsf)
    finished = time.perf_counter()

    row = make_base_result_row(
        "TSF", dataset_name, seed, X_train, y_train, X_test, y_test,
        label_map, dpath, loader_name, None, NORMALIZE_BY_TRAIN,
    )
    row.update({
        "final_test_acc": float(accuracy_score(y_test, y_pred)),
        "final_test_macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "final_test_balanced_acc": float(balanced_accuracy_score(y_test, y_pred)),
        "fit_seconds": float(fitted - started),
        "predict_seconds": float(finished - fitted),
        "total_seconds": float(finished - started),
        "n_estimators": N_ESTIMATORS,
        "n_jobs": N_JOBS,
        "multivariate_adapter": "channel_major_concatenation",
        "original_channels": original_channels,
        "original_length": original_length,
        "adapted_univariate_length": int(X_train_tsf.shape[2]),
        "implementation_note": "Reconstructed runner; original TSF source was absent.",
        "epochs": "not_applicable",
        "status": "OK",
        "error": "",
    })
    return row


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = get_existing_rows(OUTPUT_ROOT / "TSF_per_seed_results.csv")
    for dataset_name in DATASETS:
        for seed in SEEDS:
            print(f"\nTSF | dataset={dataset_name} | seed={seed}", flush=True)
            if SKIP_COMPLETED and row_already_ok(rows, "TSF", dataset_name, seed):
                print("[SKIP] Existing OK row with all three metrics.", flush=True)
                continue
            try:
                row = run_one(dataset_name, seed)
                print(
                    f"[OK] acc={row['final_test_acc']:.6f} | "
                    f"macro_f1={row['final_test_macro_f1']:.6f} | "
                    f"bacc={row['final_test_balanced_acc']:.6f} | "
                    f"time={format_seconds(row['total_seconds'])}",
                    flush=True,
                )
            except Exception as error:
                traceback.print_exc()
                row = {
                    "model": "TSF", "dataset": dataset_name, "seed": int(seed),
                    "status": "ERROR", "error": repr(error),
                }
            rows = remove_same_key(rows, "TSF", dataset_name, seed)
            rows.append(row)
            save_experiment_tables(rows, OUTPUT_ROOT, "TSF")


if __name__ == "__main__":
    main()
