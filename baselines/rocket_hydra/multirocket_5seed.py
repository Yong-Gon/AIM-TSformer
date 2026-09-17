from pathlib import Path
import os
import time
import traceback
import numpy as np

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from common_sktime_mtsc_utils import (
    DEFAULT_DATASETS,
    get_existing_rows,
    load_uea_numpy3d,
    make_base_result_row,
    remove_same_key,
    row_already_ok,
    save_experiment_tables,
    set_seed,
    format_seconds,
)




BASE_PARENT = Path(os.environ.get(
    "AIM_DATASET_ROOT", str(Path(__file__).resolve().parents[2] / "dataset")
))
TS_TYPE = "Multivariate_ts"
DATASETS = DEFAULT_DATASETS
SEEDS = [42, 43, 44, 45, 46]
OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "MultiRocket"

FIXED_LENGTH = None
UPSAMPLE_METHOD = "linear"
NORMALIZE_BY_TRAIN = True

NUM_KERNELS = 10000
MAX_DILATIONS_PER_KERNEL = 32
N_FEATURES_PER_KERNEL = 4
N_JOBS = int(os.environ.get("AIM_BASELINE_N_JOBS", "-1"))
SKIP_COMPLETED = True


def make_classifier(seed: int):
    from aeon.classification.convolution_based import MultiRocketClassifier

    return MultiRocketClassifier(
        n_kernels=NUM_KERNELS,
        max_dilations_per_kernel=MAX_DILATIONS_PER_KERNEL,
        n_features_per_kernel=N_FEATURES_PER_KERNEL,
        class_weight=None,
        n_jobs=N_JOBS,
        random_state=int(seed),
    )


def run_one(dataset_name: str, seed: int):
    set_seed(seed)
    X_train, y_train, X_test, y_test, label_map, dpath, loader_name = load_uea_numpy3d(
        BASE_PARENT,
        dataset_name,
        ts_type=TS_TYPE,
        fixed_length=FIXED_LENGTH,
        upsample_method=UPSAMPLE_METHOD,
        normalize_by_train=NORMALIZE_BY_TRAIN,
    )

    X_train = np.ascontiguousarray(X_train, dtype=np.float32)
    X_test = np.ascontiguousarray(X_test, dtype=np.float32)

    clf = make_classifier(seed)
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    t1 = time.perf_counter()
    y_pred = clf.predict(X_test)
    t2 = time.perf_counter()

    row = make_base_result_row(
        "MultiRocket", dataset_name, seed, X_train, y_train, X_test, y_test,
        label_map, dpath, loader_name, FIXED_LENGTH, NORMALIZE_BY_TRAIN,
    )
    row.update({
        "final_test_acc": float(accuracy_score(y_test, y_pred)),
        "final_test_macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "final_test_balanced_acc": float(balanced_accuracy_score(y_test, y_pred)),
        "fit_seconds": float(t1 - t0),
        "predict_seconds": float(t2 - t1),
        "total_seconds": float(t2 - t0),
        "num_kernels": NUM_KERNELS,
        "max_dilations_per_kernel": MAX_DILATIONS_PER_KERNEL,
        "n_features_per_kernel": N_FEATURES_PER_KERNEL,
        "n_jobs": N_JOBS,
        "backend": "aeon.classification.convolution_based.MultiRocketClassifier",
        "status": "OK",
        "error": "",
    })
    return row


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = get_existing_rows(OUTPUT_ROOT / "MultiRocket_per_seed_results.csv")

    for dataset_name in DATASETS:
        for seed in SEEDS:
            print("\n" + "=" * 90, flush=True)
            print(f"MultiRocket | dataset={dataset_name} | seed={seed}", flush=True)
            if SKIP_COMPLETED and row_already_ok(rows, "MultiRocket", dataset_name, seed):
                print("[SKIP] Existing OK row found.", flush=True)
                continue
            try:
                row = run_one(dataset_name, seed)
                print(
                    f"[OK] acc={row['final_test_acc']:.6f} | "
                    f"fit={format_seconds(row['fit_seconds'])} | pred={format_seconds(row['predict_seconds'])}",
                    flush=True,
                )
            except Exception as e:
                traceback.print_exc()
                row = {"model": "MultiRocket", "dataset": dataset_name, "seed": int(seed), "status": "ERROR", "error": repr(e)}
            rows = remove_same_key(rows, "MultiRocket", dataset_name, seed)
            rows.append(row)
            save_experiment_tables(rows, OUTPUT_ROOT, "MultiRocket")

    print("\nDone. Use MultiRocket_summary_mean_std.csv for the paper table.", flush=True)


if __name__ == "__main__":
    main()
