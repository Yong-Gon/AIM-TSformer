from pathlib import Path
import os
import sys
import traceback
import shutil
from datetime import datetime
import gc

import pandas as pd
import torch

from common_mtsc_utils import (
    DEFAULT_DATASETS, ArrayDataset, adjust_sequence_length, cuda_is_available, get_device, load_uea_train_test,
    load_existing_rows, make_loader, row_already_ok, save_experiment_tables,
    set_seed, should_retry_cuda_oom, train_fixed_epochs, z_normalize_by_train,
)




BASE_PARENT = Path(os.environ.get(
    "AIM_DATASET_ROOT", str(Path(__file__).resolve().parent.parent / "dataset")
))
TS_TYPE = "Multivariate_ts"

START_DATASET = DEFAULT_DATASETS[0]
DATASETS = DEFAULT_DATASETS

SEEDS = [42, 43, 44, 45, 46]
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs" / "TimesNet"

SCRIPT_DIR = Path(__file__).resolve().parent
TSLIB_DIR = SCRIPT_DIR / "Time-Series-Library-main"

BATCH_SIZE = 32
EPOCHS = 300
LR = 1e-3
WEIGHT_DECAY = 0.0
PRINT_EVERY = 1
FIXED_LENGTH = None
UPSAMPLE_METHOD = "linear"


D_MODEL = 32
D_FF = 64
E_LAYERS = 2
TOP_K = 3
NUM_KERNELS = 6
DROPOUT = 0.5
N_HEADS = 8


if not TSLIB_DIR.exists():
    raise FileNotFoundError(
        f"Time-Series-Library-main folder not found: {TSLIB_DIR}\n"
        "Place it next to this script or edit TSLIB_DIR."
    )
sys.path.append(str(TSLIB_DIR))
from models import TimesNet  
TimesNetModel = TimesNet.Model


class TimesNetConfig:
    def __init__(self, seq_len: int, enc_in: int, num_class: int):
        self.task_name = "classification"
        self.seq_len = seq_len
        self.label_len = 0
        self.pred_len = 0
        self.enc_in = enc_in
        self.c_out = enc_in
        self.num_class = num_class
        self.e_layers = E_LAYERS
        self.d_model = D_MODEL
        self.d_ff = D_FF
        self.top_k = TOP_K
        self.num_kernels = NUM_KERNELS
        self.embed = "timeF"
        self.freq = "h"
        self.dropout = DROPOUT
        self.n_heads = N_HEADS
        self.d_layers = 1
        self.factor = 1
        self.distil = False
        self.channels = enc_in


def timesnet_forward(model, xb):
    mask = torch.ones(xb.size(0), xb.size(1), device=xb.device, dtype=xb.dtype)
    return model(xb, mask, None, None)


def run_one(dataset_name: str, seed: int):
    set_seed(seed)
    device = get_device()
    X_train, y_train, X_test, y_test, label_map, dpath = load_uea_train_test(BASE_PARENT, dataset_name, ts_type=TS_TYPE)
    X_train = adjust_sequence_length(X_train, FIXED_LENGTH, upsample_method=UPSAMPLE_METHOD)
    X_test = adjust_sequence_length(X_test, FIXED_LENGTH, upsample_method=UPSAMPLE_METHOD)
    X_train, X_test = z_normalize_by_train(X_train, X_test)

    num_classes = int(y_train.max()) + 1
    seq_len = int(X_train.shape[1])
    enc_in = int(X_train.shape[2])
    train_loader = make_loader(ArrayDataset(X_train, y_train), BATCH_SIZE, True, seed)
    test_loader = make_loader(ArrayDataset(X_test, y_test), BATCH_SIZE, False, seed)

    cfg = TimesNetConfig(seq_len=seq_len, enc_in=enc_in, num_class=num_classes)
    model = TimesNetModel(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    history = train_fixed_epochs(
        model, train_loader, test_loader, device, EPOCHS, optimizer,
        forward_fn=timesnet_forward,
        print_every=PRINT_EVERY, tag=f"[TimesNet][{dataset_name}][seed={seed}]"
    )
    final = history[-1]
    return {
        "model": "TimesNet", "dataset": dataset_name, "seed": seed,
        "seq_len": seq_len, "channels": enc_in,
        "num_classes": num_classes, "final_test_acc": float(final["test_acc"]),
        "final_test_macro_f1": float(final["test_macro_f1"]),
        "final_test_balanced_acc": float(final["test_balanced_acc"]),
        "final_train_acc": float(final["train_acc"]), "final_train_loss": float(final["train_loss"]),
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
        "d_model": D_MODEL, "d_ff": D_FF, "e_layers": E_LAYERS,
        "status": "OK", "error": "",
    }


def backup_existing_result_files():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for name in [
        "TimesNet_per_seed_results.csv",
        "TimesNet_summary_mean_std.csv",
    ]:
        src = OUTPUT_ROOT / name
        if src.exists():
            dst = OUTPUT_ROOT / f"{src.stem}_backup_{timestamp}{src.suffix}"
            shutil.copy2(src, dst)
            print(f"[BACKUP] {src} -> {dst}", flush=True)


def load_existing_rows_before_target_datasets(target_datasets):







    per_seed_csv = OUTPUT_ROOT / "TimesNet_per_seed_results.csv"

    if not per_seed_csv.exists():
        print("[RESUME] Existing per-seed CSV not found. Starting from empty rows.", flush=True)
        return []

    old_df = pd.read_csv(per_seed_csv)

    if "dataset" not in old_df.columns:
        print("[RESUME] Existing CSV has no dataset column. Starting from empty rows.", flush=True)
        return []

    keep_df = old_df[~old_df["dataset"].isin(target_datasets)].copy()

    print(
        f"[RESUME] Loaded existing rows: {len(old_df)} | "
        f"kept rows before {target_datasets[0]}: {len(keep_df)} | "
        f"removed rows to rerun: {len(old_df) - len(keep_df)}",
        flush=True
    )

    return keep_df.to_dict("records")


def remove_same_key(rows, model_name, dataset_name, seed):



    new_rows = []
    for r in rows:
        same_model = str(r.get("model", "")) == str(model_name)
        same_dataset = str(r.get("dataset", "")) == str(dataset_name)

        try:
            same_seed = int(r.get("seed")) == int(seed)
        except Exception:
            same_seed = str(r.get("seed", "")) == str(seed)

        if same_model and same_dataset and same_seed:
            continue

        new_rows.append(r)

    return new_rows


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("[RUN MODE] TimesNet rerun from target dataset")
    print(f"START_DATASET = {START_DATASET}")
    print(f"DATASETS = {DATASETS}")
    print(f"SEEDS = {SEEDS}")
    print("=" * 80, flush=True)

    rows = load_existing_rows(OUTPUT_ROOT / "TimesNet_per_seed_results.csv")

    for dataset_name in DATASETS:
        for seed in SEEDS:
            print("\n" + "=" * 80)
            print(f"TimesNet | dataset={dataset_name} | seed={seed}", flush=True)

            if row_already_ok(rows, "TimesNet", dataset_name, seed):
                print("[SKIP] Existing OK row with all three metrics.", flush=True)
                continue

            try:
                row = run_one(dataset_name, seed)

            except Exception as e:
                if should_retry_cuda_oom(e):
                    torch.cuda.empty_cache()
                    raise
                traceback.print_exc()
                row = {
                    "model": "TimesNet",
                    "dataset": dataset_name,
                    "seed": seed,
                    "status": "ERROR",
                    "error": repr(e),
                }

            
            rows = remove_same_key(rows, "TimesNet", dataset_name, seed)
            rows.append(row)

            
            save_experiment_tables(rows, OUTPUT_ROOT, "TimesNet", group_cols=("dataset",))

            if cuda_is_available():
                torch.cuda.empty_cache()
            gc.collect()

    print("\nDone. Use TimesNet_summary_mean_std.csv for the paper table.", flush=True)


if __name__ == "__main__":
    main()
