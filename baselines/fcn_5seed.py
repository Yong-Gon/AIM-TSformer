from pathlib import Path
import os
import traceback

import torch
import torch.nn as nn

from common_mtsc_utils import (
    DEFAULT_DATASETS, ArrayDataset, adjust_sequence_length, get_device, load_uea_train_test,
    load_existing_rows, make_loader, remove_same_key, row_already_ok,
    save_experiment_tables, set_seed, train_fixed_epochs, z_normalize_by_train,
)




BASE_PARENT = Path(os.environ.get(
    "AIM_DATASET_ROOT", str(Path(__file__).resolve().parent.parent / "dataset")
))
TS_TYPE = "Multivariate_ts"
DATASETS = DEFAULT_DATASETS
SEEDS = [42, 43, 44, 45, 46]
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs" / "FCN"

BATCH_SIZE = 32
EPOCHS = 300
LR = 1e-3
WEIGHT_DECAY = 0.0
PRINT_EVERY = 1
FIXED_LENGTH = None
UPSAMPLE_METHOD = "linear"


class BaselineFCN(nn.Module):
    def __init__(self, input_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 128, kernel_size=8, padding="same")
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding="same")
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding="same")
        self.bn3 = nn.BatchNorm1d(128)
        self.act = nn.ReLU()
        self.fc = nn.Linear(128, num_classes)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = x.mean(dim=-1)
        return self.fc(x)


def run_one(dataset_name: str, seed: int):
    set_seed(seed)
    device = get_device()
    X_train, y_train, X_test, y_test, label_map, dpath = load_uea_train_test(BASE_PARENT, dataset_name, ts_type=TS_TYPE)
    X_train = adjust_sequence_length(X_train, FIXED_LENGTH, upsample_method=UPSAMPLE_METHOD)
    X_test = adjust_sequence_length(X_test, FIXED_LENGTH, upsample_method=UPSAMPLE_METHOD)
    X_train, X_test = z_normalize_by_train(X_train, X_test)

    num_classes = int(y_train.max()) + 1
    train_loader = make_loader(ArrayDataset(X_train, y_train), BATCH_SIZE, True, seed)
    test_loader = make_loader(ArrayDataset(X_test, y_test), BATCH_SIZE, False, seed)

    model = BaselineFCN(input_channels=int(X_train.shape[2]), num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    history = train_fixed_epochs(
        model, train_loader, test_loader, device, EPOCHS, optimizer,
        print_every=PRINT_EVERY, tag=f"[FCN][{dataset_name}][seed={seed}]"
    )
    final = history[-1]
    return {
        "model": "FCN", "dataset": dataset_name, "seed": seed,
        "seq_len": int(X_train.shape[1]), "channels": int(X_train.shape[2]),
        "num_classes": num_classes, "final_test_acc": float(final["test_acc"]),
        "final_test_macro_f1": float(final["test_macro_f1"]),
        "final_test_balanced_acc": float(final["test_balanced_acc"]),
        "final_train_acc": float(final["train_acc"]), "final_train_loss": float(final["train_loss"]),
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "status": "OK", "error": "",
    }


def main():
    rows = load_existing_rows(OUTPUT_ROOT / "FCN_per_seed_results.csv")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for dataset_name in DATASETS:
        for seed in SEEDS:
            print("\n" + "=" * 80)
            print(f"FCN | dataset={dataset_name} | seed={seed}")
            if row_already_ok(rows, "FCN", dataset_name, seed):
                print("[SKIP] Existing OK row with all three metrics.")
                continue
            try:
                row = run_one(dataset_name, seed)
            except Exception as e:
                traceback.print_exc()
                row = {"model": "FCN", "dataset": dataset_name, "seed": seed, "status": "ERROR", "error": repr(e)}
            rows = remove_same_key(rows, "FCN", dataset_name, seed)
            rows.append(row)
            save_experiment_tables(rows, OUTPUT_ROOT, "FCN", group_cols=("dataset",))
    print("\nDone. Use FCN_summary_mean_std.csv for the paper table.")


if __name__ == "__main__":
    main()
