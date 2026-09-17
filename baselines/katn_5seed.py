from pathlib import Path
import os
import math
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from common_mtsc_utils import (
    DEFAULT_DATASETS, ArrayDataset, adjust_sequence_length, get_device, load_uea_train_test,
    load_existing_rows, make_loader, remove_same_key, row_already_ok,
    save_experiment_tables, set_seed, should_retry_cuda_oom, train_fixed_epochs,
    z_normalize_by_train,
)




BASE_PARENT = Path(os.environ.get(
    "AIM_DATASET_ROOT", str(Path(__file__).resolve().parent.parent / "dataset")
))
TS_TYPE = "Multivariate_ts"
DATASETS = DEFAULT_DATASETS
SEEDS = [42, 43, 44, 45, 46]
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs" / "KATN"

BATCH_SIZE = 32
EPOCHS = 300
LR = 1e-5
WEIGHT_DECAY = 5e-4
PRINT_EVERY = 1
GRAD_CLIP_NORM = 1.0
FIXED_LENGTH = None
UPSAMPLE_METHOD = "linear"


D_MODELS = [128, 128, 256, 256]
N_HEADS = [8, 8, 16, 16]
DROPOUT = 0.2
OPTIMIZER_NAME = "adam"


class MResNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=8, padding="same")
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding="same")
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding="same")
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x_in = x.transpose(1, 2)
        o1 = self.act(self.bn1(self.conv1(x_in)))
        o2 = self.act(self.bn2(self.conv2(o1)))
        o3 = self.bn3(self.conv3(o2))
        q = o1.transpose(1, 2)
        k = o2.transpose(1, 2)
        v = o3.transpose(1, 2)
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(k.size(-1))
        attn = F.softmax(scores, dim=-1)
        caf = torch.bmm(attn, v)
        return self.act(caf + o3.transpose(1, 2))


class AggregationTransformerBlock(nn.Module):
    def __init__(self, in_dim: int, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model) if in_dim != d_model else nn.Identity()
        self.norm = nn.LayerNorm(d_model)
        self.mresnet = MResNet(d_model, d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.dense = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_proj = self.proj(x)
        x_norm = self.norm(x_proj)
        local = self.mresnet(x_norm)
        global_, _ = self.mha(x_norm, x_norm, x_norm, need_weights=False)
        fused = local + global_
        out = self.dropout(self.act(self.dense(fused)))
        return out + x_proj


class KATN(nn.Module):
    def __init__(self, n_classes: int, seq_len: int, in_dim: int,
                 d_models=None, n_heads=None, dropout: float = 0.2):
        super().__init__()
        d_models = d_models or D_MODELS
        n_heads = n_heads or N_HEADS
        self.blocks = nn.ModuleList()
        cur = in_dim
        for i, d_model in enumerate(d_models):
            head = n_heads[i] if i < len(n_heads) else n_heads[-1]
            self.blocks.append(AggregationTransformerBlock(cur, d_model, head, dropout=dropout))
            cur = d_model
        self.norm = nn.LayerNorm(cur)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * cur, 128),
            nn.GELU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        h = x
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return self.classifier(h)


def make_optimizer(model):
    name = OPTIMIZER_NAME.lower()
    if name == "adam":
        return optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.95, 0.999))
    if name == "adamw":
        return optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.95, 0.999))
    raise ValueError(f"Unknown optimizer: {OPTIMIZER_NAME}")


def run_one(dataset_name: str, seed: int):
    set_seed(seed)
    device = get_device()
    X_train, y_train, X_test, y_test, label_map, dpath = load_uea_train_test(BASE_PARENT, dataset_name, ts_type=TS_TYPE)
    X_train = adjust_sequence_length(X_train, FIXED_LENGTH, upsample_method=UPSAMPLE_METHOD)
    X_test = adjust_sequence_length(X_test, FIXED_LENGTH, upsample_method=UPSAMPLE_METHOD)
    X_train, X_test = z_normalize_by_train(X_train, X_test)

    num_classes = int(y_train.max()) + 1
    seq_len = int(X_train.shape[1])
    in_dim = int(X_train.shape[2])
    train_loader = make_loader(ArrayDataset(X_train, y_train), BATCH_SIZE, True, seed)
    test_loader = make_loader(ArrayDataset(X_test, y_test), BATCH_SIZE, False, seed)

    model = KATN(num_classes, seq_len, in_dim, D_MODELS, N_HEADS, dropout=DROPOUT).to(device)
    optimizer = make_optimizer(model)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=5)
    history = train_fixed_epochs(
        model, train_loader, test_loader, device, EPOCHS, optimizer,
        scheduler=scheduler,
        grad_clip_norm=GRAD_CLIP_NORM,
        print_every=PRINT_EVERY,
        tag=f"[KATN][{dataset_name}][seed={seed}]"
    )
    final = history[-1]
    return {
        "model": "KATN", "dataset": dataset_name, "seed": seed,
        "seq_len": seq_len, "channels": in_dim,
        "num_classes": num_classes, "final_test_acc": float(final["test_acc"]),
        "final_test_macro_f1": float(final["test_macro_f1"]),
        "final_test_balanced_acc": float(final["test_balanced_acc"]),
        "final_train_acc": float(final["train_acc"]), "final_train_loss": float(final["train_loss"]),
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
        "d_models": str(D_MODELS), "n_heads": str(N_HEADS),
        "status": "OK", "error": "",
    }


def main():
    rows = load_existing_rows(OUTPUT_ROOT / "KATN_per_seed_results.csv")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for dataset_name in DATASETS:
        for seed in SEEDS:
            print("\n" + "=" * 80)
            print(f"KATN | dataset={dataset_name} | seed={seed}")
            if row_already_ok(rows, "KATN", dataset_name, seed):
                print("[SKIP] Existing OK row with all three metrics.")
                continue
            try:
                row = run_one(dataset_name, seed)
            except Exception as e:
                if should_retry_cuda_oom(e):
                    torch.cuda.empty_cache()
                    raise
                traceback.print_exc()
                row = {"model": "KATN", "dataset": dataset_name, "seed": seed, "status": "ERROR", "error": repr(e)}
            rows = remove_same_key(rows, "KATN", dataset_name, seed)
            rows.append(row)
            save_experiment_tables(rows, OUTPUT_ROOT, "KATN", group_cols=("dataset",))
    print("\nDone. Use KATN_summary_mean_std.csv for the paper table.")


if __name__ == "__main__":
    main()
