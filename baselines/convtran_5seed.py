from pathlib import Path
import os
import sys
import traceback
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader

from common_mtsc_utils import (
    DEFAULT_DATASETS,
    adjust_sequence_length,
    cuda_is_available,
    get_device,
    load_existing_rows,
    load_uea_train_test,
    remove_same_key,
    row_already_ok,
    save_experiment_tables,
    set_seed,
    should_retry_cuda_oom,
    z_normalize_by_train,
)










BASE_PARENT = Path(os.environ.get(
    "AIM_DATASET_ROOT", str(Path(__file__).resolve().parent.parent / "dataset")
))
TS_TYPE = "Multivariate_ts"


DATASETS = DEFAULT_DATASETS

SEEDS = [42, 43, 44, 45, 46]
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs" / "ConvTran"

SCRIPT_DIR = Path(__file__).resolve().parent
CONVTRAN_PATH = SCRIPT_DIR / "ConvTran-main"


BATCH_SIZE = 32
EPOCHS = 300
LR = 5e-4
WEIGHT_DECAY = 1e-2
PRINT_EVERY = 25
GRAD_CLIP_NORM = 1.0
FIXED_LENGTH = None
UPSAMPLE_METHOD = "linear"


USE_AMP = True
DETERMINISTIC = False
NUM_WORKERS = 0  


NET_TYPE = "C-T"
EMB_SIZE = 64
DIM_FF = 256
NUM_HEADS = 8
DROPOUT = 0.1
FIX_POS_ENCODE = "tAPE"
REL_POS_ENCODE = "eRPE"
OPTIMIZER_NAME = "radam"

if not CONVTRAN_PATH.exists():
    raise FileNotFoundError(
        f"ConvTran-main folder not found: {CONVTRAN_PATH}\n"
        "Place it next to this script or edit CONVTRAN_PATH."
    )
sys.path.append(str(CONVTRAN_PATH))
from Models.model import model_factory  





def _is_cuda(device):
    return str(device).startswith("cuda")


def configure_speed(seed: int):



    set_seed(seed)
    torch.backends.cudnn.deterministic = DETERMINISTIC
    torch.backends.cudnn.benchmark = not DETERMINISTIC
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def get_batch_size(dataset_name: str) -> int:
    return BATCH_SIZE


class ConvTranDataset(Dataset):
    def __init__(self, x_1d, y):





        x_1d = np.asarray(x_1d, dtype=np.float32)
        x_1d = np.ascontiguousarray(x_1d.transpose(0, 2, 1))
        y = np.asarray(y, dtype=np.int64)

        self.x_1d = torch.from_numpy(x_1d).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.x_1d.shape[0]

    def __getitem__(self, idx):
        return self.x_1d[idx], self.y[idx]


def make_fast_loader(dataset, batch_size, shuffle, seed, device):
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator if shuffle else None,
        num_workers=NUM_WORKERS,
        pin_memory=_is_cuda(device),
        drop_last=False,
    )


def make_optimizer(model):
    name = OPTIMIZER_NAME.lower()
    if name == "radam":
        return optim.RAdam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    if name == "adamw":
        return optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    if name == "adam":
        return optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    raise ValueError(f"Unknown optimizer: {OPTIMIZER_NAME}")


@torch.no_grad()
def evaluate_convtran(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_pred = []
    all_y = []
    amp_on = USE_AMP and _is_cuda(device)

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp_on):
            logits = model(xb)
            loss = loss_fn(logits, yb)

        total_loss += float(loss.item()) * yb.size(0)
        predicted = logits.argmax(dim=1)
        correct += int((predicted == yb).sum().item())
        total += int(yb.size(0))
        all_pred.extend(predicted.detach().cpu().numpy().tolist())
        all_y.extend(yb.detach().cpu().numpy().tolist())

    return (
        total_loss / max(total, 1),
        correct / max(total, 1),
        float(f1_score(all_y, all_pred, average="macro", zero_division=0)),
        float(balanced_accuracy_score(all_y, all_pred)),
    )


def train_convtran_fast(model, train_loader, test_loader, device, epochs, optimizer, tag=""):







    loss_fn = nn.CrossEntropyLoss()
    amp_on = USE_AMP and _is_cuda(device)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_on)
    last_train_loss = np.nan

    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        total = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_on):
                logits = model(xb)
                loss = loss_fn(logits, yb)

            scaler.scale(loss).backward()

            if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item()) * yb.size(0)
            total += int(yb.size(0))

        last_train_loss = total_loss / max(total, 1)

        if epoch == 1 or epoch % PRINT_EVERY == 0 or epoch == epochs:
            print(f"{tag} Ep {epoch:03d}/{epochs} | train_loss={last_train_loss:.4f}", flush=True)

    final_test_loss, final_test_acc, final_test_macro_f1, final_test_balanced_acc = (
        evaluate_convtran(model, test_loader, loss_fn, device)
    )

    return {
        "train_loss": float(last_train_loss),
        "test_loss": float(final_test_loss),
        "test_acc": float(final_test_acc),
        "test_macro_f1": float(final_test_macro_f1),
        "test_balanced_acc": float(final_test_balanced_acc),
    }





def run_one(dataset_name: str, seed: int):
    configure_speed(seed)
    device = get_device()

    X_train, y_train, X_test, y_test, label_map, dpath = load_uea_train_test(
        BASE_PARENT, dataset_name, ts_type=TS_TYPE
    )
    X_train = adjust_sequence_length(X_train, FIXED_LENGTH, upsample_method=UPSAMPLE_METHOD)
    X_test = adjust_sequence_length(X_test, FIXED_LENGTH, upsample_method=UPSAMPLE_METHOD)
    X_train, X_test = z_normalize_by_train(X_train, X_test)

    num_classes = int(y_train.max()) + 1
    seq_len = int(X_train.shape[1])
    n_channels = int(X_train.shape[2])
    batch_size = get_batch_size(dataset_name)

    train_ds = ConvTranDataset(X_train, y_train)
    test_ds = ConvTranDataset(X_test, y_test)
    train_loader = make_fast_loader(train_ds, batch_size, True, seed, device)
    test_loader = make_fast_loader(test_ds, batch_size, False, seed, device)

    config = {
        "Net_Type": [NET_TYPE],
        "Data_shape": (None, n_channels, seq_len),
        "num_labels": int(num_classes),
        "emb_size": EMB_SIZE,
        "dim_ff": DIM_FF,
        "num_heads": NUM_HEADS,
        "dropout": DROPOUT,
        "Fix_pos_encode": FIX_POS_ENCODE,
        "Rel_pos_encode": REL_POS_ENCODE,
    }

    print(
        f"[ConvTran setup] dataset={dataset_name} | seed={seed} | "
        f"seq_len={seq_len} | channels={n_channels} | batch_size={batch_size} | "
        f"device={device} | amp={USE_AMP and _is_cuda(device)}",
        flush=True,
    )

    model = model_factory(config).to(device)
    optimizer = make_optimizer(model)

    final = train_convtran_fast(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=EPOCHS,
        optimizer=optimizer,
        tag=f"[ConvTran][{dataset_name}][seed={seed}]",
    )

    return {
        "model": "ConvTran",
        "dataset": dataset_name,
        "seed": seed,
        "seq_len": seq_len,
        "channels": n_channels,
        "num_classes": num_classes,
        "final_test_acc": float(final["test_acc"]),
        "final_test_macro_f1": float(final["test_macro_f1"]),
        "final_test_balanced_acc": float(final["test_balanced_acc"]),
        "final_test_loss": float(final["test_loss"]),
        "final_train_acc": np.nan,
        "final_train_loss": float(final["train_loss"]),
        "epochs": EPOCHS,
        "batch_size": batch_size,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "net_type": NET_TYPE,
        "emb_size": EMB_SIZE,
        "dim_ff": DIM_FF,
        "num_heads": NUM_HEADS,
        "fix_pos_encode": FIX_POS_ENCODE,
        "rel_pos_encode": REL_POS_ENCODE,
        "use_amp": USE_AMP,
        "deterministic": DETERMINISTIC,
        "eval_protocol": "final_test_only_no_scheduler_no_checkpoint",
        "status": "OK",
        "error": "",
    }





def main():
    rows = load_existing_rows(OUTPUT_ROOT / "ConvTran_per_seed_results.csv")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for dataset_name in DATASETS:
        for seed in SEEDS:
            print("\n" + "=" * 80, flush=True)
            print(f"ConvTran | dataset={dataset_name} | seed={seed}", flush=True)

            if row_already_ok(rows, "ConvTran", dataset_name, seed):
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
                    "model": "ConvTran",
                    "dataset": dataset_name,
                    "seed": seed,
                    "status": "ERROR",
                    "error": repr(e),
                }

            rows = remove_same_key(rows, "ConvTran", dataset_name, seed)
            rows.append(row)
            save_experiment_tables(rows, OUTPUT_ROOT, "ConvTran", group_cols=("dataset",))

            if cuda_is_available():
                torch.cuda.empty_cache()
            gc.collect()

    print("\nDone. Use ConvTran_summary_mean_std.csv for the paper table.", flush=True)


if __name__ == "__main__":
    main()
