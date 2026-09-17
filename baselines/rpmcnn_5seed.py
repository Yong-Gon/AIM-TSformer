from pathlib import Path
from contextlib import nullcontext
import gc
import math
import os
import re
import traceback
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.optim import Adagrad
from torch.utils.data import Dataset, DataLoader

from common_mtsc_utils import (
    DEFAULT_DATASETS,
    get_device,
    load_existing_rows,
    load_uea_train_test,
    row_already_ok,
    resolve_dataset_dir,
    save_experiment_tables,
    set_seed,
    should_retry_cuda_oom,
)




BASE_PARENT = Path(os.environ.get(
    "AIM_DATASET_ROOT", str(Path(__file__).resolve().parent.parent / "dataset")
))
TS_TYPE = "Multivariate_ts"
DATASETS = [
    item.strip() for item in os.environ.get("AIM_DATASETS", "").split(",") if item.strip()
] or DEFAULT_DATASETS

SEEDS = [
    int(item.strip()) for item in os.environ.get("AIM_SEEDS", "42,43,44,45,46").split(",")
    if item.strip()
]



EXCLUDE_DATASETS_FROM_CSV = ["PenDigits", "RacketSports"]
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs" / "RPM-CNN"

BATCH_SIZE = int(os.environ.get("AIM_BATCH_SIZE", "32"))
EPOCHS = int(os.environ.get("AIM_EPOCHS", "300"))
LR = 2e-4
WEIGHT_DECAY = 0.0
PRINT_EVERY = 25
FC_HIDDEN_DIM = 128 

IMAGE_SUFFIX = ""  


NUM_WORKERS = 0
USE_AMP = True
DETERMINISTIC = False
GRAD_CLIP_NORM = None
SCALE = 255.0
MINMAX_CHUNK_SAMPLES = 16
NORMALIZE_CHUNK_SAMPLES = 8
EVAL_TRAIN_AT_END = False










SEQ_LENS_BY_DATASET = {}
RUN_ALL_AVAILABLE_LENGTHS = False
LENGTH_SELECTION_POLICY = "original_if_available_else_min"  

class RPMCNN(nn.Module):
    def __init__(self, input_channels: int, input_size: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.SELU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.SELU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.SELU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.SELU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.SELU(inplace=True), nn.MaxPool2d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_size, input_size)
            flat_dim = int(self.features(dummy).reshape(1, -1).shape[1])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, FC_HIDDEN_DIM), nn.SELU(inplace=True), nn.Dropout(0.5),
            nn.Linear(FC_HIDDEN_DIM, FC_HIDDEN_DIM), nn.SELU(inplace=True), nn.Dropout(0.5),
            nn.Linear(FC_HIDDEN_DIM, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ImageTensorDataset(Dataset):
    def __init__(self, x_nchw: np.ndarray, y: np.ndarray):
        if x_nchw.ndim != 4:
            raise ValueError(f"x_nchw must be 4D (N,C,H,W), got {x_nchw.shape}")
        self.X = x_nchw
        if not getattr(x_nchw, "is_lazy_normalized_memmap", False):
            self.X = torch.from_numpy(np.ascontiguousarray(x_nchw, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64)).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        if not torch.is_tensor(x):
            x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        return x, self.y[idx]


class NormalizedMemmapImageArray:

    is_lazy_normalized_memmap = True

    def __init__(self, raw: np.ndarray, mins: np.ndarray, maxs: np.ndarray, scale: float = SCALE):
        self.raw = ensure_4d_hwcn(raw)
        n, h, w, c = self.raw.shape
        self.shape = (n, c, h, w)
        self.ndim = 4
        self.mins = np.asarray(mins, dtype=np.float32).reshape(1, 1, c)
        self.den = np.maximum(
            np.asarray(maxs, dtype=np.float32) - np.asarray(mins, dtype=np.float32),
            1e-8,
        ).reshape(1, 1, c)
        self.scale = float(scale)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        sample = np.asarray(self.raw[idx], dtype=np.float32)
        sample = self.scale * (sample - self.mins) / self.den
        sample = np.nan_to_num(
            sample, nan=0.0, posinf=self.scale, neginf=0.0
        ).astype(np.float32, copy=False)
        return np.ascontiguousarray(sample.transpose(2, 0, 1))


def make_image_loader(x_nchw, y, batch_size, shuffle, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ImageTensorDataset(x_nchw, y),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=NUM_WORKERS,
        generator=generator if shuffle else None,
        pin_memory=(get_device().type == "cuda"),
    )


def amp_context(device):
    if not (USE_AMP and device.type == "cuda"):
        return nullcontext()
    try:
        return torch.amp.autocast("cuda")
    except Exception:
        return torch.cuda.amp.autocast()


def make_scaler(device):
    enabled = bool(USE_AMP and device.type == "cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def discover_seq_lens(dataset_dir: Path, dataset_name: str):
    suffix = re.escape(IMAGE_SUFFIX)
    pat = re.compile(rf"{re.escape(dataset_name)}_rpm_train_(\d+){suffix}\.npy$")
    seq_lens = []
    for f in dataset_dir.iterdir():
        m = pat.match(f.name)
        if m:
            seq_lens.append(int(m.group(1)))
    return sorted(set(seq_lens))


def select_seq_lens(dataset_name: str, dataset_dir: Path, original_len: int):
    if dataset_name in SEQ_LENS_BY_DATASET:
        return list(SEQ_LENS_BY_DATASET[dataset_name])

    available = discover_seq_lens(dataset_dir, dataset_name)
    if not available:
        return []

    if RUN_ALL_AVAILABLE_LENGTHS:
        return available

    if LENGTH_SELECTION_POLICY == "original_if_available_else_min":
        return [original_len if original_len in available else min(available)]
    if LENGTH_SELECTION_POLICY == "min":
        return [min(available)]
    if LENGTH_SELECTION_POLICY == "max":
        return [max(available)]

    raise ValueError(f"Unknown LENGTH_SELECTION_POLICY={LENGTH_SELECTION_POLICY}")


def ensure_4d_hwcn(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        return arr[:, :, :, None]
    if arr.ndim == 4:
        return arr
    raise ValueError(f"Expected 3D or 4D image array, got shape={arr.shape}")


def fit_channel_minmax_chunked(train_img: np.ndarray):
    train_img = ensure_4d_hwcn(train_img)
    n, _, _, c = train_img.shape
    mins = np.full(c, np.inf, dtype=np.float32)
    maxs = np.full(c, -np.inf, dtype=np.float32)

    for s in range(0, n, MINMAX_CHUNK_SAMPLES):
        e = min(s + MINMAX_CHUNK_SAMPLES, n)
        chunk = np.asarray(train_img[s:e], dtype=np.float32)
        mins = np.minimum(mins, np.nanmin(chunk, axis=(0, 1, 2)).astype(np.float32))
        maxs = np.maximum(maxs, np.nanmax(chunk, axis=(0, 1, 2)).astype(np.float32))

    mins = np.nan_to_num(mins, nan=0.0, posinf=0.0, neginf=0.0)
    maxs = np.nan_to_num(maxs, nan=1.0, posinf=1.0, neginf=1.0)
    return mins.astype(np.float32), maxs.astype(np.float32)


def normalize_to_nchw_chunked(img: np.ndarray, mins: np.ndarray, maxs: np.ndarray, scale: float = SCALE):
    img = ensure_4d_hwcn(img)
    n, h, w, c = img.shape
    out = np.empty((n, c, h, w), dtype=np.float32)

    mins_r = mins.reshape(1, 1, 1, c).astype(np.float32)
    den_r = np.maximum(maxs - mins, 1e-8).reshape(1, 1, 1, c).astype(np.float32)

    for s in range(0, n, NORMALIZE_CHUNK_SAMPLES):
        e = min(s + NORMALIZE_CHUNK_SAMPLES, n)
        chunk = np.asarray(img[s:e], dtype=np.float32)
        chunk = scale * (chunk - mins_r) / den_r
        chunk = np.nan_to_num(chunk, nan=0.0, posinf=scale, neginf=0.0).astype(np.float32, copy=False)
        out[s:e] = np.ascontiguousarray(chunk.transpose(0, 3, 1, 2))

    return out


def load_rpm_arrays_nchw(dataset_dir: Path, dataset_name: str, seq_len: int):
    train_path = dataset_dir / f"{dataset_name}_rpm_train_{seq_len}{IMAGE_SUFFIX}.npy"
    test_path = dataset_dir / f"{dataset_name}_rpm_test_{seq_len}{IMAGE_SUFFIX}.npy"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing RPM npy files: {train_path}, {test_path}")

    
    train_raw = np.load(train_path, mmap_mode="r")
    test_raw = np.load(test_path, mmap_mode="r")

    print(f"[LOAD] {dataset_name} L={seq_len} train_raw={train_raw.shape}, test_raw={test_raw.shape}", flush=True)
    stats_path = train_path.with_suffix(".stats.npz")
    if stats_path.exists():
        with np.load(stats_path, allow_pickle=False) as stats:
            mins = np.asarray(stats["mins"], dtype=np.float32)
            maxs = np.asarray(stats["maxs"], dtype=np.float32)
        print(f"[LOAD] channel min/max sidecar={stats_path.name}", flush=True)
    else:
        mins, maxs = fit_channel_minmax_chunked(train_raw)
    train_x = NormalizedMemmapImageArray(train_raw, mins, maxs, scale=SCALE)
    test_x = NormalizedMemmapImageArray(test_raw, mins, maxs, scale=SCALE)

    return train_x, test_x


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_pred = []
    all_y = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with amp_context(device):
                logits = model(xb)
                loss = criterion(logits, yb)
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


def train_final_eval(model, train_loader, test_loader, device, epochs, optimizer, tag):
    criterion = nn.CrossEntropyLoss()
    scaler = make_scaler(device)
    last_train_loss = math.nan

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with amp_context(device):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item()) * yb.size(0)
            total += int(yb.size(0))

        last_train_loss = total_loss / max(total, 1)
        if PRINT_EVERY and (epoch == 1 or epoch == epochs or epoch % PRINT_EVERY == 0):
            print(f"{tag} Ep {epoch:03d}/{epochs} | train_loss={last_train_loss:.4f}", flush=True)

    test_loss, test_acc, test_macro_f1, test_balanced_acc = evaluate(
        model, test_loader, criterion, device
    )
    if EVAL_TRAIN_AT_END:
        _, train_acc, _, _ = evaluate(model, train_loader, criterion, device)
    else:
        train_acc = math.nan

    return {
        "final_train_loss": float(last_train_loss),
        "final_train_acc": float(train_acc),
        "final_test_loss": float(test_loss),
        "final_test_acc": float(test_acc),
        "final_test_macro_f1": float(test_macro_f1),
        "final_test_balanced_acc": float(test_balanced_acc),
    }


def run_one_preloaded(dataset_name: str, seq_len: int, seed: int, train_x, test_x, y_train, y_test, num_classes: int):
    set_seed(seed, deterministic=DETERMINISTIC)
    device = get_device()

    input_channels = int(train_x.shape[1])
    input_size = int(train_x.shape[2])
    batch_size = BATCH_SIZE

    train_loader = make_image_loader(train_x, y_train, batch_size, True, seed)
    test_loader = make_image_loader(test_x, y_test, batch_size, False, seed)

    model = RPMCNN(input_channels=input_channels, input_size=input_size, num_classes=num_classes).to(device)
    optimizer = Adagrad(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    result = train_final_eval(
        model, train_loader, test_loader, device, EPOCHS, optimizer,
        tag=f"[RPM-CNN][{dataset_name}][L={seq_len}][seed={seed}]"
    )

    del model, optimizer, train_loader, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "model": "RPM-CNN", "dataset": dataset_name, "seq_len": seq_len, "image_size": input_size,
        "image_suffix": IMAGE_SUFFIX, "seed": seed, "channels": input_channels,
        "num_classes": num_classes, "final_test_acc": result["final_test_acc"],
        "final_test_macro_f1": result["final_test_macro_f1"],
        "final_test_balanced_acc": result["final_test_balanced_acc"],
        "final_train_acc": result["final_train_acc"], "final_train_loss": result["final_train_loss"],
        "final_test_loss": result["final_test_loss"],
        "epochs": EPOCHS, "batch_size": batch_size, "lr": LR,
        "run_all_available_lengths": RUN_ALL_AVAILABLE_LENGTHS,
        "length_selection_policy": LENGTH_SELECTION_POLICY,
        "use_amp": USE_AMP, "deterministic": DETERMINISTIC,
        "status": "OK", "error": "",
    }



def _existing_per_seed_csv_path():
    return OUTPUT_ROOT / "RPM-CNN_per_seed_results.csv"


def backup_existing_csv_files():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for name in ["RPM-CNN_per_seed_results.csv", "RPM-CNN_summary_mean_std.csv"]:
        src = OUTPUT_ROOT / name
        if src.exists():
            dst = OUTPUT_ROOT / f"{src.stem}_backup_{timestamp}{src.suffix}"
            shutil.copy2(src, dst)
            print(f"[BACKUP] {src} -> {dst}", flush=True)


def load_existing_rows_for_resume(target_datasets):






    csv_path = _existing_per_seed_csv_path()
    if not csv_path.exists():
        print("[RESUME] Existing per-seed CSV not found. Starting from empty rows.", flush=True)
        return []

    old_df = pd.read_csv(csv_path)
    if "dataset" not in old_df.columns:
        print("[RESUME] Existing CSV has no dataset column. Starting from empty rows.", flush=True)
        return []

    remove_datasets = set(target_datasets) | set(EXCLUDE_DATASETS_FROM_CSV)
    keep_df = old_df[~old_df["dataset"].isin(remove_datasets)].copy()

    print(
        f"[RESUME] loaded rows={len(old_df)} | kept rows={len(keep_df)} | "
        f"removed datasets={sorted(remove_datasets)} | removed rows={len(old_df) - len(keep_df)}",
        flush=True,
    )
    return keep_df.to_dict("records")


def remove_same_key(rows, model_name, dataset_name, seq_len, image_suffix, seed):
    new_rows = []
    for r in rows:
        same_model = str(r.get("model", "")) == str(model_name)
        same_dataset = str(r.get("dataset", "")) == str(dataset_name)
        same_suffix = str(r.get("image_suffix", "")) == str(image_suffix)

        try:
            same_seq = int(float(r.get("seq_len"))) == int(seq_len)
        except Exception:
            same_seq = str(r.get("seq_len", "")) == str(seq_len)

        try:
            same_seed = int(float(r.get("seed"))) == int(seed)
        except Exception:
            same_seed = str(r.get("seed", "")) == str(seed)

        if same_model and same_dataset and same_seq and same_suffix and same_seed:
            continue
        new_rows.append(r)
    return new_rows

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80, flush=True)
    print("[RUN MODE] RPM-CNN resume-safe 5-seed experiment", flush=True)
    print(f"DATASETS = {DATASETS}", flush=True)
    print(f"SEQ_LENS_BY_DATASET = {SEQ_LENS_BY_DATASET}", flush=True)
    print(f"IMAGE_SUFFIX = {IMAGE_SUFFIX!r}", flush=True)
    print(f"EXCLUDE_DATASETS_FROM_CSV = {EXCLUDE_DATASETS_FROM_CSV}", flush=True)
    print("=" * 80, flush=True)

    rows = load_existing_rows(_existing_per_seed_csv_path())

    for dataset_name in DATASETS:
        try:
            dataset_dir = resolve_dataset_dir(BASE_PARENT, dataset_name, ts_type=TS_TYPE)
            X_train_meta, y_train, _, y_test, label_map, _ = load_uea_train_test(BASE_PARENT, dataset_name, ts_type=TS_TYPE)
            original_len = int(X_train_meta.shape[1])
            num_classes = int(y_train.max()) + 1
            del X_train_meta

            seq_lens = select_seq_lens(dataset_name, dataset_dir, original_len)
            if not seq_lens:
                print(f"[RPM-CNN] skip {dataset_name}: no RPM files with suffix={IMAGE_SUFFIX!r}", flush=True)
                continue

            for seq_len in seq_lens:
                train_x = test_x = None
                try:
                    train_x, test_x = load_rpm_arrays_nchw(dataset_dir, dataset_name, seq_len)
                    for seed in SEEDS:
                        print("\n" + "=" * 80, flush=True)
                        print(f"RPM-CNN | dataset={dataset_name} | seq_len={seq_len} | suffix={IMAGE_SUFFIX} | seed={seed}", flush=True)
                        if row_already_ok(rows, "RPM-CNN", dataset_name, seed):
                            print("[SKIP] Existing OK row with all three metrics.", flush=True)
                            continue
                        try:
                            row = run_one_preloaded(dataset_name, seq_len, seed, train_x, test_x, y_train, y_test, num_classes)
                        except Exception as e:
                            if should_retry_cuda_oom(e):
                                torch.cuda.empty_cache()
                                raise
                            traceback.print_exc()
                            row = {
                                "model": "RPM-CNN", "dataset": dataset_name, "seq_len": seq_len,
                                "image_suffix": IMAGE_SUFFIX, "seed": seed,
                                "status": "ERROR", "error": repr(e),
                            }

                        rows = remove_same_key(rows, "RPM-CNN", dataset_name, seq_len, IMAGE_SUFFIX, seed)
                        rows.append(row)
                        save_experiment_tables(rows, OUTPUT_ROOT, "RPM-CNN", group_cols=("dataset", "seq_len", "image_suffix"))

                finally:
                    del train_x, test_x
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

        except Exception as e:
            if should_retry_cuda_oom(e):
                torch.cuda.empty_cache()
                raise
            traceback.print_exc()
            for seed in SEEDS:
                row = {
                    "model": "RPM-CNN", "dataset": dataset_name,
                    "seq_len": SEQ_LENS_BY_DATASET.get(dataset_name, [None])[0],
                    "image_suffix": IMAGE_SUFFIX,
                    "seed": seed, "status": "ERROR", "error": repr(e),
                }
                rows = remove_same_key(rows, "RPM-CNN", dataset_name, row["seq_len"], IMAGE_SUFFIX, seed)
                rows.append(row)
            save_experiment_tables(rows, OUTPUT_ROOT, "RPM-CNN", group_cols=("dataset", "seq_len", "image_suffix"))

    print("\nDone. RPM-CNN results were merged with the existing per-seed CSV.", flush=True)


if __name__ == "__main__":
    main()
