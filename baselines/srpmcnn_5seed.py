from pathlib import Path
from contextlib import nullcontext
import gc
import math
import os
import re
import traceback
import time

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader

from common_mtsc_utils import (
    DEFAULT_DATASETS,
    get_device,
    load_existing_rows,
    load_uea_train_test,
    remove_same_key,
    resolve_dataset_dir,
    row_already_ok,
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
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs" / "SRPM-CNN"

BATCH_SIZE = int(os.environ.get("AIM_BATCH_SIZE", "32"))
EPOCHS = int(os.environ.get("AIM_EPOCHS", "300"))
LR = 5e-6
WEIGHT_DECAY = 0.0
PRINT_EVERY = 25
WINDOW = 2
IMAGE_SUFFIX = ""  

NUM_WORKERS = 0
USE_AMP = True
DETERMINISTIC = False
GRAD_CLIP_NORM = None
SCALE = 255.0
MINMAX_CHUNK_SAMPLES = 16
NORMALIZE_CHUNK_SAMPLES = 8
EVAL_TRAIN_AT_END = False
USE_TQDM = True
SHOW_BATCH_PROGRESS = False
TQDM_MININTERVAL = 1.0

SEQ_LENS_BY_DATASET = {}
RUN_ALL_AVAILABLE_LENGTHS = False
LENGTH_SELECTION_POLICY = "original_if_available_else_min"  

def format_seconds(seconds):
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def get_cuda_memory_text():
    if not torch.cuda.is_available():
        return ""

    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    return f" | cuda_alloc={allocated:.1f}MB | cuda_reserved={reserved:.1f}MB"


def maybe_tqdm(iterable, **kwargs):
    if USE_TQDM and tqdm is not None:
        return tqdm(iterable, **kwargs)
    return iterable

class SRPMCNN(nn.Module):
    def __init__(self, input_channels: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)


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
    pat = re.compile(rf"{re.escape(dataset_name)}_srpm_train_win{WINDOW}_len(\d+){suffix}\.npy$")
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


def load_srpm_arrays_nchw(dataset_dir: Path, dataset_name: str, seq_len: int):
    train_path = dataset_dir / f"{dataset_name}_srpm_train_win{WINDOW}_len{seq_len}{IMAGE_SUFFIX}.npy"
    test_path = dataset_dir / f"{dataset_name}_srpm_test_win{WINDOW}_len{seq_len}{IMAGE_SUFFIX}.npy"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing SRPM npy files: {train_path}, {test_path}")

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

    run_start_time = time.perf_counter()
    epoch_times = []

    epoch_iter = range(1, epochs + 1)

    if USE_TQDM and tqdm is not None:
        epoch_iter = tqdm(
            epoch_iter,
            total=epochs,
            desc=f"{tag} epochs",
            dynamic_ncols=True,
            mininterval=TQDM_MININTERVAL,
            leave=True,
        )

    for epoch in epoch_iter:
        epoch_start_time = time.perf_counter()

        model.train()
        total_loss = 0.0
        total = 0

        batch_iter = train_loader
        if SHOW_BATCH_PROGRESS and USE_TQDM and tqdm is not None:
            batch_iter = tqdm(
                train_loader,
                total=len(train_loader),
                desc=f"{tag} Ep {epoch:03d}/{epochs}",
                dynamic_ncols=True,
                mininterval=TQDM_MININTERVAL,
                leave=False,
            )

        for xb, yb in batch_iter:
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

        epoch_elapsed = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_elapsed)

        avg_epoch_time = float(np.mean(epoch_times))
        remaining_epochs = epochs - epoch
        eta_seconds = avg_epoch_time * remaining_epochs
        total_elapsed = time.perf_counter() - run_start_time

        msg = (
            f"{tag} Ep {epoch:03d}/{epochs} | "
            f"train_loss={last_train_loss:.4f} | "
            f"epoch_time={format_seconds(epoch_elapsed)} | "
            f"elapsed={format_seconds(total_elapsed)} | "
            f"eta={format_seconds(eta_seconds)}"
            f"{get_cuda_memory_text()}"
        )

        if USE_TQDM and tqdm is not None and hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix({
                "loss": f"{last_train_loss:.4f}",
                "epoch": format_seconds(epoch_elapsed),
                "eta": format_seconds(eta_seconds),
            })

        if PRINT_EVERY and (epoch == 1 or epoch == epochs or epoch % PRINT_EVERY == 0):
            print(msg, flush=True)

    eval_start_time = time.perf_counter()
    test_loss, test_acc, test_macro_f1, test_balanced_acc = evaluate(
        model, test_loader, criterion, device
    )
    test_eval_elapsed = time.perf_counter() - eval_start_time

    if EVAL_TRAIN_AT_END:
        train_eval_start = time.perf_counter()
        _, train_acc, _, _ = evaluate(model, train_loader, criterion, device)
        train_eval_elapsed = time.perf_counter() - train_eval_start
    else:
        train_acc = math.nan
        train_eval_elapsed = 0.0

    total_run_elapsed = time.perf_counter() - run_start_time
    train_time = total_run_elapsed - test_eval_elapsed - train_eval_elapsed

    print(
        f"{tag} finished | "
        f"test_acc={test_acc:.4f} | "
        f"test_loss={test_loss:.4f} | "
        f"train_time={format_seconds(train_time)} | "
        f"eval_time={format_seconds(test_eval_elapsed)} | "
        f"total_time={format_seconds(total_run_elapsed)}"
        f"{get_cuda_memory_text()}",
        flush=True,
    )

    return {
        "final_train_loss": float(last_train_loss),
        "final_train_acc": float(train_acc),
        "final_test_loss": float(test_loss),
        "final_test_acc": float(test_acc),
        "final_test_macro_f1": float(test_macro_f1),
        "final_test_balanced_acc": float(test_balanced_acc),
        "train_time_sec": float(train_time),
        "test_eval_time_sec": float(test_eval_elapsed),
        "train_eval_time_sec": float(train_eval_elapsed),
        "total_time_sec": float(total_run_elapsed),
        "avg_epoch_time_sec": float(np.mean(epoch_times)) if epoch_times else math.nan,
    }


def run_one_preloaded(dataset_name: str, seq_len: int, seed: int, train_x, test_x, y_train, y_test, num_classes: int):
    set_seed(seed, deterministic=DETERMINISTIC)
    device = get_device()

    input_channels = int(train_x.shape[1])
    input_size = int(train_x.shape[2])
    batch_size = BATCH_SIZE

    train_loader = make_image_loader(train_x, y_train, batch_size, True, seed)
    test_loader = make_image_loader(test_x, y_test, batch_size, False, seed)

    model = SRPMCNN(input_channels=input_channels, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    result = train_final_eval(
        model, train_loader, test_loader, device, EPOCHS, optimizer,
        tag=f"[SRPM-CNN][{dataset_name}][L={seq_len}][seed={seed}]"
    )

    del model, optimizer, train_loader, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "model": "SRPM-CNN", "dataset": dataset_name, "seq_len": seq_len, "image_size": input_size,
        "window": WINDOW, "image_suffix": IMAGE_SUFFIX, "seed": seed, "channels": input_channels,
        "num_classes": num_classes, "final_test_acc": result["final_test_acc"],
        "final_test_macro_f1": result["final_test_macro_f1"],
        "final_test_balanced_acc": result["final_test_balanced_acc"],
        "final_train_acc": result["final_train_acc"], "final_train_loss": result["final_train_loss"],
        "final_test_loss": result["final_test_loss"],
        "train_time_sec": result["train_time_sec"],
        "test_eval_time_sec": result["test_eval_time_sec"],
        "train_eval_time_sec": result["train_eval_time_sec"],
        "total_time_sec": result["total_time_sec"],
        "avg_epoch_time_sec": result["avg_epoch_time_sec"],
        "epochs": EPOCHS, "batch_size": batch_size, "lr": LR,
        "run_all_available_lengths": RUN_ALL_AVAILABLE_LENGTHS,
        "length_selection_policy": LENGTH_SELECTION_POLICY,
        "use_amp": USE_AMP, "deterministic": DETERMINISTIC,
        "status": "OK", "error": "",
    }


def main():
    script_start_time = time.perf_counter()
    completed_jobs = 0
    job_times = []

    rows = load_existing_rows(OUTPUT_ROOT / "SRPM-CNN_per_seed_results.csv")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80, flush=True)
    print("[RUN START] SRPM-CNN 5-seed experiment", flush=True)
    print(f"DATASETS = {DATASETS}", flush=True)
    print(f"SEEDS = {SEEDS}", flush=True)
    print(f"EPOCHS = {EPOCHS}", flush=True)
    print(f"BATCH_SIZE = {BATCH_SIZE}", flush=True)
    print(f"USE_AMP = {USE_AMP}", flush=True)
    print(f"USE_TQDM = {USE_TQDM}", flush=True)
    print("=" * 80, flush=True)

    for dataset_name in DATASETS:
        dataset_start_time = time.perf_counter()

        try:
            dataset_dir = resolve_dataset_dir(BASE_PARENT, dataset_name, ts_type=TS_TYPE)

            load_meta_start = time.perf_counter()
            X_train_meta, y_train, _, y_test, label_map, _ = load_uea_train_test(
                BASE_PARENT, dataset_name, ts_type=TS_TYPE
            )
            original_len = int(X_train_meta.shape[1])
            num_classes = int(y_train.max()) + 1
            del X_train_meta
            load_meta_elapsed = time.perf_counter() - load_meta_start

            print(
                f"\n[DATASET] {dataset_name} | "
                f"meta_load_time={format_seconds(load_meta_elapsed)} | "
                f"original_len={original_len} | num_classes={num_classes}",
                flush=True,
            )

            seq_lens = select_seq_lens(dataset_name, dataset_dir, original_len)
            if not seq_lens:
                print(
                    f"[SRPM-CNN] skip {dataset_name}: "
                    f"no SRPM files with suffix={IMAGE_SUFFIX!r}, window={WINDOW}",
                    flush=True,
                )
                continue

            print(f"[DATASET] {dataset_name} | selected_seq_lens={seq_lens}", flush=True)

            for seq_len in seq_lens:
                seq_start_time = time.perf_counter()
                train_x = test_x = None

                try:
                    load_img_start = time.perf_counter()
                    train_x, test_x = load_srpm_arrays_nchw(dataset_dir, dataset_name, seq_len)
                    load_img_elapsed = time.perf_counter() - load_img_start

                    print(
                        f"[LOAD DONE] dataset={dataset_name} | seq_len={seq_len} | "
                        f"train_x={train_x.shape} | test_x={test_x.shape} | "
                        f"load_norm_time={format_seconds(load_img_elapsed)}"
                        f"{get_cuda_memory_text()}",
                        flush=True,
                    )

                    for seed in SEEDS:
                        job_start_time = time.perf_counter()

                        print("\n" + "=" * 80, flush=True)
                        print(
                            f"SRPM-CNN | dataset={dataset_name} | seq_len={seq_len} | "
                            f"seed={seed} | completed_jobs={completed_jobs}",
                            flush=True,
                        )

                        if row_already_ok(rows, "SRPM-CNN", dataset_name, seed):
                            print("[SKIP] Existing OK row with all three metrics.", flush=True)
                            continue

                        if job_times:
                            avg_job_time = float(np.mean(job_times))
                            print(
                                f"[TIME BEFORE JOB] avg_job_time={format_seconds(avg_job_time)} | "
                                f"script_elapsed={format_seconds(time.perf_counter() - script_start_time)}",
                                flush=True,
                            )

                        try:
                            row = run_one_preloaded(
                                dataset_name, seq_len, seed,
                                train_x, test_x, y_train, y_test, num_classes
                            )

                        except Exception as e:
                            if should_retry_cuda_oom(e):
                                torch.cuda.empty_cache()
                                raise
                            traceback.print_exc()
                            row = {
                                "model": "SRPM-CNN",
                                "dataset": dataset_name,
                                "seq_len": seq_len,
                                "window": WINDOW,
                                "image_suffix": IMAGE_SUFFIX,
                                "seed": seed,
                                "status": "ERROR",
                                "error": repr(e),
                            }

                        rows = remove_same_key(rows, "SRPM-CNN", dataset_name, seed)
                        rows.append(row)
                        save_experiment_tables(
                            rows,
                            OUTPUT_ROOT,
                            "SRPM-CNN",
                            group_cols=("dataset", "seq_len", "image_suffix", "window"),
                        )

                        job_elapsed = time.perf_counter() - job_start_time
                        job_times.append(job_elapsed)
                        completed_jobs += 1

                        script_elapsed = time.perf_counter() - script_start_time
                        avg_job_time = float(np.mean(job_times))

                        print(
                            f"[TIME] finished job | "
                            f"dataset={dataset_name} | seq_len={seq_len} | seed={seed} | "
                            f"job_time={format_seconds(job_elapsed)} | "
                            f"avg_job_time={format_seconds(avg_job_time)} | "
                            f"script_elapsed={format_seconds(script_elapsed)}"
                            f"{get_cuda_memory_text()}",
                            flush=True,
                        )

                finally:
                    del train_x, test_x
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                    seq_elapsed = time.perf_counter() - seq_start_time
                    print(
                        f"[SEQ DONE] dataset={dataset_name} | seq_len={seq_len} | "
                        f"seq_elapsed={format_seconds(seq_elapsed)}",
                        flush=True,
                    )

        except Exception as e:
            if should_retry_cuda_oom(e):
                torch.cuda.empty_cache()
                raise
            traceback.print_exc()
            for seed in SEEDS:
                rows = remove_same_key(rows, "SRPM-CNN", dataset_name, seed)
                rows.append({
                    "model": "SRPM-CNN",
                    "dataset": dataset_name,
                    "seed": seed,
                    "status": "ERROR",
                    "error": repr(e),
                })
            save_experiment_tables(
                rows,
                OUTPUT_ROOT,
                "SRPM-CNN",
                group_cols=("dataset", "seq_len", "image_suffix", "window"),
            )

        finally:
            dataset_elapsed = time.perf_counter() - dataset_start_time
            print(
                f"[DATASET DONE] dataset={dataset_name} | "
                f"dataset_elapsed={format_seconds(dataset_elapsed)}",
                flush=True,
            )

    script_total_time = time.perf_counter() - script_start_time
    print(
        f"\nDone. Use SRPM-CNN_summary_mean_std.csv for the paper table. "
        f"total_script_time={format_seconds(script_total_time)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
