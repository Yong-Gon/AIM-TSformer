import os
import json
import random
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

DEFAULT_DATASETS = [
    "ArticularyWordRecognition", "AtrialFibrillation", "BasicMotions", "Cricket", "DuckDuckGeese",
    "Epilepsy", "ERing", "EthanolConcentration", "FaceDetection", "FingerMovements",
    "HandMovementDirection", "Handwriting", "Heartbeat", "Libras", "LSST",
    "NATOPS", "PEMS-SF", "PenDigits", "PhonemeSpectra", "RacketSports",
    "SelfRegulationSCP1", "SelfRegulationSCP2", "StandWalkJump", "UWaveGestureLibrary",
]


def cuda_is_available() -> bool:
    if os.environ.get("AIM_FORCE_CPU", "").strip().lower() in {"1", "true", "yes"}:
        return False
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None and visible_devices.strip() in {"", "-1"}:
        return False
    return bool(torch.cuda.is_available())


def should_retry_cuda_oom(error: BaseException) -> bool:
    enabled = os.environ.get("AIM_RETRY_CUDA_OOM", "").strip().lower()
    if enabled not in {"1", "true", "yes"}:
        return False
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        "out of memory" in str(error).lower()
        and ("cuda" in str(error).lower() or cuda_is_available())
    )

def now_string() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if cuda_is_available() else "cpu")


def resolve_dataset_dir(base_parent: str, dataset_name: str, ts_type: str = "Multivariate_ts") -> Path:
    base = Path(base_parent)
    cand1 = base / ts_type / dataset_name
    cand2 = base / dataset_name
    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2
    raise FileNotFoundError(
        f"Dataset directory not found for {dataset_name}. Tried: {cand1} and {cand2}"
    )


def load_uea_ts_file(file_path: str, label_to_int: Optional[Dict[str, int]] = None):



    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(str(file_path))

    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    in_data = False
    records: List[List[List[float]]] = []
    labels: List[str] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("@"):
            if line.lower().startswith("@data"):
                in_data = True
            continue
        if not in_data:
            continue

        parts = line.split(":")
        if len(parts) < 2:
            continue
        *dim_strs, label_str = parts
        dims = []
        for ds in dim_strs:
            vals = []
            for v in ds.strip().split(","):
                v = v.strip()
                if v == "" or v == "?":
                    continue
                vals.append(float(v))
            dims.append(vals)
        records.append(dims)
        labels.append(label_str.strip())

    if not records:
        raise ValueError(f"No data records read from {file_path}")

    num_samples = len(records)
    num_dims = len(records[0])
    seq_len = len(records[0][0])
    for i, rec in enumerate(records):
        if len(rec) != num_dims:
            raise ValueError(f"Different number of dimensions at sample {i} in {file_path}")
        for c, arr in enumerate(rec):
            if len(arr) != seq_len:
                raise ValueError(
                    f"Variable-length series detected in {file_path}, sample={i}, channel={c}. "
                    "This 5-seed baseline runner assumes fixed-length .ts files."
                )

    X = np.zeros((num_samples, seq_len, num_dims), dtype=np.float32)
    for i in range(num_samples):
        for c in range(num_dims):
            X[i, :, c] = np.asarray(records[i][c], dtype=np.float32)

    labels_np = np.asarray(labels)
    if label_to_int is None:
        uniq = sorted(np.unique(labels_np))
        label_to_int = {lab: idx for idx, lab in enumerate(uniq)}
    y = np.asarray([label_to_int[lab] for lab in labels_np], dtype=np.int64)
    return X, y, label_to_int


def load_uea_train_test(base_parent: str, dataset_name: str, ts_type: str = "Multivariate_ts"):
    dpath = resolve_dataset_dir(base_parent, dataset_name, ts_type=ts_type)
    train_path = dpath / f"{dataset_name}_TRAIN.ts"
    test_path = dpath / f"{dataset_name}_TEST.ts"
    X_train, y_train, label_map = load_uea_ts_file(train_path, label_to_int=None)
    X_test, y_test, _ = load_uea_ts_file(test_path, label_to_int=label_map)
    return X_train, y_train, X_test, y_test, label_map, dpath


def _paa_downsample(X: np.ndarray, desired_length: int) -> np.ndarray:
    n, cur = X.shape
    if desired_length == cur:
        return X.copy()
    window = cur / desired_length
    out = np.empty((n, desired_length), dtype=X.dtype)
    for i in range(desired_length):
        s = int(round(i * window))
        e = int(round((i + 1) * window))
        e = min(e, cur)
        out[:, i] = X[:, s:e].mean(axis=1) if e > s else X[:, min(s, cur - 1)]
    return out


def adjust_sequence_length(data: np.ndarray, desired_length: Optional[int], upsample_method: str = "linear") -> np.ndarray:
    if desired_length is None:
        return data.copy()
    N, L, C = data.shape
    if L == desired_length:
        return data.copy()
    out = np.empty((N, desired_length, C), dtype=np.float32)
    x_old = np.arange(L)
    x_new = np.linspace(0, L - 1, desired_length)
    for c in range(C):
        X = data[:, :, c]
        if L > desired_length:
            out[:, :, c] = _paa_downsample(X, desired_length)
        else:
            if upsample_method == "linear":
                out[:, :, c] = np.vstack([np.interp(x_new, x_old, row) for row in X])
            elif upsample_method == "step":
                idx = np.rint(np.linspace(0, L - 1, desired_length)).astype(int)
                out[:, :, c] = X[:, idx]
            else:
                raise ValueError(f"Unknown upsample_method={upsample_method}")
    return out


def z_normalize_by_train(X_train: np.ndarray, X_test: np.ndarray):
    mean = np.nanmean(X_train, axis=(0, 1), keepdims=True)
    std = np.nanstd(X_train, axis=(0, 1), keepdims=True)
    X_train_n = (X_train - mean) / (std + 1e-6)
    X_test_n = (X_test - mean) / (std + 1e-6)
    X_train_n = np.nan_to_num(X_train_n, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_test_n = np.nan_to_num(X_test_n, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return X_train_n, X_test_n


def fit_channel_minmax(train_img: np.ndarray):
    if train_img.ndim == 3:
        train_img = train_img[:, :, :, None]
    c = train_img.shape[-1]
    mins = np.zeros(c, dtype=np.float32)
    maxs = np.zeros(c, dtype=np.float32)
    for i in range(c):
        ch = train_img[..., i].astype(np.float32)
        mins[i] = np.nanmin(ch)
        maxs[i] = np.nanmax(ch)
    return mins, maxs


def apply_channel_minmax(img: np.ndarray, mins: np.ndarray, maxs: np.ndarray, scale: float = 255.0):
    if img.ndim == 3:
        img = img[:, :, :, None]
    out = np.empty_like(img, dtype=np.float32)
    for i in range(img.shape[-1]):
        den = max(float(maxs[i] - mins[i]), 1e-8)
        out[..., i] = scale * (img[..., i].astype(np.float32) - mins[i]) / den
    return np.nan_to_num(out, nan=0.0, posinf=scale, neginf=0.0).astype(np.float32)


class ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ImageDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        if X.ndim == 3:
            X = X[:, :, :, None]
        self.X = torch.from_numpy(X.transpose(0, 3, 1, 2).astype(np.float32)).float()
        self.y = torch.from_numpy(y.astype(np.int64)).long()
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int, num_workers: int = 0):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        generator=generator if shuffle else None,
        pin_memory=cuda_is_available(),
    )


def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device,
                   forward_fn: Optional[Callable] = None):
    model.eval()
    total_loss = 0.0
    all_pred = []
    all_y = []
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = forward_fn(model, xb) if forward_fn is not None else model(xb)
            loss = criterion(logits, yb)
            total_loss += float(loss.item()) * xb.size(0)
            all_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
            all_y.extend(yb.detach().cpu().numpy().tolist())
    acc = accuracy_score(all_y, all_pred) if all_y else float("nan")
    macro_f1 = (
        f1_score(all_y, all_pred, average="macro", zero_division=0)
        if all_y else float("nan")
    )
    balanced_acc = (
        balanced_accuracy_score(all_y, all_pred) if all_y else float("nan")
    )
    loss = total_loss / max(1, len(loader.dataset))
    return loss, acc, macro_f1, balanced_acc


def train_fixed_epochs(model: torch.nn.Module,
                       train_loader: DataLoader,
                       test_loader: DataLoader,
                       device: torch.device,
                       epochs: int,
                       optimizer: torch.optim.Optimizer,
                       criterion: Optional[torch.nn.Module] = None,
                       forward_fn: Optional[Callable] = None,
                       scheduler=None,
                       grad_clip_norm: Optional[float] = None,
                       print_every: int = 1,
                       tag: str = "",
                       evaluate_test_every_epoch: bool = False):



    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        preds, labels = [], []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = forward_fn(model, xb) if forward_fn is not None else model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            total_loss += float(loss.item()) * xb.size(0)
            preds.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
            labels.extend(yb.detach().cpu().numpy().tolist())
        train_loss = total_loss / max(1, len(train_loader.dataset))
        train_acc = accuracy_score(labels, preds) if labels else float("nan")
        should_evaluate_test = evaluate_test_every_epoch or epoch == epochs
        if should_evaluate_test:
            test_loss, test_acc, test_macro_f1, test_balanced_acc = evaluate_model(
                model, test_loader, device, forward_fn=forward_fn
            )
        else:
            test_loss = test_acc = test_macro_f1 = test_balanced_acc = float("nan")
        if scheduler is not None:
            try:
                scheduler.step(test_loss if should_evaluate_test else train_loss)
            except TypeError:
                scheduler.step()
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_macro_f1": test_macro_f1,
            "test_balanced_acc": test_balanced_acc,
        })
        if print_every and (epoch == 1 or epoch == epochs or epoch % print_every == 0):
            test_text = f" | test_acc={test_acc:.4f}" if should_evaluate_test else ""
            print(
                f"{tag} Ep {epoch:03d}/{epochs} | train_loss={train_loss:.4f} "
                f"| train_acc={train_acc:.4f}{test_text}"
            )
    return history


def summarize_seed_results(per_seed_df: pd.DataFrame,
                           group_cols: Sequence[str] = ("dataset",),
                           metric_col: str = "final_test_acc") -> pd.DataFrame:
    ok = per_seed_df[per_seed_df["status"].eq("OK")].copy()
    if ok.empty:
        return pd.DataFrame(columns=list(group_cols) + ["n", "mean_acc", "std_acc", "mean_pm_std"])
    metric_specs = [
        ("final_test_acc", "acc"),
        ("final_test_macro_f1", "macro_f1"),
        ("final_test_balanced_acc", "balanced_acc"),
    ]
    summary = None
    for column, short_name in metric_specs:
        if column not in ok.columns:
            continue
        part = ok.groupby(list(group_cols))[column].agg(["count", "mean", "std"]).reset_index()
        part = part.rename(
            columns={
                "count": "n" if short_name == "acc" else f"n_{short_name}",
                "mean": f"mean_{short_name}",
                "std": f"std_{short_name}",
            }
        )
        std_col = f"std_{short_name}"
        part[std_col] = part[std_col].fillna(0.0)
        part[f"{short_name}_mean_pm_std"] = part.apply(
            lambda row: f"{row[f'mean_{short_name}']:.3f} ± {row[std_col]:.3f}",
            axis=1,
        )
        summary = part if summary is None else summary.merge(part, on=list(group_cols), how="outer")
    if summary is None:
        return pd.DataFrame(columns=list(group_cols))
    if "acc_mean_pm_std" in summary.columns:
        summary["mean_pm_std"] = summary["acc_mean_pm_std"]
    return summary


def load_existing_rows(path: Path) -> List[Dict]:
    if not path.is_file():
        return []
    return pd.read_csv(path).to_dict("records")


def row_already_ok(rows: Sequence[Dict], model_name: str, dataset: str, seed: int) -> bool:
    for row in rows:
        if (
            str(row.get("model", "")) == model_name
            and str(row.get("dataset", "")) == dataset
            and int(row.get("seed", -1)) == int(seed)
            and str(row.get("status", "")) == "OK"
            and pd.notna(row.get("final_test_macro_f1"))
            and pd.notna(row.get("final_test_balanced_acc"))
        ):
            return True
    return False


def remove_same_key(rows: Sequence[Dict], model_name: str, dataset: str, seed: int) -> List[Dict]:
    return [
        row for row in rows
        if not (
            str(row.get("model", "")) == model_name
            and str(row.get("dataset", "")) == dataset
            and int(row.get("seed", -1)) == int(seed)
        )
    ]


def save_experiment_tables(rows: List[Dict], output_dir: Path, model_name: str,
                           group_cols: Sequence[str] = ("dataset",)):
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed = pd.DataFrame(rows)
    per_seed_csv = output_dir / f"{model_name}_per_seed_results.csv"
    summary_csv = output_dir / f"{model_name}_summary_mean_std.csv"
    per_seed.to_csv(per_seed_csv, index=False, encoding="utf-8-sig")
    summary = summarize_seed_results(per_seed, group_cols=group_cols)
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"Saved: {per_seed_csv}")
    print(f"Saved: {summary_csv}")
    return per_seed, summary


def load_labels_from_ts_with_train_mapping(dataset_dir: Path, dataset_name: str):
    _, y_train, _, y_test, label_map, _ = load_uea_train_test(str(dataset_dir.parent.parent), dataset_name)
    return y_train, y_test, label_map
