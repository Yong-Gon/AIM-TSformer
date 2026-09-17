from __future__ import annotations

import json
import os
import random
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd



NUMBA_CACHE_DIR = Path(os.environ.get(
    "AIM_NUMBA_CACHE_DIR", str(Path(__file__).resolve().parent / ".numba_cache")
))
NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))


DEFAULT_DATASETS: List[str] = [
    "ArticularyWordRecognition",
    "BasicMotions",
    "Epilepsy",
    "ERing",
    "FaceDetection",
    "FingerMovements",
    "Handwriting",
    "Libras",
    "LSST",
    "NATOPS",
    "PEMS-SF",
    "PenDigits",
    "PhonemeSpectra",
    "RacketSports",
    "DuckDuckGeese",
    "UWaveGestureLibrary",
    "AtrialFibrillation",
    "Cricket",
    "EthanolConcentration",
    "HandMovementDirection",
    "Heartbeat",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "StandWalkJump",
]

ABBREVIATIONS: Dict[str, str] = {
    "ArticularyWordRecognition": "AWR",
    "BasicMotions": "BM",
    "Epilepsy": "EP",
    "ERing": "ER",
    "FaceDetection": "FD",
    "FingerMovements": "FM",
    "Handwriting": "HW",
    "Libras": "LIB",
    "LSST": "LSST",
    "NATOPS": "NA",
    "PEMS-SF": "PEMS",
    "PenDigits": "PD",
    "PhonemeSpectra": "PM",
    "RacketSports": "RS",
    "DuckDuckGeese": "DDG",
    "UWaveGestureLibrary": "UWG",
    "AtrialFibrillation": "AF",
    "Cricket": "CR",
    "EthanolConcentration": "EC",
    "HandMovementDirection": "HMD",
    "Heartbeat": "HB",
    "SelfRegulationSCP1": "SRS1",
    "SelfRegulationSCP2": "SRS2",
    "StandWalkJump": "SWJ",
}


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    os.environ["PYTHONHASHSEED"] = str(int(seed))


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def format_seconds(seconds: float) -> str:
    seconds = int(max(0, round(float(seconds))))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def _case_insensitive_file_search(folder: Path, filenames: Sequence[str]) -> Optional[Path]:
    if not folder.exists() or not folder.is_dir():
        return None
    targets = {name.lower(): name for name in filenames}
    for child in folder.iterdir():
        if child.is_file() and child.name.lower() in targets:
            return child
    return None


def resolve_dataset_dir(base_parent: str | Path, dataset_name: str, ts_type: str = "Multivariate_ts") -> Path:
    base = Path(base_parent)
    candidates = [
        base / ts_type / dataset_name,
        base / dataset_name,
        base / ts_type,
        base,
    ]

    train_names = [f"{dataset_name}_TRAIN.ts", f"{dataset_name}_train.ts"]
    test_names = [f"{dataset_name}_TEST.ts", f"{dataset_name}_test.ts"]

    for cand in candidates:
        train = _case_insensitive_file_search(cand, train_names)
        test = _case_insensitive_file_search(cand, test_names)
        if train is not None and test is not None:
            return cand

    
    if base.exists():
        train_matches = []
        test_matches = []
        train_lowers = {name.lower() for name in train_names}
        test_lowers = {name.lower() for name in test_names}
        for p in base.rglob("*.ts"):
            lower = p.name.lower()
            if lower in train_lowers:
                train_matches.append(p)
            elif lower in test_lowers:
                test_matches.append(p)
        for train_path in train_matches:
            for test_path in test_matches:
                if train_path.parent == test_path.parent:
                    return train_path.parent

    raise FileNotFoundError(
        "Could not locate UEA .ts files for dataset "
        f"{dataset_name!r} under BASE_PARENT={str(base)!r}. "
        f"Expected files like {dataset_name}_TRAIN.ts and {dataset_name}_TEST.ts."
    )


def find_train_test_ts_files(base_parent: str | Path, dataset_name: str, ts_type: str = "Multivariate_ts") -> Tuple[Path, Path, Path]:
    dpath = resolve_dataset_dir(base_parent, dataset_name, ts_type=ts_type)
    train = _case_insensitive_file_search(dpath, [f"{dataset_name}_TRAIN.ts", f"{dataset_name}_train.ts"])
    test = _case_insensitive_file_search(dpath, [f"{dataset_name}_TEST.ts", f"{dataset_name}_test.ts"])
    if train is None or test is None:
        raise FileNotFoundError(f"Resolved {dpath}, but train/test .ts files were not found for {dataset_name}.")
    return train, test, dpath


def _import_ts_loader():
    try:
        from sktime.datasets import load_from_tsfile_to_dataframe
        return load_from_tsfile_to_dataframe, "sktime.datasets.load_from_tsfile_to_dataframe"
    except Exception:
        pass

    try:
        from aeon.utils.data_io import load_from_tsfile_to_dataframe
        return load_from_tsfile_to_dataframe, "aeon.utils.data_io.load_from_tsfile_to_dataframe"
    except Exception as e:
        raise ImportError(
            "Neither sktime nor aeon could be imported for UEA .ts loading. "
            "Install at least one of them, e.g. pip install sktime aeon."
        ) from e


def encode_labels(y_train: Sequence, y_test: Sequence) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    classes = list(pd.unique(y_train))
    label_map = {str(cls): i for i, cls in enumerate(classes)}
    y_train_enc = np.asarray([label_map[str(v)] for v in y_train], dtype=np.int64)
    unknown = sorted({str(v) for v in y_test} - set(label_map))
    if unknown:
        raise ValueError(f"TEST labels absent from TRAIN: {unknown}")
    y_test_enc = np.asarray([label_map[str(v)] for v in y_test], dtype=np.int64)
    return y_train_enc, y_test_enc, label_map


def nested_dataframe_to_numpy3d(X_nested) -> np.ndarray:



    if isinstance(X_nested, np.ndarray):
        arr = np.asarray(X_nested)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D numpy array, got shape={arr.shape}")
        
        return arr.astype(np.float32, copy=False)

    if not hasattr(X_nested, "shape"):
        raise TypeError(f"Unsupported X type: {type(X_nested)}")

    n_cases, n_channels = X_nested.shape
    series0 = np.asarray(X_nested.iloc[0, 0], dtype=np.float32)
    length = int(series0.shape[0])
    out = np.empty((n_cases, n_channels, length), dtype=np.float32)

    for i in range(n_cases):
        for c in range(n_channels):
            vals = np.asarray(X_nested.iloc[i, c], dtype=np.float32)
            if vals.ndim != 1:
                vals = vals.reshape(-1)
            if vals.shape[0] != length:
                raise ValueError(
                    f"Unequal length detected at row={i}, channel={c}. "
                    f"Expected {length}, got {vals.shape[0]}. These runners assume fixed length."
                )
            out[i, c, :] = vals
    return out


def _paa_downsample_2d(X: np.ndarray, desired_length: int) -> np.ndarray:
    n_rows, current_length = X.shape
    if current_length == desired_length:
        return X.copy()
    window = current_length / desired_length
    out = np.empty((n_rows, desired_length), dtype=np.float32)
    for i in range(desired_length):
        start = int(round(i * window))
        end = int(round((i + 1) * window))
        end = min(end, current_length)
        if end > start:
            out[:, i] = X[:, start:end].mean(axis=1)
        else:
            out[:, i] = X[:, min(start, current_length - 1)]
    return out


def adjust_length_numpy3d(X_ncl: np.ndarray, desired_length: Optional[int], upsample_method: str = "linear") -> np.ndarray:
    X_ncl = np.asarray(X_ncl, dtype=np.float32)
    if desired_length is None:
        return X_ncl
    desired_length = int(desired_length)
    n, c, current_length = X_ncl.shape
    if current_length == desired_length:
        return X_ncl.copy()

    X_flat = X_ncl.reshape(n * c, current_length)
    if current_length > desired_length:
        out_flat = _paa_downsample_2d(X_flat, desired_length)
    else:
        x_old = np.arange(current_length)
        x_new = np.linspace(0, current_length - 1, desired_length)
        out_flat = np.empty((n * c, desired_length), dtype=np.float32)
        if upsample_method == "linear":
            for r in range(n * c):
                out_flat[r] = np.interp(x_new, x_old, X_flat[r]).astype(np.float32)
        elif upsample_method == "step":
            idx = np.rint(np.linspace(0, current_length - 1, desired_length)).astype(int)
            out_flat = X_flat[:, idx].astype(np.float32, copy=False)
        else:
            raise ValueError(f"Unknown UPSAMPLE_METHOD={upsample_method!r}")
    return out_flat.reshape(n, c, desired_length).astype(np.float32, copy=False)


def z_normalize_by_train_numpy3d(
    X_train_ncl: np.ndarray,
    X_test_ncl: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:




    X_train_ncl = np.asarray(X_train_ncl, dtype=np.float32)
    X_test_ncl = np.asarray(X_test_ncl, dtype=np.float32)
    mean = X_train_ncl.mean(axis=(0, 2), keepdims=True)
    std = X_train_ncl.std(axis=(0, 2), keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return ((X_train_ncl - mean) / std).astype(np.float32), ((X_test_ncl - mean) / std).astype(np.float32)


def load_uea_numpy3d(
    base_parent: str | Path,
    dataset_name: str,
    ts_type: str = "Multivariate_ts",
    fixed_length: Optional[int] = None,
    upsample_method: str = "linear",
    normalize_by_train: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int], Path, str]:


    train_file, test_file, dpath = find_train_test_ts_files(base_parent, dataset_name, ts_type=ts_type)
    loader, loader_name = _import_ts_loader()
    X_train_nested, y_train_raw = loader(str(train_file))
    X_test_nested, y_test_raw = loader(str(test_file))

    y_train, y_test, label_map = encode_labels(y_train_raw, y_test_raw)
    X_train = nested_dataframe_to_numpy3d(X_train_nested)
    X_test = nested_dataframe_to_numpy3d(X_test_nested)

    X_train = adjust_length_numpy3d(X_train, fixed_length, upsample_method=upsample_method)
    X_test = adjust_length_numpy3d(X_test, fixed_length, upsample_method=upsample_method)

    if normalize_by_train:
        X_train, X_test = z_normalize_by_train_numpy3d(X_train, X_test)

    return X_train, y_train, X_test, y_test, label_map, dpath, loader_name


def get_existing_rows(per_seed_csv: Path) -> List[dict]:
    if per_seed_csv.exists():
        try:
            df = pd.read_csv(per_seed_csv)
            return df.to_dict("records")
        except Exception:
            traceback.print_exc()
            print(f"[WARN] Could not read existing CSV: {per_seed_csv}. Starting with empty rows.", flush=True)
    return []


def row_already_ok(rows: Sequence[dict], model_name: str, dataset_name: str, seed: int) -> bool:
    for row in rows:
        if str(row.get("model", "")) != str(model_name):
            continue
        if str(row.get("dataset", "")) != str(dataset_name):
            continue
        try:
            same_seed = int(row.get("seed")) == int(seed)
        except Exception:
            same_seed = str(row.get("seed", "")) == str(seed)
        has_metrics = all(
            pd.notna(row.get(column))
            for column in (
                "final_test_acc",
                "final_test_macro_f1",
                "final_test_balanced_acc",
            )
        )
        if same_seed and str(row.get("status", "")) == "OK" and has_metrics:
            return True
    return False


def remove_same_key(rows: Sequence[dict], model_name: str, dataset_name: str, seed: int) -> List[dict]:
    kept = []
    for row in rows:
        same = str(row.get("model", "")) == str(model_name) and str(row.get("dataset", "")) == str(dataset_name)
        if same:
            try:
                same = same and int(row.get("seed")) == int(seed)
            except Exception:
                same = same and str(row.get("seed", "")) == str(seed)
        if not same:
            kept.append(row)
    return kept


def _safe_float(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def save_experiment_tables(rows: Sequence[dict], output_root: str | Path, model_name: str) -> None:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(rows))
    if df.empty:
        return

    per_seed_path = output_root / f"{model_name}_per_seed_results.csv"
    df.to_csv(per_seed_path, index=False, encoding="utf-8-sig")

    
    ok = df[df.get("status", "") == "OK"].copy() if "status" in df.columns else df.copy()
    if ok.empty or "dataset" not in ok.columns:
        return

    for col in [
        "final_test_acc", "final_test_macro_f1", "final_test_balanced_acc",
        "fit_seconds", "predict_seconds", "total_seconds",
    ]:
        if col in ok.columns:
            ok[col] = ok[col].map(_safe_float)

    agg_dict = {}
    for col in [
        "final_test_acc", "final_test_macro_f1", "final_test_balanced_acc",
        "fit_seconds", "predict_seconds", "total_seconds",
    ]:
        if col in ok.columns:
            agg_dict[col] = ["mean", "std", "count"]

    if not agg_dict:
        return

    summary = ok.groupby("dataset", dropna=False).agg(agg_dict)
    summary.columns = ["_".join([str(a), str(b)]).strip("_") for a, b in summary.columns]
    summary = summary.reset_index()

    
    meta_cols = ["model", "abbr", "seq_len", "channels", "num_classes", "n_train", "n_test"]
    meta_cols = [c for c in meta_cols if c in ok.columns]
    if meta_cols:
        meta = ok.groupby("dataset", dropna=False)[meta_cols].first().reset_index()
        summary = meta.merge(summary, on="dataset", how="left")

    if "final_test_acc_mean" in summary.columns:
        summary = summary.sort_values(["final_test_acc_mean", "dataset"], ascending=[False, True])

    for column, label in (
        ("final_test_acc", "accuracy"),
        ("final_test_macro_f1", "macro_f1"),
        ("final_test_balanced_acc", "balanced_accuracy"),
    ):
        mean_col = f"{column}_mean"
        std_col = f"{column}_std"
        if mean_col in summary.columns and std_col in summary.columns:
            summary[f"{label}_mean_pm_std"] = summary.apply(
                lambda row, m=mean_col, s=std_col: f"{row[m]:.3f} ± {row[s]:.3f}",
                axis=1,
            )

    summary_path = output_root / f"{model_name}_summary_mean_std.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    
    manifest = {
        "model": model_name,
        "updated_at": now_text(),
        "num_rows": int(len(df)),
        "num_ok_rows": int(len(ok)),
        "per_seed_csv": str(per_seed_path),
        "summary_csv": str(summary_path),
    }
    with open(output_root / f"{model_name}_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def make_base_result_row(
    model_name: str,
    dataset_name: str,
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_map: Dict[str, int],
    dpath: Path,
    loader_name: str,
    fixed_length: Optional[int],
    normalize_by_train: bool,
) -> dict:
    return {
        "model": model_name,
        "dataset": dataset_name,
        "abbr": ABBREVIATIONS.get(dataset_name, dataset_name),
        "seed": int(seed),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "channels": int(X_train.shape[1]),
        "seq_len": int(X_train.shape[2]),
        "num_classes": int(len(label_map)),
        "fixed_length": "None" if fixed_length is None else int(fixed_length),
        "normalize_by_train": bool(normalize_by_train),
        "dataset_dir": str(dpath),
        "loader": loader_name,
    }
