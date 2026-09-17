from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dataset_directories(dataset_root: Path, ts_type: str, names: Iterable[str] | None):
    base = dataset_root / ts_type
    if not base.is_dir():
        raise FileNotFoundError(base)
    requested = set(names or [])
    paths = sorted(path for path in base.iterdir() if path.is_dir())
    if requested:
        found = {path.name for path in paths}
        missing = sorted(requested - found)
        if missing:
            raise FileNotFoundError(f"Dataset folders not found: {missing}")
        paths = [path for path in paths if path.name in requested]
    return paths


def hpo_sweep(*, patch_grid: int = 4, lambda_contrast: float = 1.0):
    return {
        "stages": [1],
        "patch_grid": [patch_grid],
        "d_model": [[128]],
        "dense_units": [128],
        "mha_heads": [[8]],
        "mha_modules": [1, 2, 4],
        "backbone_depth": [1, 2, 4],
        "dropout": [0.1],
        "lr": [0.0005],
        "temperature": [1.0],
        "label_smoothing": [0.0],
        "optimizer": ["adamw"],
        "w_main": [1.0],
        "w_ts": [1.0],
        "w_img": [1.0],
        "lambda_contrast": [lambda_contrast],
    }


def model_args(
    cli,
    *,
    output_root: Path,
    datasets: Iterable[str],
    preprocess_mode: str,
    image_type: str,
):
    return SimpleNamespace(
        dataset_root=str(cli.dataset_root),
        ts_type=cli.ts_type,
        ts_preprocess_mode=preprocess_mode,
        sequence_lengths=None,
        output_root=str(output_root),
        datasets_to_run=list(datasets),
        datasets_to_skip=[],
        backbone="convmixer",
        conv_dw_kernel=9,
        img_types=[image_type],
        max_inner=None,
        eval_source="fusion",
        use_gated_fusion=True,
        fusion_gate_init=1.0,
        symmetric_fusion_norm=True,
        modality_dropout=0.1,
        use_contrast_projection=True,
        use_segment_embedding=True,
        seeds=list(cli.seeds),
        epochs=cli.epochs,
        batch_size=cli.batch_size,
        label_smoothing=0.0,
        temperature=1.0,
        early_patience=cli.early_patience,
        early_stop_metric="train_loss",
        optimizer="adamw",
        weight_decay=cli.weight_decay,
        use_amp=not cli.no_amp,
    )
