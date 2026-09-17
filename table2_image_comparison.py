from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace


IMAGE_TYPES = ("gasf", "gadf", "mtf", "rp")


def load_public_module(path: Path):
    specification = importlib.util.spec_from_file_location("aim_tsformer_public", path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-file", type=Path, default=here / "AIM_TSformer.py")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ.get("AIM_DATASET_ROOT", here / "dataset")),
    )
    parser.add_argument("--ts-type", default="Multivariate_ts")
    parser.add_argument("--output-root", type=Path, default=here / "results_baseline_images")
    parser.add_argument("--image-types", nargs="+", choices=IMAGE_TYPES, default=list(IMAGE_TYPES))
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--early-patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    public = load_public_module(cli.model_file)
    public.USE_AMP = not cli.no_amp

    
    sweep = {
        "stages": [1],
        "patch_grid": [4],
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
        "lambda_contrast": [1.0],
    }
    experiment = SimpleNamespace(
        dataset_root=str(cli.dataset_root),
        ts_type=cli.ts_type,
        ts_preprocess_mode="original",
        output_root=str(cli.output_root),
        datasets_to_run=list(cli.datasets) if cli.datasets else None,
        datasets_to_skip=[],
        backbone="convmixer",
        conv_dw_kernel=9,
        img_types=list(cli.image_types),
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
    public.main_grid(experiment, sweep)


if __name__ == "__main__":
    main()
