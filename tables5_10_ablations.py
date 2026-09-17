from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import pickle
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable


TABLE5_VARIANTS = (
    "original",
    "angle_only",
    "sign_only",
    "diagonal_only",
    "symmetric",
    "flat_as_positive",
)
TABLE6_MODES = (
    "baseline",
    "shifted_alignment",
    "random_alignment",
    "no_contrast",
)
TABLE7_MODES = ("baseline", "grid_size", "overlapping", "diagonal_band")
TABLE9_MODES = ("global_infonce", "baseline", "batch_all", "class_aware")
TABLE10_LENGTHS = ("native", 224, 384)


def load_module(path: Path, module_name: str):
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def image_type(preprocess_mode: str, variant: str) -> str:
    return f"aim_{preprocess_mode.lower()}_{variant}"


def base_sweep(*, eval_source: str = "fusion", lambda_contrast: float = 1.0) -> dict:
    del eval_source  
    return {
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
        "lambda_contrast": [lambda_contrast],
    }


def make_core_args(
    cli: argparse.Namespace,
    output_root: Path,
    image_types: Iterable[str],
    *,
    eval_source: str = "fusion",
    sequence_lengths=None,
) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_root=str(cli.dataset_root),
        ts_type=cli.ts_type,
        ts_preprocess_mode=cli.preprocess_mode,
        sequence_lengths=sequence_lengths,
        output_root=str(output_root),
        datasets_to_run=list(cli.datasets) if cli.datasets else None,
        datasets_to_skip=[],
        backbone="convmixer",
        conv_dw_kernel=9,
        img_types=list(image_types),
        max_inner=None,
        eval_source=eval_source,
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


def selected_dataset_dirs(cli: argparse.Namespace) -> list[Path]:
    base = cli.dataset_root / cli.ts_type
    if not base.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {base}. Pass --dataset-root or set AIM_DATASET_ROOT."
        )
    requested = set(cli.datasets or [])
    directories = sorted(path for path in base.iterdir() if path.is_dir())
    if requested:
        found = {path.name for path in directories}
        missing = sorted(requested - found)
        if missing:
            raise FileNotFoundError(f"Dataset folders not found: {missing}")
        directories = [path for path in directories if path.name in requested]
    return directories


def prepare_images(cli: argparse.Namespace, dataset_dirs: list[Path]) -> None:
    generator = load_module(cli.generator_file, "aim_public_generator_tables_5_10")
    generation_args = SimpleNamespace(
        engine=cli.generation_engine,
        device=cli.generation_device,
        batch_size=cli.generation_batch_size,
        overwrite=cli.overwrite_images,
    )
    if generation_args.engine == "torch" and generation_args.device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            print("[WARN] CUDA is unavailable for AIM generation; using CPU.")
            generation_args.device = "cpu"

    if 5 in cli.tables:
        for dataset_dir in dataset_dirs:
            generator.process_dataset(
                dataset_dir,
                [cli.preprocess_mode],
                TABLE5_VARIANTS,
                None,
                generation_args,
            )

    if 10 in cli.tables:
        for dataset_dir in dataset_dirs:
            
            generator.process_dataset(
                dataset_dir,
                [cli.preprocess_mode],
                ["original"],
                None,
                generation_args,
            )
            generator.process_dataset(
                dataset_dir,
                [cli.preprocess_mode],
                ["original"],
                [224, 384],
                generation_args,
            )


def expected_image_lengths(dataset_dir: Path, preprocess_mode: str, variant: str) -> list[int]:
    prefix = f"{dataset_dir.name}_{image_type(preprocess_mode, variant)}_train_"
    result = []
    for path in dataset_dir.glob(f"{prefix}*.npy"):
        suffix = path.stem.removeprefix(prefix)
        if suffix.isdigit() and (dataset_dir / path.name.replace("_train_", "_test_")).is_file():
            result.append(int(suffix))
    return sorted(set(result))


def validate_inputs(cli: argparse.Namespace, dataset_dirs: list[Path]) -> None:
    rows = []
    for dataset_dir in dataset_dirs:
        split_root = dataset_dir / "preprocessed" / cli.preprocess_mode
        train_pkl = split_root / f"{dataset_dir.name}_train_df.pkl"
        test_pkl = split_root / f"{dataset_dir.name}_test_df.pkl"
        native_length = None
        if train_pkl.is_file():
            with train_pkl.open("rb") as handle:
                native_length = int(pickle.load(handle).shape[1])
        row = {
            "dataset": dataset_dir.name,
            "preprocessed_train": str(train_pkl),
            "preprocessed_test": str(test_pkl),
            "preprocessed_pair_exists": train_pkl.is_file() and test_pkl.is_file(),
            "native_length": native_length,
            "images": {},
        }
        if 5 in cli.tables:
            row["images"]["table5"] = {
                variant: expected_image_lengths(dataset_dir, cli.preprocess_mode, variant)
                for variant in TABLE5_VARIANTS
            }
        if any(table in cli.tables for table in (6, 7, 8, 9, 10)):
            row["images"]["original"] = expected_image_lengths(
                dataset_dir, cli.preprocess_mode, "original"
            )
        rows.append(row)

    errors = []
    for row in rows:
        if not row["preprocessed_pair_exists"]:
            errors.append(f"{row['dataset']}: missing preprocessed {cli.preprocess_mode} split")
            continue
        if 5 in cli.tables:
            for variant, lengths in row["images"].get("table5", {}).items():
                if row["native_length"] not in lengths:
                    errors.append(
                        f"{row['dataset']}: Table-5 variant {variant} is missing "
                        f"native length {row['native_length']}"
                    )
        original_lengths = row["images"].get("original", [])
        if any(table in cli.tables for table in (6, 7, 8, 9)) and row["native_length"] not in original_lengths:
            errors.append(
                f"{row['dataset']}: original AIM is missing native length {row['native_length']}"
            )
        if 10 in cli.tables:
            required = {row["native_length"], 224, 384}
            if not required.issubset(set(original_lengths)):
                errors.append(
                    f"{row['dataset']}: Table-10 images must include lengths {sorted(required)}"
                )
    if errors:
        raise FileNotFoundError("Input validation failed:\n- " + "\n- ".join(errors))


def run_patch_modes(cli: argparse.Namespace, table: int, modes: Iterable[str]) -> None:
    output_root = cli.output_root / f"table{table}"
    command = [
        sys.executable,
        str(cli.patch_runner),
        "--model-file",
        str(cli.model_file),
        "--dataset-root",
        str(cli.dataset_root),
        "--ts-type",
        cli.ts_type,
        "--output-root",
        str(output_root),
        "--modes",
        *modes,
        "--image-type",
        image_type(cli.preprocess_mode, "original"),
        "--preprocess-mode",
        cli.preprocess_mode,
        "--seeds",
        *(str(seed) for seed in cli.seeds),
        "--epochs",
        str(cli.epochs),
        "--early-patience",
        str(cli.early_patience),
        "--batch-size",
        str(cli.batch_size),
        "--weight-decay",
        str(cli.weight_decay),
    ]
    if cli.datasets:
        command.extend(["--datasets", *cli.datasets])
    if cli.no_amp:
        command.append("--no-amp")
    print("[RUN]", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def update_selected_hparams(root: Path, additions: dict) -> None:
    scan_root = Path("\\\\?\\" + str(root.resolve())) if os.name == "nt" else root
    for path in scan_root.rglob("best_hparams.json"):
        values = json.loads(path.read_text(encoding="utf-8"))
        values.update(additions)
        path.write_text(json.dumps(values, indent=2), encoding="utf-8")


def run_table8(public, cli: argparse.Namespace) -> None:
    conditions = {
        "only_ts": {"eval_source": "ts", "weights": (0.0, 1.0, 0.0), "lambda": 0.0},
        "only_img": {"eval_source": "img", "weights": (0.0, 0.0, 1.0), "lambda": 0.0},
        "without_contrast": {"eval_source": "fusion", "weights": (1.0, 1.0, 1.0), "lambda": 0.0},
        "full": {"eval_source": "fusion", "weights": (1.0, 1.0, 1.0), "lambda": 1.0},
    }
    for name, condition in conditions.items():
        root = cli.output_root / "table8" / name
        sweep = base_sweep(lambda_contrast=condition["lambda"])
        sweep["w_main"] = [condition["weights"][0]]
        sweep["w_ts"] = [condition["weights"][1]]
        sweep["w_img"] = [condition["weights"][2]]
        args = make_core_args(
            cli,
            root,
            [image_type(cli.preprocess_mode, "original")],
            eval_source=condition["eval_source"],
            sequence_lengths=["native"],
        )
        public.main_grid(args, sweep)
        update_selected_hparams(root, {"table": 8, "table_condition": name})


def condition_for_result(table_dir: str, hp: dict, metrics: dict, path: Path) -> str:
    if table_dir == "table5":
        return str(hp.get("img_type", "unknown"))
    if table_dir in {"table6", "table7", "table9"}:
        mode = str(hp.get("ablation_mode", path.parts[-6] if len(path.parts) >= 6 else "unknown"))
        if mode == "grid_size":
            return f"grid_size_S{hp.get('patch_grid_ablation', hp.get('patch_grid', 'unknown'))}"
        return mode
    if table_dir == "table8":
        return str(hp.get("table_condition", path.parts[-6] if len(path.parts) >= 6 else "unknown"))
    if table_dir == "table10":
        return str(metrics.get("seq_len", hp.get("seq_len", "unknown")))
    return "unknown"


def aggregate_results(output_root: Path, expected_seeds: Iterable[int]) -> tuple[Path, Path]:
    expected_seed_set = {int(seed) for seed in expected_seeds}
    rows = []
    scan_root = Path("\\\\?\\" + str(output_root.resolve())) if os.name == "nt" else output_root
    for metrics_path in scan_root.rglob("final_test_metrics.json"):
        try:
            relative = metrics_path.relative_to(scan_root)
        except ValueError:
            continue
        if not relative.parts or relative.parts[0] not in {f"table{i}" for i in range(5, 11)}:
            continue
        hp_path = metrics_path.with_name("best_hparams.json")
        if not hp_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        hp = json.loads(hp_path.read_text(encoding="utf-8"))
        table_dir = relative.parts[0]
        rows.append(
            {
                "table": int(table_dir.removeprefix("table")),
                "condition": condition_for_result(table_dir, hp, metrics, metrics_path),
                "dataset": metrics.get("dataset", hp.get("dataset")),
                "seed": int(metrics.get("seed", hp.get("seed"))),
                "seq_len": metrics.get("seq_len", hp.get("seq_len")),
                "test_acc": metrics.get("test_acc"),
                "test_macro_f1": metrics.get("test_macro_f1"),
                "test_balanced_acc": metrics.get("test_balanced_acc"),
                "selected_epoch": metrics.get("selected_epoch"),
                "selection_metric": metrics.get("selection_metric"),
                "selection_value": metrics.get("selection_value"),
                "result_dir": str(metrics_path.parent),
            }
        )

    per_seed_path = output_root / "tables_5_10_per_seed.csv"
    if not rows:
        print(f"No completed metrics found under {output_root}")
        return per_seed_path, output_root / "tables_5_10_mean_std.csv"
    rows.sort(key=lambda row: (row["table"], row["condition"], row["dataset"], row["seed"]))
    with per_seed_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["table"], row["condition"], row["dataset"]), []).append(row)
    summary_rows = []
    for (table, condition, dataset), group in sorted(grouped.items()):
        seeds = {row["seed"] for row in group}
        summary = {
            "table": table,
            "condition": condition,
            "dataset": dataset,
            "n_seeds": len(seeds),
            "seeds": ",".join(map(str, sorted(seeds))),
            "complete_expected_seeds": seeds == expected_seed_set,
        }
        for metric in ("test_acc", "test_macro_f1", "test_balanced_acc"):
            values = [float(row[metric]) for row in group if row.get(metric) is not None]
            summary[f"{metric}_mean"] = statistics.fmean(values) if values else None
            summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary_rows.append(summary)
    summary_path = output_root / "tables_5_10_mean_std.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {per_seed_path}")
    print(f"Wrote {summary_path}")
    return per_seed_path, summary_path


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", nargs="+", type=int, choices=range(5, 11), default=[5, 6, 7, 8, 9, 10])
    parser.add_argument("--model-file", type=Path, default=here / "AIM_TSformer.py")
    parser.add_argument("--generator-file", type=Path, default=here / "generate_aim_variants.py")
    parser.add_argument("--patch-runner", type=Path, default=here / "patch_ablations.py")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ.get("AIM_DATASET_ROOT", here / "dataset")),
    )
    parser.add_argument("--ts-type", default="Multivariate_ts")
    parser.add_argument(
        "--preprocess-mode",
        default="original",
        help="Public original means native-valued time series.",
    )
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--early-patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--output-root", type=Path, default=here / "results_tables_5_10")
    parser.add_argument("--prepare-images", action="store_true")
    parser.add_argument("--overwrite-images", action="store_true")
    parser.add_argument("--generation-engine", choices=("torch", "numpy"), default="torch")
    parser.add_argument("--generation-device", default="cuda")
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    cli.tables = sorted(set(cli.tables))
    cli.seeds = list(dict.fromkeys(cli.seeds))
    if not cli.seeds:
        raise ValueError("At least one seed is required")
    if cli.aggregate_only:
        aggregate_results(cli.output_root, cli.seeds)
        return

    dataset_dirs = selected_dataset_dirs(cli)
    if cli.prepare_images:
        prepare_images(cli, dataset_dirs)
    validate_inputs(cli, dataset_dirs)

    public = load_module(cli.model_file, "aim_tsformer_public_tables_5_10")
    public.USE_AMP = not cli.no_amp

    if 5 in cli.tables:
        root = cli.output_root / "table5"
        public.main_grid(
            make_core_args(
                cli,
                root,
                [image_type(cli.preprocess_mode, variant) for variant in TABLE5_VARIANTS],
                sequence_lengths=["native"],
            ),
            base_sweep(),
        )

    if 6 in cli.tables:
        run_patch_modes(cli, 6, TABLE6_MODES)

    if 7 in cli.tables:
        run_patch_modes(cli, 7, TABLE7_MODES)

    if 8 in cli.tables:
        run_table8(public, cli)

    if 9 in cli.tables:
        run_patch_modes(cli, 9, TABLE9_MODES)

    if 10 in cli.tables:
        root = cli.output_root / "table10"
        public.main_grid(
            make_core_args(
                cli,
                root,
                [image_type(cli.preprocess_mode, "original")],
                sequence_lengths=list(TABLE10_LENGTHS),
            ),
            base_sweep(),
        )

    aggregate_results(cli.output_root, cli.seeds)


if __name__ == "__main__":
    main()
