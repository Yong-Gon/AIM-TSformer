from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

from public_runner_utils import (
    dataset_directories,
    hpo_sweep,
    load_module,
    model_args,
)


VARIANTS = ("original", "no_exp", "no_abs", "no_exp_no_abs")


def parse_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-file", type=Path, default=here / "AIM_TSformer.py")
    parser.add_argument("--preprocessor-file", type=Path, default=here / "preprocess_time_series.py")
    parser.add_argument("--generator-file", type=Path, default=here / "generate_aim_variants.py")
    parser.add_argument("--dataset-root", type=Path, default=Path(os.environ.get("AIM_DATASET_ROOT", here / "dataset")))
    parser.add_argument("--ts-type", default="Multivariate_ts")
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--prepare-inputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--generation-engine", choices=("torch", "numpy"), default="torch")
    parser.add_argument("--generation-device", default="cuda")
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=here / "results_mapping_ablation")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--early-patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main():
    cli = parse_args()
    public = load_module(cli.model_file, "aim_tsformer_public_mapping")
    preprocessor = load_module(cli.preprocessor_file, "aim_preprocessor_mapping")
    generator = load_module(cli.generator_file, "aim_generator_mapping")
    dataset_dirs = dataset_directories(cli.dataset_root, cli.ts_type, cli.datasets)

    if cli.prepare_inputs:
        generation_args = SimpleNamespace(
            engine=cli.generation_engine,
            device=cli.generation_device,
            batch_size=cli.generation_batch_size,
            overwrite=cli.overwrite,
        )
        for dataset_dir in dataset_dirs:
            preprocessor.process_dataset(dataset_dir, ["original"], cli.overwrite)
            generator.process_dataset(dataset_dir, ["original"], cli.variants, None, generation_args)

    for variant in cli.variants:
        result_root = cli.output_root / variant
        public.main_grid(
            model_args(
                cli,
                output_root=result_root,
                datasets=[path.name for path in dataset_dirs],
                preprocess_mode="original",
                image_type=f"aim_original_{variant}",
            ),
            hpo_sweep(),
        )


if __name__ == "__main__":
    main()
