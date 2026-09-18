# AIM-TSformer

Code for the 24 fixed-length datasets listed in `configs/datasets.json`.

## Setup

Use Python 3.11 with `environment.yml` or `requirements.txt`. Place the official UEA files under:

```text
dataset/Multivariate_ts/<dataset>/<dataset>_TRAIN.ts
dataset/Multivariate_ts/<dataset>/<dataset>_TEST.ts
```

The source data are available from the [UEA Multivariate Time Series Classification Archive](https://doi.org/10.48550/arXiv.1811.00075).

## Main experiment

```bash
python preprocess_time_series.py --modes original
python generate_aim_variants.py --preprocess-modes original --variants original
python AIM_TSformer.py
```

The main script uses seeds 42–46 and searches MHA and ConvMixer depths in `{1, 2, 4}`. It selects the configuration and epoch by training loss, with early stopping patience 30, then evaluates the selected model on the official TEST split once. The other settings are defined in `AIM_TSformer.py`. Package versions are listed in the environment files, and dataset names and lengths are listed in `configs/datasets.json`.

## Other experiments

```bash
python generate_baseline_images.py --preprocess-mode original
python table2_image_comparison.py
python table4_angular_mapping.py --prepare-inputs
python tables5_10_ablations.py --prepare-images
python table11_input_sensitivity.py --prepare-inputs
python baselines/tables15_17_baselines.py --skip-gpu-wait
```

Image transformation settings are in `generate_baseline_images.py`, and baseline configurations are in the scripts under `baselines/`. The existing per-seed and summary tables are under `results/`. Training results may vary across software and hardware environments.

Datasets vary widely in channel count, and higher channel counts increase GPU memory usage substantially. If a run hits an out of memory error, lower --batch-size for that dataset.
