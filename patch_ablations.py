from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np


MODES = (
    "baseline",
    "grid_size",
    "overlapping",
    "diagonal_band",
    "class_aware",
    "shifted_alignment",
    "random_alignment",
    "no_contrast",
    "global_infonce",
    "batch_all",
)


@dataclass
class AblationConfig:
    mode: str = "baseline"
    overlap_ratio: float = 0.0
    diagonal_band_radius: Optional[float] = None
    contrastive_variant: str = "temporal"
    alignment_variant: str = "diagonal"
    random_alignment_seed: int = 3407


ACTIVE = AblationConfig()


def load_public_module(path: Path):
    specification = importlib.util.spec_from_file_location("aim_tsformer_public_patch", path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def patch_window_geometry(length: int, grid: int, overlap_ratio: float):
    if overlap_ratio <= 0:
        window = max(1, length // grid)
        return [index * window for index in range(grid)], window
    denominator = 1.0 + (grid - 1) * (1.0 - overlap_ratio)
    window = min(length, max(1, int(math.ceil(length / denominator))))
    starts = np.rint(np.linspace(0, max(0, length - window), grid)).astype(int).tolist()
    return starts, window


def install_ablation_model(public):
    torch = public.torch
    F = public.F
    optim = public.optim
    BaseModel = public.AIM_TSformer_Agg
    BaseEncoder = public.LocalPatchEncoder2D

    def encoder_forward(encoder, x_hw_channel):
        batch = x_hw_channel.size(0)
        grid = encoder.S
        x = x_hw_channel.permute(0, 3, 1, 2)
        height, width = x.shape[-2:]
        row_starts, patch_height = patch_window_geometry(height, grid, ACTIVE.overlap_ratio)
        col_starts, patch_width = patch_window_geometry(width, grid, ACTIVE.overlap_ratio)
        patches = []
        geometry = []
        for row_start in row_starts:
            for col_start in col_starts:
                patch = x[
                    :,
                    :,
                    row_start : row_start + patch_height,
                    col_start : col_start + patch_width,
                ]
                patch = F.adaptive_avg_pool2d(patch, (encoder.inner, encoder.inner))
                patches.append(patch)
                geometry.append((row_start, col_start, patch_height, patch_width))
        x = torch.stack(patches, dim=1).reshape(
            batch * encoder.N, x.size(1), encoder.inner, encoder.inner
        )
        x = encoder.stem(x)
        x = x.reshape(batch, encoder.N, -1, encoder.inner, encoder.inner) + encoder.patch_pos
        x = x.reshape(batch * encoder.N, -1, encoder.inner, encoder.inner)
        for block in encoder.blocks:
            x = block(x)
        feature_maps = x.reshape(batch, encoder.N, x.size(1), encoder.inner, encoder.inner)
        full_tokens = feature_maps.mean(dim=(-1, -2))
        contrast_tokens = full_tokens

        radius = ACTIVE.diagonal_band_radius
        if radius is not None:
            contrast_tokens = full_tokens.clone()
            local_axis = torch.arange(
                encoder.inner, device=x.device, dtype=x.dtype
            ) + 0.5
            for patch_index, (row_start, col_start, ph, pw) in enumerate(geometry):
                patch_row, patch_col = divmod(patch_index, grid)
                if patch_row != patch_col:
                    continue
                global_rows = row_start + local_axis * (ph / encoder.inner)
                global_cols = col_start + local_axis * (pw / encoder.inner)
                mask = (global_rows[:, None] - global_cols[None, :]).abs() <= float(radius)
                count = mask.sum()
                if int(count.item()) > 0:
                    weighted = (
                        feature_maps[:, patch_index] * mask[None, None]
                    ).sum(dim=(-1, -2)) / count.to(feature_maps.dtype)
                    contrast_tokens[:, patch_index] = weighted

        
        encoder.last_contrast_tokens = contrast_tokens
        encoder.last_window_geometry = geometry
        return full_tokens, full_tokens.mean(dim=1)

    def pool_ts_intervals(ts_tokens, grid: int, overlap_ratio: float):
        length = ts_tokens.size(1)
        if overlap_ratio <= 0:
            patch = length // grid
            usable = grid * patch
            if usable <= 0:
                return ts_tokens.mean(dim=1, keepdim=True)
            
            aligned = ts_tokens[:, :usable, :]
            return aligned.reshape(ts_tokens.size(0), grid, patch, ts_tokens.size(2)).mean(dim=2)
        starts, window = patch_window_geometry(length, grid, overlap_ratio)
        return torch.stack(
            [ts_tokens[:, start : start + window].mean(dim=1) for start in starts], dim=1
        )

    def contrastive_loss(ts_tokens, img_tokens, temperature: float, labels=None):
        batch, _, _ = ts_tokens.shape
        patch_count = img_tokens.size(1)
        grid = int(round(math.sqrt(patch_count)))
        if grid * grid != patch_count:
            ts_mean = F.normalize(ts_tokens.mean(dim=1), dim=-1)
            image_mean = F.normalize(img_tokens.mean(dim=1), dim=-1)
            return 1.0 - F.cosine_similarity(ts_mean, image_mean).mean()
        ts_vectors = F.normalize(
            pool_ts_intervals(ts_tokens, grid, ACTIVE.overlap_ratio), dim=-1, eps=1e-6
        )
        image_vectors = F.normalize(img_tokens, dim=-1, eps=1e-6)
        diagonal = torch.arange(grid, device=ts_tokens.device) * (grid + 1)

        if ACTIVE.contrastive_variant == "global_infonce":
            ts_global = F.normalize(ts_tokens.mean(dim=1), dim=-1, eps=1e-6)
            image_global = F.normalize(img_tokens.mean(dim=1), dim=-1, eps=1e-6)
            logits = torch.matmul(ts_global, image_global.transpose(0, 1)) / temperature
            targets = torch.arange(batch, device=ts_tokens.device)
            return F.cross_entropy(logits, targets)

        if ACTIVE.contrastive_variant == "batch_all":
            
            logits = torch.einsum("ngd,mid->ngmi", ts_vectors, image_vectors) / temperature
            logits = logits.reshape(batch * grid, batch * patch_count)
            sample_index = torch.arange(batch, device=ts_tokens.device)[:, None]
            targets = sample_index * patch_count + diagonal[None, :]
            return F.cross_entropy(logits, targets.reshape(-1))

        if ACTIVE.contrastive_variant == "temporal":
            logits = torch.bmm(ts_vectors, image_vectors.transpose(1, 2)) / temperature
            targets_per_interval = diagonal
            if ACTIVE.alignment_variant == "shifted":
                targets_per_interval = torch.roll(diagonal, shifts=-1)
            elif ACTIVE.alignment_variant == "random_fixed":
                candidates = [
                    index for index in range(patch_count)
                    if index not in set(diagonal.detach().cpu().tolist())
                ]
                if len(candidates) < grid:
                    candidates = list(range(patch_count))
                generator = torch.Generator(device="cpu")
                generator.manual_seed(ACTIVE.random_alignment_seed + grid * 1009)
                order = torch.randperm(len(candidates), generator=generator)[:grid].tolist()
                targets_per_interval = torch.tensor(
                    [candidates[index] for index in order],
                    device=ts_tokens.device,
                    dtype=torch.long,
                )
            elif ACTIVE.alignment_variant != "diagonal":
                raise ValueError(f"Unknown alignment variant: {ACTIVE.alignment_variant}")
            targets = targets_per_interval.unsqueeze(0).expand(batch, -1).reshape(-1)
            return F.cross_entropy(logits.reshape(batch * grid, patch_count), targets)

        if ACTIVE.contrastive_variant != "class_aware":
            raise ValueError(f"Unknown contrastive variant: {ACTIVE.contrastive_variant}")
        if labels is None:
            raise ValueError("class_aware contrast requires labels")

        
        logits = torch.einsum("ngd,mid->ngmi", ts_vectors, image_vectors) / temperature
        candidate_patch = torch.arange(patch_count, device=labels.device)
        same_class = labels[:, None] == labels[None, :]
        same_interval = candidate_patch[None, :] == diagonal[:, None]
        positives = same_class[:, None, :, None] & same_interval[None, :, None, :]
        negatives = (~same_class)[:, None, :, None].expand_as(logits)
        valid = positives | negatives
        negative_infinity = torch.finfo(logits.dtype).min
        log_denominator = torch.logsumexp(logits.masked_fill(~valid, negative_infinity), dim=(-1, -2))
        log_numerator = torch.logsumexp(logits.masked_fill(~positives, negative_infinity), dim=(-1, -2))
        return (log_denominator - log_numerator).mean()

    class AblationModel(BaseModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            for stage in self.stages:
                if isinstance(stage.emb2d, BaseEncoder):
                    stage.emb2d.forward = types.MethodType(encoder_forward, stage.emb2d)

        def forward(self, x1d, x2d, labels=None):
            contrastive_losses: List[torch.Tensor] = []
            current_x1d = x1d
            ts_global_pure = None
            image_global_last = None
            last_ts_tokens = None
            last_image_tokens = None

            for stage_index in range(self.num_stages):
                output = self.stages[stage_index](current_x1d, x2d)
                ts_features = output["ts_tokens"]
                image_features = output["img_tokens"]
                contrast_image_features = getattr(
                    self.stages[stage_index].emb2d,
                    "last_contrast_tokens",
                    image_features,
                )
                if stage_index == 0:
                    ts_global_pure = output["ts_global"]
                last_ts_tokens = ts_features
                last_image_tokens = image_features

                if ts_features.shape[1] != image_features.shape[1]:
                    resized_image = F.interpolate(
                        image_features.transpose(1, 2),
                        size=ts_features.shape[1],
                        mode="linear",
                        align_corners=False,
                    ).transpose(1, 2)
                else:
                    resized_image = image_features
                if self.use_gated_fusion:
                    image_normalized = self.img_norms[stage_index](resized_image)
                    ts_for_fusion = (
                        self.ts_norms[stage_index](ts_features)
                        if self.symmetric_fusion_norm
                        else ts_features
                    )
                    fused = ts_for_fusion + self.fusion_gates[stage_index] * image_normalized
                    current_x1d = self.fusion_projs[stage_index](fused)
                else:
                    current_x1d = ts_features

                if self.use_contrast_projection:
                    ts_projection = self.ts_projs[stage_index](ts_features)
                    image_projection = self.img_projs[stage_index](contrast_image_features)
                else:
                    ts_projection, image_projection = ts_features, contrast_image_features
                contrastive_losses.append(
                    contrastive_loss(
                        ts_projection,
                        image_projection,
                        self.temperature,
                        labels=labels,
                    )
                )
                if stage_index == self.num_stages - 1:
                    image_global_last = output["img_global"]

            ts_fused_global = current_x1d.mean(dim=1)
            ts_global = ts_global_pure if ts_global_pure is not None else ts_fused_global
            image_global = image_global_last
            if self.training and self.modality_dropout > 0.0:
                probability = float(self.modality_dropout)
                random_values = torch.rand(ts_fused_global.size(0), device=ts_fused_global.device)
                keep_ts = (random_values >= probability).float().unsqueeze(1)
                keep_image = (
                    (random_values < probability) | (random_values >= 2.0 * probability)
                ).float().unsqueeze(1)
                ts_part = ts_fused_global * keep_ts
                image_part = image_global * keep_image
            else:
                ts_part, image_part = ts_fused_global, image_global
            return {
                "logits": self.head(torch.cat([ts_part, image_part], dim=1)),
                "ts_logits": self.ts_head(ts_global),
                "img_logits": self.img_head(image_global),
                "contrast_losses": torch.stack(contrastive_losses),
                "ts_tokens": last_ts_tokens,
                "img_tokens": last_image_tokens,
                "ts_global": ts_global,
                "img_global": image_global,
            }

    def train_one_trainloss(
        model,
        train_loader,
        epochs,
        lr,
        optimizer_name,
        weight_decay,
        early_patience=30,
        min_delta=1e-6,
        eval_source="fusion",
    ):
        optimizer_class = {"adam": optim.Adam, "adamw": optim.AdamW}.get(
            optimizer_name.lower()
        )
        if optimizer_class is None:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)
        try:
            scaler = torch.amp.GradScaler(
                "cuda", enabled=(public.USE_AMP and public.DEVICE == "cuda")
            )
        except TypeError:
            scaler = torch.cuda.amp.GradScaler(
                enabled=(public.USE_AMP and public.DEVICE == "cuda")
            )
        best = {"loss": float("inf"), "acc": -1.0, "state": None, "epoch": -1}
        curves = {"train_loss": [], "train_acc": []}
        epochs_without_improvement = 0
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss, predictions, targets = 0.0, [], []
            part_totals = {"ce": 0.0, "ts_ce": 0.0, "img_ce": 0.0, "contrast": 0.0}
            for x1d, x2d, labels in train_loader:
                x1d = x1d.to(public.DEVICE)
                x2d = x2d.to(public.DEVICE)
                labels = labels.to(public.DEVICE)
                optimizer.zero_grad()
                with torch.amp.autocast(
                    "cuda", enabled=(public.USE_AMP and public.DEVICE == "cuda")
                ):
                    output = model(x1d, x2d, labels=labels)
                    loss, parts = model.compute_total_loss(output, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), public.GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item() * x1d.size(0)
                for name in part_totals:
                    part_totals[name] += parts[name] * x1d.size(0)
                logits = public.select_eval_logits(output, eval_source)
                predictions.extend(logits.argmax(1).detach().cpu().numpy())
                targets.extend(labels.detach().cpu().numpy())
            train_loss = total_loss / len(train_loader.dataset)
            train_acc = public.accuracy_score(targets, predictions)
            curves["train_loss"].append(train_loss)
            curves["train_acc"].append(train_acc)
            print(
                f"Epoch {epoch}/{epochs} | train_loss={train_loss:.6f} "
                f"train_acc={train_acc:.4f} | monitor=train_loss"
            )
            if train_loss < best["loss"] - min_delta:
                best.update(
                    loss=train_loss,
                    acc=train_acc,
                    state={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                    epoch=epoch,
                )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= early_patience:
                print(f"Early stopping at epoch {epoch}; selected epoch {best['epoch']}.")
                break
        return {
            "state": best["state"],
            "curves": curves,
            "best_epoch": best["epoch"],
            "best_train_loss": best["loss"],
            "best_train_acc": best["acc"],
            "selection_metric": "train_loss",
            "selection_value": best["loss"],
            "selection_score": -best["loss"],
            "eval_source": eval_source,
        }

    def evaluate_test_once(model, test_loader, eval_source="fusion"):
        model.eval()
        total_loss, predictions, targets = 0.0, [], []
        with torch.no_grad():
            for x1d, x2d, labels in test_loader:
                x1d = x1d.to(public.DEVICE)
                x2d = x2d.to(public.DEVICE)
                labels = labels.to(public.DEVICE)
                with torch.amp.autocast(
                    "cuda", enabled=(public.USE_AMP and public.DEVICE == "cuda")
                ):
                    output = model(x1d, x2d, labels=labels)
                    loss, _ = model.compute_total_loss(output, labels)
                total_loss += loss.item() * x1d.size(0)
                logits = public.select_eval_logits(output, eval_source)
                predictions.extend(logits.argmax(1).cpu().numpy())
                targets.extend(labels.cpu().numpy())
        y_true, y_pred = np.asarray(targets), np.asarray(predictions)
        result = {
            "test_acc": float(public.accuracy_score(y_true, y_pred)),
            "test_loss": total_loss / len(test_loader.dataset),
            "test_macro_f1": float(public.f1_score(y_true, y_pred, average="macro")),
            "test_balanced_acc": float(public.balanced_accuracy_score(y_true, y_pred)),
            "test_parts": {},
            "y_true": y_true,
            "y_pred": y_pred,
            "eval_source": eval_source,
        }
        print(
            f"[FINAL TEST ONCE] loss={result['test_loss']:.6f} "
            f"accuracy={result['test_acc']:.4f}"
        )
        return result

    public.AIM_TSformer_Agg = AblationModel
    public.train_one_trainloss = train_one_trainloss
    public.evaluate_test_once = evaluate_test_once
    return AblationModel


def configuration_for_mode(mode: str, band_radius: float) -> AblationConfig:
    if mode == "overlapping":
        return AblationConfig(mode=mode, overlap_ratio=0.5)
    if mode == "diagonal_band":
        return AblationConfig(mode=mode, diagonal_band_radius=band_radius)
    if mode == "class_aware":
        return AblationConfig(mode=mode, contrastive_variant="class_aware")
    if mode == "shifted_alignment":
        return AblationConfig(mode=mode, alignment_variant="shifted")
    if mode == "random_alignment":
        return AblationConfig(mode=mode, alignment_variant="random_fixed")
    if mode == "global_infonce":
        return AblationConfig(mode=mode, contrastive_variant="global_infonce")
    if mode == "batch_all":
        return AblationConfig(mode=mode, contrastive_variant="batch_all")
    return AblationConfig(mode=mode)


def make_sweep(mode: str, patch_grid: Optional[int] = None) -> Dict[str, list]:
    return {
        "stages": [1],
        "patch_grid": [int(patch_grid)] if patch_grid is not None else [4],
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
        "lambda_contrast": [0.0 if mode == "no_contrast" else 1.0],
    }


def make_args(cli, mode: str, output_root: Optional[Path] = None) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_root=str(cli.dataset_root),
        ts_type=cli.ts_type,
        output_root=str(output_root or (cli.output_root / mode)),
        datasets_to_run=list(cli.datasets) if cli.datasets else None,
        datasets_to_skip=[],
        backbone="convmixer",
        conv_dw_kernel=9,
        img_types=[cli.image_type],
        ts_preprocess_mode=cli.preprocess_mode,
        sequence_lengths=None,
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


def annotate_hparams(
    output_root: Path,
    config: AblationConfig,
    additions: Optional[Dict[str, object]] = None,
) -> None:
    additions = {
        "ablation_mode": config.mode,
        "patch_overlap_ratio": config.overlap_ratio,
        "diagonal_band_radius_pixels": config.diagonal_band_radius,
        "contrastive_variant": config.contrastive_variant,
        "alignment_variant": config.alignment_variant,
        **(additions or {}),
    }
    scan_root = Path("\\\\?\\" + str(output_root.resolve())) if os.name == "nt" else output_root
    for path in scan_root.rglob("best_hparams.json"):
        values = json.loads(path.read_text(encoding="utf-8"))
        values.update(additions)
        path.write_text(json.dumps(values, indent=2), encoding="utf-8")
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
    parser.add_argument("--output-root", type=Path, default=here / "results_patch_ablation")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=["grid_size", "overlapping", "diagonal_band", "class_aware"],
        help="Patch-grid, overlap, diagonal-band, and class-aware modes are run by default.",
    )
    parser.add_argument(
        "--image-type",
        default="aim_original_original",
        help="Public original AIM generated from native-valued time series.",
    )
    parser.add_argument(
        "--preprocess-mode",
        default="original",
        help="Public original is the native-valued preprocessed split.",
    )
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--early-patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--band-radius", type=float, default=1.0)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    global ACTIVE
    cli = parse_args()
    public = load_public_module(cli.model_file)
    install_ablation_model(public)
    public.USE_AMP = not cli.no_amp

    for mode in cli.modes:
        ACTIVE = configuration_for_mode(mode, cli.band_radius)
        grid_values = [2, 4, 8] if mode == "grid_size" else [4]
        for patch_grid in grid_values:
            mode_root = (
                cli.output_root / mode / f"S{patch_grid}"
                if mode == "grid_size"
                else cli.output_root / mode
            )
            mode_root.mkdir(parents=True, exist_ok=True)
            public.main_grid(
                make_args(cli, mode, mode_root),
                make_sweep(mode, patch_grid=patch_grid),
            )
            annotate_hparams(
                mode_root,
                ACTIVE,
                {"patch_grid_ablation": patch_grid},
            )


if __name__ == "__main__":
    main()
