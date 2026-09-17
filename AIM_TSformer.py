import os
import re
import json
import pickle
from typing import Tuple, Dict, Optional, List
from types import SimpleNamespace
from itertools import product

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sktime.datasets import load_from_tsfile_to_dataframe

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = True
GRAD_CLIP_NORM = 1.0

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pjoin(*parts): return os.path.join(*parts)

from scipy.interpolate import interp1d

def safe_paa_transform(X: np.ndarray, n_samples: int, desired_length: int) -> np.ndarray:
    current_length = X.shape[1]
    window_size = current_length / desired_length
    out = np.zeros((n_samples, desired_length), dtype=X.dtype)
    for i in range(desired_length):
        start = int(round(i * window_size))
        end = int(round((i + 1) * window_size))
        end = min(end, current_length)
        if end > start: out[:, i] = np.mean(X[:, start:end], axis=1)
        else: out[:, i] = X[:, start]
    return out

def adjust_sequence_length(data: np.ndarray, desired_length: int, upsample_method: str = 'linear') -> np.ndarray:
    n_samples, current_length, n_vars = data.shape
    if current_length == desired_length: return data.copy()
    adjusted = np.zeros((n_samples, desired_length, n_vars), dtype=data.dtype)
    x_old = np.arange(current_length)
    x_new = np.linspace(0, current_length - 1, desired_length)
    for v in range(n_vars):
        X = data[:, :, v]
        if current_length > desired_length:
            adjusted[:, :, v] = safe_paa_transform(X, n_samples, desired_length)
        else:
            for i in range(n_samples):
                f = interp1d(x_old, X[i], kind='linear' if upsample_method == 'linear' else 'nearest')
                adjusted[i, :, v] = f(x_new)
    return adjusted

def fit_train_labels(train_ts_file: str):
    _, y_tr = load_from_tsfile_to_dataframe(train_ts_file)
    classes = list(pd.unique(y_tr))
    mapping = {cls: i for i, cls in enumerate(classes)}
    return np.array([mapping[v] for v in y_tr]), mapping


def transform_labels_with_mapping(ts_file: str, mapping: Dict):
    _, y = load_from_tsfile_to_dataframe(ts_file)
    unknown = [v for v in pd.unique(y) if v not in mapping]
    if unknown:
        raise ValueError(f"Labels not present in TRAIN mapping: {unknown}")
    return np.array([mapping[v] for v in y])

def instance_z_score(x: np.ndarray, axis: Tuple[int, ...]):
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True) + 1e-8
    return (x - mean) / std


class AIMDataset(Dataset):
    def __init__(self, x_1d, x_2d, y): self.x_1d, self.x_2d, self.y = x_1d, x_2d, y
    def __len__(self): return len(self.x_1d)
    def __getitem__(self, idx):
        return torch.from_numpy(self.x_1d[idx]).float(), torch.from_numpy(self.x_2d[idx]).float(), torch.tensor(self.y[idx]).long()

def gelu(): return nn.GELU()
def init_trunc_normal_(tensor, std=0.02): nn.init.trunc_normal_(tensor, std=std)

class MHABlock1D(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        hidden_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        h = self.norm1(x)
        out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dropout1(out)
        
        h = self.norm2(x)
        x = x + self.mlp(h)
        return x

class OneDEmbedding_MHAStack(nn.Module):
    def __init__(self, seq_len, in_dim, d_model=128, n_heads=8, n_modules=8, dropout=0.1,
                 use_positional_emb=True, use_segment_embedding=False, num_segments=4):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, d_model) if in_dim != d_model else nn.Identity()
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model)) if use_positional_emb else None

        self.use_segment_embedding = bool(use_segment_embedding)
        self.num_segments = int(num_segments)
        self.segment_emb = (
            nn.Parameter(torch.zeros(1, self.num_segments, d_model))
            if self.use_segment_embedding else None
        )

        self.blocks = nn.ModuleList([MHABlock1D(d_model, n_heads, dropout) for _ in range(n_modules)])
        self.norm_out = nn.LayerNorm(d_model)
        if self.pos_emb is not None: init_trunc_normal_(self.pos_emb)
        if self.segment_emb is not None: init_trunc_normal_(self.segment_emb)
        
    def forward(self, x):
        h = self.proj_in(x)
        L = h.size(1)

        if self.pos_emb is not None:
            if L <= self.pos_emb.size(1):
                h = h + self.pos_emb[:, :L, :]
            else:
                pos_t = self.pos_emb.transpose(1, 2)
                pos_resized = F.interpolate(pos_t, size=L, mode='linear', align_corners=False)
                h = h + pos_resized.transpose(1, 2)

        if self.segment_emb is not None:
            seg_idx = torch.div(
                torch.arange(L, device=h.device) * self.num_segments,
                L,
                rounding_mode='floor'
            ).clamp(max=self.num_segments - 1)
            h = h + self.segment_emb[:, seg_idx, :]

        for blk in self.blocks:
            h = blk(h)
        return self.norm_out(h), h.mean(dim=1)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x

class LocalPatchEncoder2D(nn.Module):
    def __init__(self, img_size, patch_grid, in_chans, dim=128, depth=4,
                 dw_kernel_size=9, max_inner=None):
        super().__init__()
        self.S = patch_grid
        self.N = patch_grid * patch_grid
        self.patch_size = max(1, img_size // patch_grid)

        self.max_inner = max_inner
        self.inner = self.patch_size if max_inner is None else min(self.patch_size, int(max_inner))
        self.resize = (nn.AdaptiveAvgPool2d((self.inner, self.inner))
                       if self.patch_size != self.inner else nn.Identity())

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )
        self.patch_pos = nn.Parameter(torch.zeros(1, self.N, dim, 1, 1))
        init_trunc_normal_(self.patch_pos, 0.02)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=dw_kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim),
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim),
            ) for _ in range(depth)
        ])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x_hwC):
        B = x_hwC.size(0); S = self.S
        x = x_hwC.permute(0, 3, 1, 2)
        H, W = x.shape[-2:]
        ph, pw = H // S, W // S
        if ph <= 0 or pw <= 0:
            raise ValueError(
                f"AIM input {(H, W)} is smaller than the {S}x{S} patch grid."
            )
        x = x[:, :, :S * ph, :S * pw]
        M = x.size(1)
        x = x.reshape(B, M, S, ph, S, pw).permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(B * self.N, M, ph, pw)
        if (ph, pw) != (self.inner, self.inner):
            x = F.adaptive_avg_pool2d(x, (self.inner, self.inner))
        x = self.stem(x)
        x = x.reshape(B, self.N, -1, self.inner, self.inner) + self.patch_pos
        x = x.reshape(B * self.N, -1, self.inner, self.inner)
        for blk in self.blocks:
            x = blk(x)
        tok = self.gap(x).flatten(1) 
        tok = tok.reshape(B, self.N, -1) 
        g = tok.mean(dim=1) 
        return tok, g

def compute_patch_level_loss(ts_tokens: torch.Tensor,
                             img_tokens: torch.Tensor,
                             temperature: float = 0.2) -> torch.Tensor:
    B, L_ts, D = ts_tokens.shape
    B2, N_patches, D2 = img_tokens.shape
    
    G = int(round(N_patches ** 0.5))
    if G * G != N_patches:
        z1 = F.normalize(ts_tokens.mean(dim=1), dim=-1)
        z2 = F.normalize(img_tokens.mean(dim=1), dim=-1)
        return 1.0 - F.cosine_similarity(z1, z2).mean()

    p = L_ts // G
    usable_len = G * p
    if usable_len <= 0:
        z1 = F.normalize(ts_tokens.mean(dim=1), dim=-1)
        z2 = F.normalize(img_tokens.mean(dim=1), dim=-1)
        return 1.0 - F.cosine_similarity(z1, z2).mean()
    ts_tokens_aligned = ts_tokens[:, :usable_len, :]

    ts_pooled = F.adaptive_avg_pool1d(ts_tokens_aligned.transpose(1, 2), G).transpose(1, 2)

    ts_vecs = F.normalize(ts_pooled, dim=-1, eps=1e-6)
    img_vecs = F.normalize(img_tokens, dim=-1, eps=1e-6)

    logits = torch.bmm(ts_vecs, img_vecs.transpose(1, 2)) / temperature
    logits = torch.clamp(logits, max=100)

    diag_idx = torch.arange(G, device=ts_tokens.device) * (G + 1)
    
    logits_flat = logits.view(B * G, N_patches)
    targets_flat = diag_idx.unsqueeze(0).expand(B, -1).reshape(-1)
    
    loss = F.cross_entropy(logits_flat, targets_flat)
    return loss

class AggregationBlock(nn.Module):
    def __init__(self, seq_len, ts_in_dim, img_size, img_chans,
                 d_model=128, mha_heads=8, mha_modules=8,
                 backbone="convmixer", patch_grid=4, backbone_depth=4,
                 conv_dw_kernel=7, dropout=0.1,
                 use_segment_embedding=False, max_inner=None):
        super().__init__()
        self.emb1d = OneDEmbedding_MHAStack(
            seq_len, ts_in_dim, d_model,
            n_heads=mha_heads, n_modules=mha_modules,
            dropout=dropout, use_positional_emb=True,
            use_segment_embedding=use_segment_embedding,
            num_segments=patch_grid
        )

        if backbone != "convmixer":
            raise ValueError(
                f"The public AIM-TSformer implementation supports only "
                f"backbone='convmixer', got {backbone!r}."
            )
        self.emb2d = LocalPatchEncoder2D(
            img_size, patch_grid, img_chans,
            dim=d_model, depth=backbone_depth,
            dw_kernel_size=conv_dw_kernel,
            max_inner=max_inner
        )

    def forward(self, x1d, x2d):
        ts_tokens, ts_global = self.emb1d(x1d) 
        img_tokens, img_global = self.emb2d(x2d) 
        return {
            "ts_tokens": ts_tokens,
            "ts_global": ts_global,
            "img_tokens": img_tokens,
            "img_global": img_global
        }

class AIM_TSformer_Agg(nn.Module):
    def __init__(self, n_classes, seq_len, ts_in_dim, img_size, img_chans,
                 num_stages=4, 
                 d_model=[128, 128, 256, 256],    
                 mha_heads=[8, 8, 16, 16],        
                 mha_modules=8,
                 backbone="convmixer", patch_grid=4, backbone_depth=4, conv_dw_kernel=7,
                 dense_units=128, dropout=0.1, temperature=1.0, lambda_contrast=1.0,
                 w_main=1.0, w_ts=1.0, w_img=1.0,
                 fusion_gate_init=1.0, symmetric_fusion_norm=True, modality_dropout=0.1, 
                 use_gated_fusion=True, use_contrast_projection=True,
                 use_segment_embedding=False, max_inner=None,
                 label_smoothing=0.0):

        super().__init__()
        self.num_stages = num_stages
        self.temperature = temperature
        self.lambda_contrast = lambda_contrast
        self.w_main = w_main
        self.w_ts = w_ts
        self.w_img = w_img
        self.fusion_gate_init = fusion_gate_init
        self.symmetric_fusion_norm = symmetric_fusion_norm
        self.modality_dropout = modality_dropout
        self.use_gated_fusion = bool(use_gated_fusion)
        self.use_contrast_projection = bool(use_contrast_projection)
        self.use_segment_embedding = bool(use_segment_embedding)
        self.max_inner = max_inner
        self.label_smoothing = float(label_smoothing)

        if isinstance(d_model, int):
            d_models = [d_model] * num_stages
        else:
            d_models = d_model[:num_stages]
        
        if isinstance(mha_heads, int):
            heads_list = [mha_heads] * num_stages
        else:
            heads_list = mha_heads[:num_stages]

        assert len(d_models) == num_stages, f"d_model list length ({len(d_model)}) is shorter than num_stages ({num_stages})"
        assert len(heads_list) == num_stages, f"mha_heads list length ({len(mha_heads)}) is shorter than num_stages ({num_stages})"

        self.stages = nn.ModuleList()
        self.fusion_projs = nn.ModuleList() 
        self.ts_projs = nn.ModuleList() 
        self.img_projs = nn.ModuleList() 

        self.img_norms = nn.ModuleList() 
        self.ts_norms = nn.ModuleList() 
        self.fusion_gates = nn.ParameterList() 

        for i in range(num_stages):
            current_in_dim = ts_in_dim if i == 0 else d_models[i-1]
            current_d_model = d_models[i]
            current_heads = heads_list[i]

            self.stages.append(
                AggregationBlock(
                    seq_len, current_in_dim, img_size, img_chans,
                    d_model=current_d_model, 
                    mha_heads=current_heads, 
                    mha_modules=mha_modules,
                    backbone=backbone, patch_grid=patch_grid,
                    backbone_depth=backbone_depth, conv_dw_kernel=conv_dw_kernel,
                    dropout=dropout,
                    use_segment_embedding=self.use_segment_embedding,
                    max_inner=self.max_inner
                )
            )

            self.fusion_projs.append(nn.Sequential(
                nn.Linear(current_d_model, current_d_model),
                nn.LayerNorm(current_d_model),
                gelu()
            ))

            self.ts_projs.append(nn.Sequential(
                nn.Linear(current_d_model, current_d_model),
                gelu(),
                nn.Linear(current_d_model, current_d_model)
            ))
            self.img_projs.append(nn.Sequential(
                nn.Linear(current_d_model, current_d_model),
                gelu(),
                nn.Linear(current_d_model, current_d_model)
            ))

            self.img_norms.append(nn.LayerNorm(current_d_model))
            self.ts_norms.append(nn.LayerNorm(current_d_model))
            self.fusion_gates.append(nn.Parameter(torch.tensor(float(self.fusion_gate_init))))

        last_dim = d_models[-1]
        
        self.head = nn.Sequential(
            nn.Linear(last_dim * 2, dense_units),
            gelu(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, n_classes)
        )

        stage1_dim = d_models[0]
        self.ts_head = nn.Sequential(
            nn.Linear(stage1_dim, dense_units),
            gelu(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, n_classes)
        )

        self.img_head = nn.Sequential(
            nn.Linear(last_dim, dense_units),
            gelu(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, n_classes)
        )

    def forward(self, x1d: torch.Tensor, x2d: torch.Tensor) -> Dict[str, torch.Tensor]:
        closses: List[torch.Tensor] = []
        current_x1d = x1d

        ts_global_pure: Optional[torch.Tensor] = None
        img_global_last: Optional[torch.Tensor] = None
        last_ts_tokens: Optional[torch.Tensor] = None
        last_img_tokens: Optional[torch.Tensor] = None

        for s in range(self.num_stages):
            out = self.stages[s](current_x1d, x2d)
            ts_feat = out["ts_tokens"]
            img_feat = out["img_tokens"]

            if s == 0:
                ts_global_pure = out["ts_global"]

            last_ts_tokens = ts_feat
            last_img_tokens = img_feat

            if ts_feat.shape[1] != img_feat.shape[1]:
                img_feat_resized = F.interpolate(
                    img_feat.transpose(1, 2),
                    size=ts_feat.shape[1],
                    mode="linear",
                    align_corners=False
                ).transpose(1, 2)
            else:
                img_feat_resized = img_feat
            
            if self.use_gated_fusion:
                img_feat_norm = self.img_norms[s](img_feat_resized)
                ts_feat_for_fusion = self.ts_norms[s](ts_feat) if self.symmetric_fusion_norm else ts_feat
                fused_feat = ts_feat_for_fusion + self.fusion_gates[s] * img_feat_norm
                current_x1d = self.fusion_projs[s](fused_feat)
            else:
                current_x1d = ts_feat

            if self.use_contrast_projection:
                ts_proj = self.ts_projs[s](ts_feat)
                img_proj = self.img_projs[s](img_feat)
            else:
                ts_proj = ts_feat
                img_proj = img_feat
            
            patch_loss = compute_patch_level_loss(
                ts_proj, img_proj, temperature=self.temperature
            )
            closses.append(patch_loss)

            if s == self.num_stages - 1:
                img_global_last = out["img_global"]

        ts_fused_global = current_x1d.mean(dim=1)

        ts_g = ts_global_pure if ts_global_pure is not None else ts_fused_global
        img_g = img_global_last

        if self.training and self.modality_dropout > 0.0:
            p = float(self.modality_dropout)
            B = ts_fused_global.size(0)
            u = torch.rand(B, device=ts_fused_global.device)
            keep_ts  = (u >= p).float().unsqueeze(1)
            keep_img = ((u < p) | (u >= 2.0 * p)).float().unsqueeze(1)
            ts_part  = ts_fused_global * keep_ts
            img_part = img_g * keep_img
        else:
            ts_part  = ts_fused_global
            img_part = img_g

        fused_final = torch.cat([ts_part, img_part], dim=1)

        return {
            "logits": self.head(fused_final),
            "ts_logits": self.ts_head(ts_g),
            "img_logits": self.img_head(img_g),
            "contrast_losses": torch.stack(closses),
            "ts_tokens": last_ts_tokens,
            "img_tokens": last_img_tokens,
            "ts_global": ts_g,
            "img_global": img_g
        }

    def compute_total_loss(self, outputs: Dict[str, torch.Tensor], y: torch.Tensor):






        ls = float(getattr(self, "label_smoothing", 0.0))

        ce = F.cross_entropy(outputs["logits"], y, label_smoothing=ls) 
        ts_ce = F.cross_entropy(outputs["ts_logits"], y, label_smoothing=ls) 
        img_ce = F.cross_entropy(outputs["img_logits"], y, label_smoothing=ls)
        cl = outputs["contrast_losses"].mean()

        total = (
            self.w_main * ce
            + self.w_ts * ts_ce
            + self.w_img * img_ce
            + self.lambda_contrast * cl
        )

        parts = {
            "ce": ce.item(),
            "ts_ce": ts_ce.item(),
            "img_ce": img_ce.item(),
            "contrast": cl.item(),
            "total": total.item()
        }
        return total, parts


def select_eval_logits(outputs: Dict[str, torch.Tensor], eval_source: str = "fusion") -> torch.Tensor:
    src = eval_source.lower()
    if src == "fusion":
        return outputs["logits"]
    if src == "ts":
        return outputs["ts_logits"]
    if src == "img":
        return outputs["img_logits"]
    if src in {"late_sum", "late_avg"}:
        return outputs["ts_logits"] + outputs["img_logits"]
    if src == "late_prob_avg":
        return F.softmax(outputs["ts_logits"], dim=-1) + F.softmax(outputs["img_logits"], dim=-1)
    raise ValueError(f"Unknown eval_source: {eval_source}")


def train_one_trainloss(model, train_loader, epochs, lr, optimizer_name, weight_decay,
                        early_patience=30, min_delta=1e-6,
                        eval_source="fusion"):
    opt_name = optimizer_name.lower()
    if opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.9, patience=10**9
    )

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and DEVICE == 'cuda'))
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE == 'cuda'))

    best = {
        "train_loss": float('inf'),
        "train_acc": -1.0,
        "state": None,
        "ep": -1,
    }
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        preds = []
        labels = []
        tp = {"ce": 0, "ts_ce": 0, "img_ce": 0, "contrast": 0}

        for xb1d, xb2d, yb in train_loader:
            xb1d, xb2d, yb = xb1d.to(DEVICE), xb2d.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(USE_AMP and DEVICE == 'cuda')):
                out = model(xb1d, xb2d)
                loss, parts = model.compute_total_loss(out, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * xb1d.size(0)
            for k in tp:
                tp[k] += parts[k] * xb1d.size(0)
            eval_logits = select_eval_logits(out, eval_source)
            preds.extend(eval_logits.argmax(1).detach().cpu().numpy())
            labels.extend(yb.detach().cpu().numpy())

        tr_loss = total_loss / len(train_loader.dataset)
        tr_acc = accuracy_score(labels, preds)
        tr_parts = {k: v / len(train_loader.dataset) for k, v in tp.items()}
        scheduler.step(tr_loss)
        improved = tr_loss < best["train_loss"] - min_delta

        print(
            f"Ep {epoch}/{epochs} | Eval({eval_source}) | Tr {tr_loss:.4f} Acc {tr_acc:.4f} | Monitor(train_loss) {tr_loss:.4f}",
            f"  [Tr] ce:{tr_parts['ce']:.3f} ts:{tr_parts['ts_ce']:.3f} img:{tr_parts['img_ce']:.3f} cl:{tr_parts['contrast']:.3f}"
        )

        if improved:
            best.update({
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "ep": epoch,
            })
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= early_patience:
            print(
                f"Early stopping at {epoch}. Best train loss={best['train_loss']:.6f} "
                f"(ep{best['ep']})"
            )
            break

    return {
        "state": best["state"],
        "best_epoch": best["ep"],
        "best_train_loss": best["train_loss"],
        "best_train_acc": best["train_acc"],
        "selection_metric": "train_loss",
        "selection_value": best["train_loss"],
        "selection_score": -best["train_loss"],
        "eval_source": eval_source,
    }


def evaluate_test_once(model, test_loader, eval_source="fusion", device=None):
    if device is None:
        try:
            run_device = next(model.parameters()).device
        except StopIteration:
            run_device = torch.device(DEVICE)
    else:
        run_device = torch.device(device)
    use_cuda_amp = USE_AMP and run_device.type == "cuda"
    model.eval()
    total_loss = 0.0
    preds = []
    labels = []
    ep = {"ce": 0, "ts_ce": 0, "img_ce": 0, "contrast": 0}

    with torch.no_grad():
        for xb1d, xb2d, yb in test_loader:
            xb1d, xb2d, yb = xb1d.to(run_device), xb2d.to(run_device), yb.to(run_device)
            with torch.amp.autocast('cuda', enabled=use_cuda_amp):
                out = model(xb1d, xb2d)
                loss, parts = model.compute_total_loss(out, yb)
            total_loss += loss.item() * xb1d.size(0)
            for k in ep:
                ep[k] += parts[k] * xb1d.size(0)
            eval_logits = select_eval_logits(out, eval_source)
            preds.extend(eval_logits.argmax(1).detach().cpu().numpy())
            labels.extend(yb.detach().cpu().numpy())

    test_loss = total_loss / len(test_loader.dataset)
    y_true = np.asarray(labels)
    y_pred = np.asarray(preds)
    test_acc = float(accuracy_score(y_true, y_pred))
    test_parts = {k: v / len(test_loader.dataset) for k, v in ep.items()}
    print(
        f"[FINAL TEST ONCE] Eval({eval_source}) | Loss {test_loss:.4f} Acc {test_acc:.4f}",
        f"  ce:{test_parts['ce']:.3f} ts:{test_parts['ts_ce']:.3f} img:{test_parts['img_ce']:.3f} cl:{test_parts['contrast']:.3f}"
    )
    return {
        "test_acc": test_acc,
        "test_loss": test_loss,
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "test_balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "test_parts": test_parts,
        "y_true": y_true,
        "y_pred": y_pred,
        "eval_source": eval_source,
    }


def evaluate_saved_checkpoint_arrays(
    best_dir,
    x_1d,
    x_2d,
    y,
    batch_size=32,
    device=None,
    eval_source=None,
    normalize=True,
):
    x_1d = np.asarray(x_1d, dtype=np.float32)
    x_2d = np.asarray(x_2d, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if x_1d.ndim != 3:
        raise ValueError(f"x_1d must be [N,T,M], got {x_1d.shape}")
    if x_2d.ndim != 4:
        raise ValueError(f"x_2d must be [N,H,W,M], got {x_2d.shape}")
    if not (len(x_1d) == len(x_2d) == len(y)):
        raise ValueError(
            f"Sample counts differ: TS={len(x_1d)}, image={len(x_2d)}, labels={len(y)}"
        )
    if x_1d.shape[2] != x_2d.shape[3]:
        raise ValueError(
            f"Channel counts differ: TS={x_1d.shape[2]}, image={x_2d.shape[3]}"
        )
    if x_2d.shape[1] != x_2d.shape[2]:
        raise ValueError(f"AIM input must be square, got {x_2d.shape[1:3]}")
    if normalize:
        x_1d = instance_z_score(x_1d, axis=(1,)).astype(np.float32, copy=False)
        x_2d = instance_z_score(x_2d, axis=(1, 2)).astype(np.float32, copy=False)

    run_device = torch.device(device or DEVICE)
    model, hp = load_checkpoint_model(
        best_dir,
        map_location=run_device,
        seq_len=None,
        in_dim=x_1d.shape[2],
        img_size=None,
        in_chans=x_2d.shape[3],
    )
    source = eval_source or hp.get("eval_source", "fusion")
    loader = DataLoader(
        AIMDataset(x_1d, x_2d, y),
        batch_size=int(batch_size),
        shuffle=False,
    )
    result = evaluate_test_once(
        model,
        loader,
        eval_source=source,
        device=run_device,
    )
    result.update(
        checkpoint_dir=os.path.abspath(best_dir),
        checkpoint_seed=hp.get("seed"),
        checkpoint_dataset=hp.get("dataset"),
        input_sequence_length=int(x_1d.shape[1]),
        input_image_size=int(x_2d.shape[1]),
        strict_checkpoint_load=True,
    )
    return result, hp


def should_run_dataset(args, dataset_name: str) -> bool:
    include = getattr(args, "datasets_to_run", None)
    exclude_raw = getattr(args, "datasets_to_skip", [])

    if isinstance(exclude_raw, str):
        exclude = {exclude_raw}
    else:
        exclude = set(exclude_raw)

    if include is not None:
        if isinstance(include, str):
            include_set = {include}
        else:
            include_set = set(include)

        if dataset_name not in include_set:
            return False

    if dataset_name in exclude:
        return False
    return True


def count_combos(sweep: Dict) -> int:
    n = 1
    for v in sweep.values():
        n *= len(v)
    return n


def _windows_long_path(path):
    path = os.path.abspath(os.fspath(path))
    if os.name != "nt" or path.startswith("\\\\?\\"):
        return path
    if path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + path[2:]
    return "\\\\?\\" + path if len(path) >= 248 else path


def _torch_load_state_dict_compat(path, map_location="cpu"):
    path = _windows_long_path(path)
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(checkpoint).__name__}.")
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def build_model_from_hparams(hp, n_classes, seq_len, in_dim, img_size, in_chans):
    required = [
        "stages", "d_model", "mha_heads", "mha_modules", "backbone",
        "patch_grid", "backbone_depth", "conv_dw_kernel", "dense_units",
        "dropout", "temperature", "lambda_contrast", "label_smoothing",
        "w_main", "w_ts", "w_img", "fusion_gate_init",
        "symmetric_fusion_norm", "modality_dropout", "use_gated_fusion",
        "use_contrast_projection", "use_segment_embedding", "max_inner",
    ]
    missing = [key for key in required if key not in hp]
    if missing:
        raise KeyError(f"best_hparams.json is missing required keys: {missing}")

    return AIM_TSformer_Agg(
        n_classes, seq_len, in_dim, img_size, in_chans,
        hp["stages"], hp["d_model"], hp["mha_heads"], hp["mha_modules"],
        hp["backbone"], hp["patch_grid"],
        hp["backbone_depth"], hp["conv_dw_kernel"],
        hp["dense_units"], hp["dropout"],
        hp["temperature"], hp["lambda_contrast"],
        hp["w_main"], hp["w_ts"], hp["w_img"],
        fusion_gate_init=hp["fusion_gate_init"],
        symmetric_fusion_norm=hp["symmetric_fusion_norm"],
        modality_dropout=hp["modality_dropout"],
        use_gated_fusion=hp["use_gated_fusion"],
        use_contrast_projection=hp["use_contrast_projection"],
        use_segment_embedding=hp["use_segment_embedding"],
        max_inner=hp["max_inner"],
        label_smoothing=hp["label_smoothing"],
    )


def _infer_checkpoint_dimensions(hp, state, n_classes=None, seq_len=None,
                                 in_dim=None, img_size=None, in_chans=None):
    n_classes = n_classes or hp.get("n_classes")
    if n_classes is None:
        n_classes = int(state["head.3.bias"].numel())

    seq_len = seq_len or hp.get("seq_len")
    if seq_len is None:
        seq_len = int(state["stages.0.emb1d.pos_emb"].shape[1])

    in_dim = in_dim or hp.get("in_dim")
    if in_dim is None:
        in_dim = int(state["stages.0.emb1d.proj_in.weight"].shape[1])

    in_chans = in_chans or hp.get("in_chans")
    if in_chans is None:
        key = "stages.0.emb2d.stem.0.weight"
        if key in state:
            in_chans = int(state[key].shape[1])
    if in_chans is None:
        raise KeyError("Could not infer in_chans from JSON or checkpoint tensors.")

    img_size = img_size or hp.get("img_size") or seq_len
    return tuple(map(int, (n_classes, seq_len, in_dim, img_size, in_chans)))


def load_checkpoint_model(best_dir, map_location=None, ckpt_name="selected_checkpoint.pth",
                          hparams_name="best_hparams.json", n_classes=None,
                          seq_len=None, in_dim=None, img_size=None, in_chans=None):
    hparams_path = _windows_long_path(pjoin(best_dir, hparams_name))
    checkpoint_path = _windows_long_path(pjoin(best_dir, ckpt_name))
    with open(hparams_path, "r", encoding="utf-8") as f:
        hp = json.load(f)

    state = _torch_load_state_dict_compat(checkpoint_path, map_location="cpu")
    dims = _infer_checkpoint_dimensions(
        hp, state, n_classes=n_classes, seq_len=seq_len, in_dim=in_dim,
        img_size=img_size, in_chans=in_chans,
    )
    model = build_model_from_hparams(hp, *dims)
    model.load_state_dict(state, strict=True)
    model.to(map_location or DEVICE)
    model.eval()
    return model, hp


def resolve_preprocessed_split_paths(dataset_dir, dataset_name, preprocess_mode=None):
    if preprocess_mode:
        root = pjoin(dataset_dir, "preprocessed", str(preprocess_mode))
    else:
        root = dataset_dir
    return (
        pjoin(root, f"{dataset_name}_train_df.pkl"),
        pjoin(root, f"{dataset_name}_test_df.pkl"),
    )


def filter_requested_sequence_lengths(seq_lens, native_length, requested=None):
    if not requested:
        return list(seq_lens)
    wanted = set()
    for value in requested:
        if isinstance(value, str) and value.strip().lower() == "native":
            wanted.add(int(native_length))
        else:
            wanted.add(int(value))
    return [length for length in seq_lens if int(length) in wanted]


def main_grid(args, sweep):
    if getattr(args, "early_stop_metric", "train_loss") != "train_loss":
        raise ValueError(
            "The public protocol fixes checkpoint selection and early stopping "
            "to train_loss. Set early_stop_metric='train_loss'."
        )

    base_dir = pjoin(args.dataset_root, args.ts_type)
    dataset_list = [name for name in os.listdir(base_dir) if os.path.isdir(pjoin(base_dir, name))]
    os.makedirs(args.output_root, exist_ok=True)

    for img_type in args.img_types:
        for dataset_name in dataset_list[:]:
            if not should_run_dataset(args, dataset_name):
                continue

            print(f"\n[REP] {img_type.upper()} | [DATASET] {dataset_name}")
            dpath = pjoin(base_dir, dataset_name);

            npy_files = os.listdir(dpath)
            pat = re.compile(rf"{re.escape(dataset_name)}_{img_type}_train_(\d+)\.npy")
            seq_lens = sorted(list(set(int(m.group(1)) for f in npy_files if (m := pat.match(f)))))
            if not seq_lens:
                print(f"  No files for {img_type}: {dataset_name}")
                continue

            train_pkl, test_pkl = resolve_preprocessed_split_paths(
                dpath,
                dataset_name,
                getattr(args, "ts_preprocess_mode", None),
            )
            if not (os.path.exists(train_pkl) and os.path.exists(test_pkl)):
                print(
                    "  Missing preprocessed TS pair: "
                    f"{train_pkl} | {test_pkl}"
                )
                continue

            train_y_all, label_map = fit_train_labels(
                pjoin(dpath, f"{dataset_name}_TRAIN.ts")
            )
            n_classes = len(np.unique(train_y_all))

            tr_df = pickle.load(open(train_pkl, "rb"))
            te_df = pickle.load(open(test_pkl, "rb"))

            seq_lens = filter_requested_sequence_lengths(
                seq_lens,
                native_length=np.asarray(tr_df).shape[1],
                requested=getattr(args, "sequence_lengths", None),
            )
            if not seq_lens:
                print(
                    f"  No requested lengths for {dataset_name}; "
                    f"requested={getattr(args, 'sequence_lengths', None)}"
                )
                continue

            for seq_len in seq_lens:
                print(f"  SeqLen: {seq_len}")
                train_x_1d = adjust_sequence_length(tr_df, seq_len)
                test_x_1d = adjust_sequence_length(te_df, seq_len)
    
                train_x_1d = instance_z_score(train_x_1d, axis=(1,))
                test_x_1d = instance_z_score(test_x_1d, axis=(1,))
    
                train_x_2d = np.load(pjoin(dpath, f"{dataset_name}_{img_type}_train_{seq_len}.npy"))
                test_x_2d  = np.load(pjoin(dpath, f"{dataset_name}_{img_type}_test_{seq_len}.npy"))
                
                train_x_2d = train_x_2d.astype(np.float32)
                test_x_2d = test_x_2d.astype(np.float32)
                
                train_x_2d = instance_z_score(train_x_2d, axis=(1, 2))
                test_x_2d = instance_z_score(test_x_2d, axis=(1, 2))
                
                in_dim = train_x_1d.shape[2]
                img_size = train_x_2d.shape[1]
                in_chans = train_x_2d.shape[3]

                for seed in args.seeds:
                    set_seed(seed)

                    rdir = pjoin(args.output_root, img_type, f"seed_{seed}", dataset_name)
                    os.makedirs(_windows_long_path(rdir), exist_ok=True)

                    g = torch.Generator()
                    g.manual_seed(seed)

                    print(
                        f"  BatchSize: {args.batch_size} | "
                        f"BackboneDepthGrid: {sweep['backbone_depth']} | "
                        f"WeightDecay: {args.weight_decay}"
                    )

                    tr_loader = DataLoader(
                        AIMDataset(train_x_1d, train_x_2d, train_y_all),
                        batch_size=args.batch_size,
                        shuffle=True,
                        generator=g
                    )
    
                    grid_mode = "uniform_hpo"

                    combos = list(product(
                        sweep["stages"], sweep["patch_grid"], sweep["backbone_depth"],
                        sweep["d_model"], sweep["dense_units"],
                        sweep["mha_heads"], sweep["mha_modules"], sweep["lr"],
                        sweep["dropout"], sweep["temperature"], sweep["lambda_contrast"],
                        sweep["label_smoothing"], sweep["optimizer"],
                        sweep["w_main"], sweep["w_ts"], sweep["w_img"]
                    ))
                
                    print(f"  GridMode: {grid_mode} | total combos: {len(combos)}")
        
                    overall_best = {
                        "selection_score": -float("inf"),
                        "train_loss_at_selected": float("inf"),
                        "train_acc_at_selected": -1.0,
                        "state": None,
                        "hparams": None,
                        "selected_epoch": None,
                        "selection_metric": None,
                        "selection_value": None,
                        "row_idx": None,
                        "final_test_acc": None,
                        "final_test_loss": None,
                    }
                    rows = []
                    
                    for i, (st, pg, bd, dm, du, mh, mm, lr, do, temp, lc, ls, opt, wm, wt, wi) in enumerate(combos, 1):
                        bd = int(bd)
                        print(f"\n>> Datset:{dataset_name} | Combo: count={i}/{len(combos)}, Stage={st}, Patch_Grid={pg}, BackboneDepth={bd}, dense_units={du}, mha_modules={mm}, Learning_rate={lr}, dropout={do}, tau={temp}, lambda_contrast={lc}, label_smoothing={ls}, W=[{wm},{wt},{wi}]")
                        model = AIM_TSformer_Agg(
                            n_classes,
                            seq_len, in_dim, img_size, in_chans,
                            st, dm, mh, mm,
                            args.backbone, pg, bd, args.conv_dw_kernel,
                            du, do, temp, lc, wm, wt, wi,
                            fusion_gate_init=args.fusion_gate_init,
                            symmetric_fusion_norm=args.symmetric_fusion_norm,
                            modality_dropout=args.modality_dropout,
                            use_gated_fusion=args.use_gated_fusion,
                            use_contrast_projection=args.use_contrast_projection,
                            use_segment_embedding=args.use_segment_embedding,
                            max_inner=args.max_inner,
                            label_smoothing=ls
                        ).to(DEVICE)
                        
                        res = train_one_trainloss(
                            model, tr_loader, args.epochs, lr, opt, args.weight_decay,
                            early_patience=args.early_patience,
                            eval_source=args.eval_source,
                        )
                        
                        rows.append({
                            'seq_len': seq_len,
                            "grid_mode": grid_mode,
                            "eval_source": args.eval_source,
                            "use_gated_fusion": args.use_gated_fusion,
                            "use_contrast_projection": args.use_contrast_projection,
                            "use_segment_embedding": args.use_segment_embedding,
                            "max_inner": args.max_inner,
                            "stages": st, "patch_grid": pg, "d_model": dm,
                            "dense_units": du, "mha_heads": mh, "mha_modules": mm,
                            "lr": lr, "dropout": do, "temperature": temp, "lambda_contrast": lc,
                            "selected_epoch": res["best_epoch"],
                            "selection_metric": res["selection_metric"],
                            "selection_value": res["selection_value"],
                            "train_loss_at_selected_epoch": res["best_train_loss"],
                            "train_acc_at_selected_epoch": res["best_train_acc"],
                            "backbone_depth": bd,
                            "weight_decay": args.weight_decay,
                            "label_smoothing": ls,
                            "is_selected": False,
                            "final_test_acc": np.nan,
                            "final_test_loss": np.nan,
                        })
                        combo_is_better = (
                            res["selection_score"] > overall_best["selection_score"] + 1e-12
                            or (
                                abs(res["selection_score"] - overall_best["selection_score"]) <= 1e-12
                                and res["best_train_loss"] < overall_best["train_loss_at_selected"] - 1e-12
                            )
                        )

                        if combo_is_better:
        
                            hp_save = {
                                       "dataset": dataset_name,
                                       "img_type": img_type,
                                       "seed": seed,
                                       "seq_len": seq_len,
                                       "batch_size": args.batch_size,
                                       "n_classes": n_classes,
                                       "in_dim": in_dim,
                                       "img_size": img_size,
                                       "in_chans": in_chans,
                                       "stages": st, "patch_grid": pg, "d_model": dm, "dense_units": du,
                                       "mha_heads": mh, "mha_modules": mm,
                                       "lr": lr, "dropout": do, "lambda_contrast": lc, "optimizer": opt,
                                       "backbone": args.backbone,
                                       "backbone_depth": bd,
                                       "conv_dw_kernel": args.conv_dw_kernel,
                                       "temperature": temp,
                                       "weight_decay": args.weight_decay,
                                       "label_smoothing": ls,
                                       "epochs": args.epochs,
                                       "early_patience": args.early_patience,
                                       "early_stop_metric": res["selection_metric"],
                                       "selected_epoch": res["best_epoch"],
                                       "selection_value": res["selection_value"],
                                       "w_main": wm, "w_ts": wt, "w_img": wi,
                                       "grid_mode": grid_mode, "eval_source": args.eval_source,
                                       "fusion_gate_init": args.fusion_gate_init,
                                       "symmetric_fusion_norm": args.symmetric_fusion_norm,
                                       "modality_dropout": args.modality_dropout,
                                       "use_gated_fusion": args.use_gated_fusion,
                                       "use_contrast_projection": args.use_contrast_projection,
                                       "use_segment_embedding": args.use_segment_embedding,
                                       "max_inner": args.max_inner
                                       }
        
                            overall_best.update(
                                selection_score=res["selection_score"],
                                train_loss_at_selected=res["best_train_loss"],
                                train_acc_at_selected=res["best_train_acc"],
                                state=res["state"],
                                hparams=hp_save,
                                selected_epoch=res["best_epoch"],
                                selection_metric=res["selection_metric"],
                                selection_value=res["selection_value"],
                                row_idx=len(rows) - 1,
                            )

                        del model
                        if DEVICE == "cuda":
                            torch.cuda.empty_cache()
        
                    df = pd.DataFrame(rows)
                    summary_csv = pjoin(rdir, f"summary_{img_type}_seed{seed}_{args.backbone}_L{seq_len}.csv")
                    df.to_csv(_windows_long_path(summary_csv), index=False)
                    
                    if overall_best["state"] is None or overall_best["hparams"] is None:
                        raise RuntimeError("No train-loss-selected checkpoint state is available.")

                    hp = overall_best["hparams"]

                    best_dir = pjoin(rdir, f"selected_L{seq_len}")
                    hparams_path = pjoin(best_dir, "best_hparams.json")

                    best_model = None
                    os.makedirs(_windows_long_path(best_dir), exist_ok=True)
                    with open(_windows_long_path(hparams_path), "w", encoding="utf-8") as f:
                        json.dump(hp, f, indent=2, ensure_ascii=False)

                    try:
                        best_model = build_model_from_hparams(
                            hp, n_classes, seq_len, in_dim, img_size, in_chans
                        ).to(DEVICE)
                        best_model.load_state_dict(overall_best["state"], strict=True)
                        best_model.eval()

                        test_y_all = transform_labels_with_mapping(
                            pjoin(dpath, f"{dataset_name}_TEST.ts"), label_map
                        )
                        te_loader = DataLoader(
                            AIMDataset(test_x_1d, test_x_2d, test_y_all),
                            batch_size=args.batch_size,
                            shuffle=False
                        )
                        final_eval = evaluate_test_once(
                            best_model, te_loader,
                            eval_source=hp.get("eval_source", args.eval_source)
                        )
                        overall_best["final_test_acc"] = final_eval["test_acc"]
                        overall_best["final_test_loss"] = final_eval["test_loss"]

                        if overall_best["row_idx"] is not None:
                            df.loc[overall_best["row_idx"], "is_selected"] = True
                            df.loc[overall_best["row_idx"], "final_test_acc"] = final_eval["test_acc"]
                            df.loc[overall_best["row_idx"], "final_test_loss"] = final_eval["test_loss"]
                            df.to_csv(_windows_long_path(summary_csv), index=False)

                        with open(_windows_long_path(pjoin(best_dir, "final_test_metrics.json")), "w", encoding="utf-8") as f:
                            json.dump({
                                "dataset": dataset_name,
                                "seed": seed,
                                "seq_len": seq_len,
                                "eval_source": final_eval["eval_source"],
                                "test_acc": final_eval["test_acc"],
                                "test_loss": final_eval["test_loss"],
                                "test_macro_f1": final_eval["test_macro_f1"],
                                "test_balanced_acc": final_eval["test_balanced_acc"],
                                "selected_epoch": overall_best["selected_epoch"],
                                "selection_metric": overall_best["selection_metric"],
                                "selection_value": overall_best["selection_value"],
                            }, f, indent=2, ensure_ascii=False)

                    except (RuntimeError, KeyError, TypeError, FileNotFoundError) as e:
                        print("[WARN] selected model evaluation failed.")
                        print(f"       {type(e).__name__}: {e}")
                        raise
                    finally:
                        if best_model is not None:
                            del best_model
                        if DEVICE == "cuda":
                            torch.cuda.empty_cache()





if __name__ == "__main__":
    DATASET_ROOT = os.environ.get(
        "AIM_DATASET_ROOT",
        pjoin(os.path.dirname(os.path.abspath(__file__)), "dataset"),
    )
    TS_TYPE = "Multivariate_ts"

    OUTPUT_ROOT = os.environ.get(
        "AIM_OUTPUT_ROOT",
        pjoin(
            os.path.dirname(os.path.abspath(__file__)),
            "aimtsformer_results_github_public_final",
        ),
    )
    
    IMG_TYPES = ["aim_original_original"]

    RUN_ONLY_DATASET_ABBRS = []
    RUN_ONLY_DATASETS = []

    DATASETS_TO_SKIP = []

    DATASET_ABBR = {
        "AWR": "ArticularyWordRecognition",
        "BM": "BasicMotions",
        "EP": "Epilepsy",
        "ER": "ERing",
        "FD": "FaceDetection",
        "FM": "FingerMovements",
        "HW": "Handwriting",
        "LIB": "Libras",
        "LSST": "LSST",
        "NA": "NATOPS",
        "PEMS": "PEMS-SF",
        "PD": "PenDigits",
        "PM": "PhonemeSpectra",
        "RS": "RacketSports",

        "DDG": "DuckDuckGeese",
        "UWG": "UWaveGestureLibrary",

        "AF": "AtrialFibrillation",
        "CR": "Cricket",
        "EC": "EthanolConcentration",
        "HMD": "HandMovementDirection",
        "HB": "Heartbeat",
        "SRS1": "SelfRegulationSCP1",
        "SRS2": "SelfRegulationSCP2",
        "SWJ": "StandWalkJump",
    }

    SEEDS = [42, 43, 44, 45, 46]

    EPOCHS = 300
    EARLY_PATIENCE = 30

    LABEL_SMOOTHING_GRID = [0.0]
    EARLY_STOP_METRIC = "train_loss"

    BACKBONE = "convmixer"
    CONV_DW_KERNEL = 9

    TEMPERATURE_GRID = [1.0]

    MAX_INNER = None

    OPTIMIZER_NAME = "adamw"
    WEIGHT_DECAY = 1e-2
    BATCH_SIZE = 32
    EVAL_SOURCE = "fusion"
    USE_GATED_FUSION = True
    FUSION_GATE_INIT = 1.0
    SYMMETRIC_FUSION_NORM = True
    MODALITY_DROPOUT = 0.1
    USE_CONTRAST_PROJECTION = True
    USE_SEGMENT_EMBEDDING = True
    W_MAIN = [1.0]
    W_TS = [1.0]
    W_IMG = [1.0]

    HPO_GRID = {
        "stages":      [1],
        "patch_grid":  [4],
        "backbone_depth": [1, 2, 4],
        "d_model":     [[128]],
        "dense_units": [128],
        "mha_heads":   [[8]],
        "mha_modules": [1, 2, 4],
        "dropout":     [0.1],
        "lr":          [0.0005],
        "temperature": TEMPERATURE_GRID,
        "label_smoothing": LABEL_SMOOTHING_GRID,
        "optimizer":   [OPTIMIZER_NAME],
        "w_main": W_MAIN,
        "w_ts":   W_TS,
        "w_img":  W_IMG,
        "lambda_contrast": [1.0],
    }

    def resolve_dataset_selection(abbrs, names, abbr_map):
        selected = []

        for abbr in abbrs:
            key = str(abbr).strip()
            if key not in abbr_map:
                raise ValueError(
                    f"Unknown dataset abbreviation: {key}. "
                    f"Available abbreviations: {sorted(abbr_map.keys())}"
                )
            selected.append(abbr_map[key])

        for name in names:
            selected.append(str(name).strip())

        unique = []
        seen = set()
        for name in selected:
            if name and name not in seen:
                unique.append(name)
                seen.add(name)

        return unique

    SELECTED_DATASETS = resolve_dataset_selection(
        RUN_ONLY_DATASET_ABBRS,
        RUN_ONLY_DATASETS,
        DATASET_ABBR,
    )

    DATASETS_TO_RUN = SELECTED_DATASETS if SELECTED_DATASETS else None

    ARGS = SimpleNamespace(
        dataset_root=DATASET_ROOT,
        ts_type=TS_TYPE,
        ts_preprocess_mode="original",
        sequence_lengths=None,
        output_root=OUTPUT_ROOT,
        datasets_to_run=DATASETS_TO_RUN,
        datasets_to_skip=DATASETS_TO_SKIP,

        backbone=BACKBONE,
        conv_dw_kernel=CONV_DW_KERNEL,
        img_types=IMG_TYPES,
        max_inner=MAX_INNER,

        eval_source=EVAL_SOURCE,
        use_gated_fusion=USE_GATED_FUSION,
        fusion_gate_init=FUSION_GATE_INIT,
        symmetric_fusion_norm=SYMMETRIC_FUSION_NORM,
        modality_dropout=MODALITY_DROPOUT,
        use_contrast_projection=USE_CONTRAST_PROJECTION,
        use_segment_embedding=USE_SEGMENT_EMBEDDING,

        seeds=SEEDS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        label_smoothing=LABEL_SMOOTHING_GRID[0],
        temperature=TEMPERATURE_GRID[0],
        early_patience=EARLY_PATIENCE,
        early_stop_metric=EARLY_STOP_METRIC,
        optimizer=OPTIMIZER_NAME,
        weight_decay=WEIGHT_DECAY,
        use_amp=USE_AMP,
    )

    print("\n========== RUN CONFIG ==========")
    print("Protocol      : balanced fusion")
    print(f"Img types     : {IMG_TYPES}")
    print(f"Eval source   : {EVAL_SOURCE}")
    print(f"Seeds         : {SEEDS}")
    print(f"Epochs        : {EPOCHS}")
    print(f"Early metric  : {ARGS.early_stop_metric}")
    print(f"Label smoothing grid: {LABEL_SMOOTHING_GRID}")
    print(f"Temperature grid: {TEMPERATURE_GRID}")
    print(f"Max inner default: {MAX_INNER}")
    print(f"Backbone      : {ARGS.backbone}")
    print(f"MHA module grid: {HPO_GRID['mha_modules']}")
    print(f"Backbone depth grid: {HPO_GRID['backbone_depth']}")
    print(f"Datasets to run: {DATASETS_TO_RUN if DATASETS_TO_RUN is not None else 'ALL'}")
    print(f"Datasets to skip: {DATASETS_TO_SKIP}")
    print(f"HPO combos    : {count_combos(HPO_GRID)}")
    print("================================\n")

    main_grid(ARGS, HPO_GRID)
