from pathlib import Path
import gc
import os
import time
import traceback

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import torch

if os.environ.get("AIM_FORCE_CPU", "").strip().lower() in {"1", "true", "yes"}:
    
    
    torch.cuda.is_available = lambda: False

from sktime.classification.deep_learning import TapNetClassifierTorch
from sktime.classification.deep_learning.base._base_torch import PytorchDataset
from torch.utils.data import DataLoader

from rocket_hydra.common_sktime_mtsc_utils import (
    DEFAULT_DATASETS,
    format_seconds,
    get_existing_rows,
    load_uea_numpy3d,
    make_base_result_row,
    remove_same_key,
    row_already_ok,
    save_experiment_tables,
    set_seed,
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
EPOCHS = int(os.environ.get("AIM_EPOCHS", "100"))
BATCH_SIZE = int(os.environ.get("AIM_BATCH_SIZE", "16"))
LR = 1e-3
NORMALIZE_BY_TRAIN = True
SKIP_COMPLETED = True
OUTPUT_ROOT = Path(os.environ.get(
    "AIM_TAPNET_OUTPUT_ROOT",
    str(Path(__file__).resolve().parent / "outputs" /
        f"TapNet_sktime_shuffled_{EPOCHS}epoch_batch{BATCH_SIZE}"),
))
DEVICE = torch.device(
    "cuda"
    if not os.environ.get("AIM_FORCE_CPU", "").strip().lower() in {"1", "true", "yes"}
    and torch.cuda.is_available()
    else "cpu"
)


class DeviceTapNetClassifierTorch(TapNetClassifierTorch):

    execution_device = torch.device("cpu")

    def _build_dataloader(self, X, y=None):
        # The official TRAIN/TEST split stays fixed. Shuffle TRAIN minibatches
        # only; prediction must preserve the TEST sample order.
        return DataLoader(
            PytorchDataset(X, y),
            batch_size=self.batch_size,
            shuffle=y is not None,
        )

    def _build_network(self, X, y):
        network = super()._build_network(X, y)
        return network.to(self.execution_device)

    def _move_inputs(self, inputs):
        return {
            key: value.to(self.execution_device, non_blocking=True)
            if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

    def _run_epoch(self, epoch, dataloader):
        losses = []
        metric_values = {name: [] for name in (self._metrics_objects or {})}
        for inputs, outputs in dataloader:
            inputs = self._move_inputs(inputs)
            outputs = outputs.to(self.execution_device, non_blocking=True)
            y_pred = self.network(**inputs)
            loss = self._criterion(y_pred, outputs)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            losses.append(loss.item())
            if self._metrics_objects:
                with torch.no_grad():
                    for metric_name, metric_obj in self._metrics_objects.items():
                        metric_value = metric_obj(y_pred, outputs)
                        metric_values[metric_name].append(metric_value.item())

        epoch_loss = float(np.average(losses))
        if self._schedulers:
            for scheduler in self._schedulers:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(epoch_loss)
                else:
                    scheduler.step()
        if self.verbose:
            message = f"Epoch {epoch + 1}: Loss: {epoch_loss}"
            for metric_name, values in metric_values.items():
                message += f", {metric_name}: {np.average(values):.4f}"
            print(message, flush=True)

    def _predict_proba(self, X):
        self.network.eval()
        dataloader = self._build_dataloader(X)
        predictions = []
        with torch.no_grad():
            for inputs in dataloader:
                inputs = self._move_inputs(inputs)
                predictions.append(self.network(**inputs).detach().cpu())
        y_pred = torch.cat(predictions, dim=0)
        if self._validated_activation is None:
            y_pred = torch.softmax(y_pred, dim=-1)
        return y_pred.numpy()


def retryable_cuda_oom(error: BaseException) -> bool:
    enabled = os.environ.get("AIM_RETRY_CUDA_OOM", "").strip().lower()
    return enabled in {"1", "true", "yes"} and (
        isinstance(error, torch.cuda.OutOfMemoryError)
        or "cuda" in str(error).lower() and "out of memory" in str(error).lower()
    )


def run_one(dataset_name: str, seed: int) -> dict:
    set_seed(seed)
    X_train, y_train, X_test, y_test, label_map, dpath, loader_name = load_uea_numpy3d(
        BASE_PARENT,
        dataset_name,
        ts_type=TS_TYPE,
        fixed_length=None,
        normalize_by_train=NORMALIZE_BY_TRAIN,
    )
    classifier = DeviceTapNetClassifierTorch(
        num_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        random_state=int(seed),
        verbose=False,
    )
    classifier.execution_device = DEVICE
    started = time.perf_counter()
    classifier.fit(X_train, y_train)
    fitted = time.perf_counter()
    y_pred = classifier.predict(X_test)
    finished = time.perf_counter()

    row = make_base_result_row(
        "TapNet", dataset_name, seed, X_train, y_train, X_test, y_test,
        label_map, dpath, loader_name, None, NORMALIZE_BY_TRAIN,
    )
    row.update({
        "final_test_acc": float(accuracy_score(y_test, y_pred)),
        "final_test_macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "final_test_balanced_acc": float(balanced_accuracy_score(y_test, y_pred)),
        "fit_seconds": float(fitted - started),
        "predict_seconds": float(finished - fitted),
        "total_seconds": float(finished - started),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "backend": "sktime.TapNetClassifierTorch+device_adapter",
        "tapnet_implementation": "sktime_non_prototypical",
        "train_shuffle": True,
        "device": str(DEVICE),
        "status": "OK",
        "error": "",
    })
    return row


def main() -> None:
    print(
        f"TapNet sktime variant | train_shuffle=True | epochs={EPOCHS} "
        f"| batch_size={BATCH_SIZE} | device={DEVICE} | output={OUTPUT_ROOT}",
        flush=True,
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = get_existing_rows(OUTPUT_ROOT / "TapNet_per_seed_results.csv")
    for dataset_name in DATASETS:
        for seed in SEEDS:
            print(f"\nTapNet | dataset={dataset_name} | seed={seed} | epochs={EPOCHS}", flush=True)
            if SKIP_COMPLETED and row_already_ok(rows, "TapNet", dataset_name, seed):
                print("[SKIP] Existing OK row with all three metrics.", flush=True)
                continue
            try:
                row = run_one(dataset_name, seed)
                print(
                    f"[OK] acc={row['final_test_acc']:.6f} | "
                    f"macro_f1={row['final_test_macro_f1']:.6f} | "
                    f"bacc={row['final_test_balanced_acc']:.6f} | "
                    f"time={format_seconds(row['total_seconds'])}",
                    flush=True,
                )
            except Exception as error:
                if retryable_cuda_oom(error):
                    torch.cuda.empty_cache()
                    raise
                traceback.print_exc()
                row = {
                    "model": "TapNet", "dataset": dataset_name, "seed": int(seed),
                    "epochs": EPOCHS, "status": "ERROR", "error": repr(error),
                }
            rows = remove_same_key(rows, "TapNet", dataset_name, seed)
            rows.append(row)
            save_experiment_tables(rows, OUTPUT_ROOT, "TapNet")
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
