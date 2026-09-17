from __future__ import annotations

import argparse
import csv
import ctypes
import json
import math
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATASET_ROOT = Path(
    os.environ.get("AIM_DATASET_ROOT", str(HERE.parent / "dataset"))
) / "Multivariate_ts"
OUTPUT_ROOT = HERE / "outputs"
STATE_PATH = HERE / "rpm_srpm_pipeline_state.json"
SEEDS = (42, 43, 44, 45, 46)
WINDOW = 2

DATASETS = (
    "ArticularyWordRecognition", "AtrialFibrillation", "BasicMotions", "Cricket",
    "DuckDuckGeese", "Epilepsy", "ERing", "EthanolConcentration", "FaceDetection",
    "FingerMovements", "HandMovementDirection", "Handwriting", "Heartbeat", "Libras",
    "LSST", "NATOPS", "PEMS-SF", "PenDigits", "PhonemeSpectra", "RacketSports",
    "SelfRegulationSCP1", "SelfRegulationSCP2", "StandWalkJump", "UWaveGestureLibrary",
)
EXCLUDED = {
    "rpm": {"PenDigits", "RacketSports"},
    "srpm": {"PenDigits"},
}
RUNNERS = {
    "rpm": HERE / "rpmcnn_5seed.py",
    "srpm": HERE / "srpmcnn_5seed.py",
}
MODEL_NAMES = {"rpm": "RPM-CNN", "srpm": "SRPM-CNN"}


def configure_windows_error_mode() -> None:
    if os.name != "nt":
        return
    mode = 0x0001 | 0x0002 | 0x8000
    ctypes.windll.kernel32.SetErrorMode(mode)
    old_mode = ctypes.c_uint()
    setter = getattr(ctypes.windll.kernel32, "SetThreadErrorMode", None)
    if setter is not None:
        setter(mode, ctypes.byref(old_mode))


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(**changes) -> dict:
    state = load_state()
    state.update(changes)
    temp = STATE_PATH.with_suffix(".partial.json")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, STATE_PATH)
    return state


@lru_cache(maxsize=None)
def load_shape(dataset: str) -> tuple[int, int, int, int]:
    dataset_dir = DATASET_ROOT / dataset
    with (dataset_dir / f"{dataset}_train_df.pkl").open("rb") as handle:
        train = pickle.load(handle)
    with (dataset_dir / f"{dataset}_test_df.pkl").open("rb") as handle:
        test = pickle.load(handle)
    return int(train.shape[0]), int(test.shape[0]), int(train.shape[1]), int(train.shape[2])


def estimated_bytes(dataset: str, transform: str) -> int:
    n_train, n_test, length, channels = load_shape(dataset)
    if transform == "rpm":
        size = length
    else:
        size = (length - 4 * WINDOW) // 2
        if size < 1:
            return 0
    return (n_train + n_test) * size * size * channels * 4


def cache_paths(dataset: str, transform: str) -> list[Path]:
    _, _, length, _ = load_shape(dataset)
    dataset_dir = DATASET_ROOT / dataset
    if transform == "rpm":
        train = dataset_dir / f"{dataset}_rpm_train_{length}.npy"
        test = dataset_dir / f"{dataset}_rpm_test_{length}.npy"
    else:
        train = dataset_dir / f"{dataset}_srpm_train_win{WINDOW}_len{length}.npy"
        test = dataset_dir / f"{dataset}_srpm_test_win{WINDOW}_len{length}.npy"
    return [train, test, train.with_suffix(".stats.npz")]


def result_csv(transform: str) -> Path:
    model = MODEL_NAMES[transform]
    return OUTPUT_ROOT / model / f"{model}_per_seed_results.csv"


def completed_seeds(dataset: str, transform: str) -> set[int]:
    path = result_csv(transform)
    if not path.exists():
        return set()
    completed = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("dataset") != dataset or row.get("status") != "OK":
                continue
            required = (
                row.get("final_test_acc", ""),
                row.get("final_test_macro_f1", ""),
                row.get("final_test_balanced_acc", ""),
            )
            try:
                values = [float(value) for value in required]
                seed = int(float(row.get("seed", "nan")))
            except Exception:
                continue
            if all(math.isfinite(value) for value in values):
                completed.add(seed)
    return completed


def gpu_snapshot() -> tuple[int, int]:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.free",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    first = result.stdout.strip().splitlines()[0]
    util_text, free_text = [part.strip() for part in first.split(",", 1)]
    return int(float(util_text)), int(float(free_text))


def wait_for_gpu_idle(max_util: int, min_free_mb: int, stable_samples: int, interval: int) -> None:
    stable = 0
    last_report = 0.0
    while stable < stable_samples:
        try:
            util, free_mb = gpu_snapshot()
            acceptable = util <= max_util and free_mb >= min_free_mb
        except Exception as exc:
            util, free_mb, acceptable = -1, -1, False
            if time.time() - last_report >= 300:
                print(f"[GPU WAIT] nvidia-smi error: {exc!r}", flush=True)
        stable = stable + 1 if acceptable else 0
        save_state(
            status="WAITING_GPU_IDLE",
            gpu_utilization_percent=util,
            gpu_free_mb=free_mb,
            idle_stable_samples=stable,
            updated_at=now_text(),
        )
        if time.time() - last_report >= 300 or stable == stable_samples:
            print(
                f"[GPU WAIT] util={util}% free={free_mb}MB stable={stable}/{stable_samples}",
                flush=True,
            )
            last_report = time.time()
        if stable < stable_samples:
            time.sleep(interval)


def run_checked(
    command: list[str],
    env: dict[str, str] | None = None,
    runtime_min_free_mb: int = 0,
    runtime_low_samples: int = 2,
    runtime_poll_seconds: int = 5,
) -> None:
    print(f"[COMMAND] {' '.join(command)}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=HERE,
        env=env,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if runtime_min_free_mb <= 0:
        returncode = process.wait()
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
        return

    low_samples = 0
    while True:
        returncode = process.poll()
        if returncode is not None:
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)
            return
        time.sleep(runtime_poll_seconds)
        try:
            utilization, free_mb = gpu_snapshot()
            low_samples = low_samples + 1 if free_mb < runtime_min_free_mb else 0
        except Exception:
            continue
        if low_samples < runtime_low_samples:
            continue
        save_state(
            status="STOPPING_FOR_GPU_MEMORY_PRESSURE",
            gpu_utilization_percent=utilization,
            gpu_free_mb=free_mb,
            runtime_low_samples=low_samples,
            stopped_child_pid=process.pid,
            updated_at=now_text(),
        )
        print(
            f"[GPU RUNTIME GUARD] free={free_mb}MB for {low_samples} samples; "
            f"stopping child PID {process.pid} and retrying later",
            flush=True,
        )
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise subprocess.CalledProcessError(75, command)


def remove_large_cache(dataset: str, transform: str, retain_max_gib: float) -> list[str]:
    size_gib = estimated_bytes(dataset, transform) / 2**30
    if size_gib <= retain_max_gib:
        return []
    removed = []
    dataset_dir = (DATASET_ROOT / dataset).resolve()
    for path in cache_paths(dataset, transform):
        resolved = path.resolve()
        if resolved.parent != dataset_dir:
            raise RuntimeError(f"Cache cleanup escaped dataset directory: {resolved}")
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed


def run_pipeline(args: argparse.Namespace) -> None:
    configure_windows_error_mode()
    save_state(
        status="STARTED",
        started_at=now_text(),
        python=sys.executable,
        seeds=list(SEEDS),
        execution_device="cpu" if args.force_cpu else "cuda_when_idle",
        cpu_threads=args.cpu_threads if args.force_cpu else None,
        retain_cache_max_gib=args.retain_cache_max_gib,
        note=(
            "Original-length float32 transforms. Large caches are deleted only after all five "
            "metric-complete seeds are saved; small caches are retained."
        ),
    )

    selected = list(DATASETS)
    if args.datasets:
        requested = [item.strip() for item in args.datasets.split(",") if item.strip()]
        unknown = sorted(set(requested) - set(DATASETS))
        if unknown:
            raise ValueError(f"Unknown datasets: {unknown}")
        selected = requested
    selected.sort(key=lambda name: sum(
        estimated_bytes(name, transform)
        for transform in ("rpm", "srpm") if name not in EXCLUDED[transform]
    ))

    completed_jobs = []
    failed_jobs = []
    removed_caches = []
    for dataset in selected:
        for transform in ("rpm", "srpm"):
            model = MODEL_NAMES[transform]
            if dataset in EXCLUDED[transform]:
                print(f"[N/A] {model} {dataset}", flush=True)
                continue
            done_before = completed_seeds(dataset, transform)
            if set(SEEDS).issubset(done_before):
                print(f"[SKIP] {model} {dataset}: five complete seeds already exist", flush=True)
                completed_jobs.append(f"{model}:{dataset}")
                removed_caches.extend(remove_large_cache(dataset, transform, args.retain_cache_max_gib))
                continue

            save_state(
                status="GENERATING",
                current_model=model,
                current_dataset=dataset,
                completed_jobs=completed_jobs,
                failed_jobs=failed_jobs,
                updated_at=now_text(),
            )
            run_checked([
                sys.executable,
                str(HERE / "generate_rpm_srpm_cache.py"),
                "--dataset", dataset,
                "--transform", transform,
                "--self-test",
            ])

            if not args.force_cpu:
                wait_for_gpu_idle(
                    max_util=args.gpu_max_util,
                    min_free_mb=args.gpu_min_free_mb,
                    stable_samples=args.gpu_stable_samples,
                    interval=args.gpu_poll_seconds,
                )
            save_state(
                status="TRAINING",
                current_model=model,
                current_dataset=dataset,
                execution_device="cpu" if args.force_cpu else "cuda",
                completed_jobs=completed_jobs,
                failed_jobs=failed_jobs,
                updated_at=now_text(),
            )
            env = os.environ.copy()
            if args.force_cpu:
                env["AIM_FORCE_CPU"] = "1"
                thread_text = str(args.cpu_threads)
                env.update({
                    "OMP_NUM_THREADS": thread_text,
                    "MKL_NUM_THREADS": thread_text,
                    "OPENBLAS_NUM_THREADS": thread_text,
                    "NUMEXPR_NUM_THREADS": thread_text,
                })
            else:
                env.pop("AIM_FORCE_CPU", None)
            env["AIM_DATASETS"] = dataset
            env["AIM_SEEDS"] = ",".join(map(str, SEEDS))
            env["AIM_EPOCHS"] = "300"
            env["AIM_BATCH_SIZE"] = "32"
            env["PYTHONFAULTHANDLER"] = "1"
            runner_error = None
            done_after = set()
            for attempt in range(1, args.runner_retries + 1):
                try:
                    run_checked(
                        [sys.executable, str(RUNNERS[transform])],
                        env=env,
                        runtime_min_free_mb=(
                            0 if args.force_cpu else args.gpu_runtime_min_free_mb
                        ),
                        runtime_low_samples=args.gpu_runtime_low_samples,
                        runtime_poll_seconds=args.gpu_runtime_poll_seconds,
                    )
                    runner_error = None
                except Exception as exc:
                    runner_error = exc
                done_after = completed_seeds(dataset, transform)
                if set(SEEDS).issubset(done_after):
                    break
                if attempt < args.runner_retries and not args.force_cpu:
                    save_state(
                        status="WAITING_TO_RETRY",
                        current_model=model,
                        current_dataset=dataset,
                        retry_attempt=attempt + 1,
                        metric_complete_seeds=sorted(done_after),
                        last_runner_error=repr(runner_error) if runner_error else None,
                        completed_jobs=completed_jobs,
                        failed_jobs=failed_jobs,
                        updated_at=now_text(),
                    )
                    wait_for_gpu_idle(
                        max_util=args.gpu_max_util,
                        min_free_mb=args.gpu_min_free_mb,
                        stable_samples=args.gpu_stable_samples,
                        interval=args.gpu_poll_seconds,
                    )

            if not set(SEEDS).issubset(done_after):
                failed_jobs.append({
                    "job": f"{model}:{dataset}",
                    "error": (
                        f"metric-complete seeds={sorted(done_after)}; "
                        f"last_runner_error={runner_error!r}"
                    ),
                })
                save_state(
                    status="JOB_INCOMPLETE",
                    current_model=model,
                    current_dataset=dataset,
                    completed_jobs=completed_jobs,
                    failed_jobs=failed_jobs,
                    updated_at=now_text(),
                )
                continue

            completed_jobs.append(f"{model}:{dataset}")
            removed_caches.extend(remove_large_cache(dataset, transform, args.retain_cache_max_gib))
            save_state(
                status="RUNNING",
                current_model=model,
                current_dataset=dataset,
                completed_jobs=completed_jobs,
                failed_jobs=failed_jobs,
                removed_large_caches=removed_caches,
                updated_at=now_text(),
            )

    final_status = "COMPLETE" if not failed_jobs else "COMPLETE_WITH_ERRORS"
    save_state(
        status=final_status,
        finished_at=now_text(),
        current_model=None,
        current_dataset=None,
        completed_jobs=completed_jobs,
        failed_jobs=failed_jobs,
        removed_large_caches=removed_caches,
        updated_at=now_text(),
    )
    print(f"[PIPELINE {final_status}] jobs={len(completed_jobs)} failures={len(failed_jobs)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="")
    parser.add_argument("--retain-cache-max-gib", type=float, default=1.0)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--gpu-max-util", type=int, default=10)
    parser.add_argument("--gpu-min-free-mb", type=int, default=14000)
    parser.add_argument("--gpu-stable-samples", type=int, default=4)
    parser.add_argument("--gpu-poll-seconds", type=int, default=30)
    parser.add_argument("--gpu-runtime-min-free-mb", type=int, default=1024)
    parser.add_argument("--gpu-runtime-low-samples", type=int, default=2)
    parser.add_argument("--gpu-runtime-poll-seconds", type=int, default=5)
    parser.add_argument("--runner-retries", type=int, default=3)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be at least 1")
    if (
        args.runner_retries < 1
        or args.gpu_runtime_low_samples < 1
        or args.gpu_runtime_poll_seconds < 1
    ):
        parser.error("--runner-retries must be at least 1")
    try:
        run_pipeline(args)
    except Exception as exc:
        save_state(status="PIPELINE_FAILED", error=repr(exc), updated_at=now_text())
        raise


if __name__ == "__main__":
    main()
