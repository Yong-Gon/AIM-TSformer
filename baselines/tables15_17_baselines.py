from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ROCKET_PYTHON = Path(os.environ.get("AIM_ROCKET_PYTHON", sys.executable))
ROCKET_MODELS = {"ROCKET", "MiniRocket", "MultiRocket", "Hydra"}
MODEL_SCRIPTS = {
    "MLP": "mlp_5seed.py",
    "FCN": "fcn_5seed.py",
    "ResNet": "resnet_5seed.py",
    "TSF": "tsf_5seed.py",
    "TapNet": "tapnet_5seed.py",
    "TimesNet": "timesnet_5seed.py",
    "ConvTran": "convtran_5seed.py",
    "KATN": "katn_5seed.py",
    "RPM-CNN": "rpmcnn_5seed.py",
    "SRPM-CNN": "srpmcnn_5seed.py",
    "ROCKET": "rocket_hydra/rocket_5seed.py",
    "MiniRocket": "rocket_hydra/minirocket_5seed.py",
    "MultiRocket": "rocket_hydra/multirocket_5seed.py",
    "Hydra": "rocket_hydra/hydra_5seed.py",
    "DTW_I": "dtwi_5seed.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(MODEL_SCRIPTS), default=list(MODEL_SCRIPTS))
    parser.add_argument(
        "--rocket-python",
        type=Path,
        default=DEFAULT_ROCKET_PYTHON,
        help="Interpreter containing aeon/sktime for the ROCKET-family models.",
    )
    parser.add_argument("--skip-gpu-wait", action="store_true")
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Hide CUDA from child processes and run PyTorch baselines on CPU.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=1,
        help="CPU thread limit exported to numerical libraries when --force-cpu is used.",
    )
    parser.add_argument("--idle-memory-mib", type=int, default=2500)
    parser.add_argument("--idle-utilization", type=int, default=10)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--state-file", default="orchestrator_state.json")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def configure_windows_error_mode() -> None:
    if sys.platform != "win32":
        return
    
    mode = 0x0001 | 0x0002 | 0x8000
    kernel32 = ctypes.windll.kernel32
    kernel32.SetErrorMode(mode)
    old_mode = ctypes.c_uint()
    set_thread_error_mode = getattr(kernel32, "SetThreadErrorMode", None)
    if set_thread_error_mode is not None:
        set_thread_error_mode(mode, ctypes.byref(old_mode))


def child_environment(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONFAULTHANDLER"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    if args.force_cpu:
        threads = str(max(1, int(args.cpu_threads)))
        
        
        
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.update(
            {
                "AIM_FORCE_CPU": "1",
                "OMP_NUM_THREADS": threads,
                "MKL_NUM_THREADS": threads,
                "OPENBLAS_NUM_THREADS": threads,
                "NUMEXPR_NUM_THREADS": threads,
            }
        )
    return env


def gpu_state() -> tuple[int, int]:
    completed = subprocess.run(
        [
            "nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    first = completed.stdout.strip().splitlines()[0]
    memory, utilization = (int(part.strip()) for part in first.split(","))
    return memory, utilization


def wait_for_idle_gpu(args: argparse.Namespace) -> None:
    if args.skip_gpu_wait:
        return
    consecutive = 0
    while consecutive < args.idle_samples:
        memory, utilization = gpu_state()
        idle = memory <= args.idle_memory_mib and utilization <= args.idle_utilization
        consecutive = consecutive + 1 if idle else 0
        print(
            f"[GPU WAIT] memory={memory} MiB utilization={utilization}% "
            f"idle_samples={consecutive}/{args.idle_samples}",
            flush=True,
        )
        if consecutive < args.idle_samples:
            time.sleep(args.poll_seconds)


def write_state(state: dict, state_file: str) -> None:
    (HERE / state_file).write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    configure_windows_error_mode()
    env = child_environment(args)
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    if any(model in ROCKET_MODELS for model in args.models) and not args.rocket_python.is_file():
        raise FileNotFoundError(args.rocket_python)
    commands = []
    for model in args.models:
        relative = MODEL_SCRIPTS[model]
        interpreter = str(args.rocket_python) if model in ROCKET_MODELS else sys.executable
        commands.append((model, [interpreter, str(HERE / relative)]))
    if args.dry_run:
        for model, command in commands:
            print(model, subprocess.list2cmdline(command))
        return

    wait_for_idle_gpu(args)
    state = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.executable,
        "rocket_python": str(args.rocket_python),
        "models_requested": args.models,
        "execution_device": "cpu" if args.force_cpu else "auto",
        "cpu_threads": max(1, int(args.cpu_threads)) if args.force_cpu else None,
        "completed": [],
        "failed": [],
        "current": None,
        "status": "RUNNING",
    }
    write_state(state, args.state_file)
    for model, command in commands:
        state["current"] = model
        state["current_started_at"] = datetime.now().isoformat(timespec="seconds")
        write_state(state, args.state_file)
        print(f"\n[MODEL START] {model}", flush=True)
        model_started = time.perf_counter()
        result = subprocess.run(
            command,
            cwd=Path(command[-1]).parent,
            check=False,
            env=env,
            creationflags=creationflags,
        )
        elapsed = time.perf_counter() - model_started
        destination = "completed" if result.returncode == 0 else "failed"
        state[destination].append(
            {
                "model": model,
                "returncode": result.returncode,
                "elapsed_seconds": elapsed,
            }
        )
        write_state(state, args.state_file)

    state["current"] = None
    state["status"] = "COMPLETE" if not state["failed"] else "COMPLETE_WITH_FAILURES"
    state["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_state(state, args.state_file)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
