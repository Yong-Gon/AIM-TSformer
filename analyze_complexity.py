from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable, Optional


def parse_optional_int(value: str) -> Optional[int]:
    text = value.strip().lower()
    if text in {"none", "null", "unlimited"}:
        return None
    parsed = int(text)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("max-inner must be positive or None")
    return parsed


def gib(byte_count: int | float) -> float:
    return float(byte_count) / (1024.0 ** 3)


def complexity_row(
    *,
    sequence_length: int,
    channels: int,
    batch_size: int,
    grid: int,
    max_inner: Optional[int],
    embedding_dim: int,
    kernel_size: int,
    image_depth: int,
    dtype_bytes: int,
) -> dict:
    if sequence_length <= 0 or channels <= 0 or batch_size <= 0 or grid <= 0:
        raise ValueError("T, M, B, and S must all be positive")
    native_inner = sequence_length // grid
    if native_inner <= 0:
        raise ValueError(f"T={sequence_length} must be at least S={grid}")
    q = native_inner if max_inner is None else min(native_inner, max_inner)
    patch_pixels = grid * grid * q * q

    aim_elements_per_sample = channels * sequence_length * sequence_length
    aim_elements_batch = batch_size * aim_elements_per_sample
    aim_bytes_per_sample = dtype_bytes * aim_elements_per_sample
    aim_bytes_batch = dtype_bytes * aim_elements_batch

    image_projection_proxy = batch_size * patch_pixels * channels * embedding_dim
    image_encoder_proxy = (
        batch_size
        * patch_pixels
        * image_depth
        * (embedding_dim * kernel_size * kernel_size + embedding_dim * embedding_dim)
    )
    image_total_proxy = image_projection_proxy + image_encoder_proxy
    activation_elements = batch_size * patch_pixels * embedding_dim
    activation_bytes = dtype_bytes * activation_elements

    return {
        "sequence_length_T": sequence_length,
        "channels_M": channels,
        "batch_size_B": batch_size,
        "grid_S": grid,
        "max_inner": max_inner,
        "effective_q": q,
        "embedding_d": embedding_dim,
        "kernel_k": kernel_size,
        "image_depth_L": image_depth,
        "dtype_bytes": dtype_bytes,
        "aim_time_proxy_BMT2": aim_elements_batch,
        "aim_elements_per_sample_MT2": aim_elements_per_sample,
        "aim_raw_bytes_per_sample": aim_bytes_per_sample,
        "aim_raw_gib_per_sample": gib(aim_bytes_per_sample),
        "aim_raw_bytes_batch": aim_bytes_batch,
        "aim_raw_gib_batch": gib(aim_bytes_batch),
        "image_patch_pixels_S2q2": patch_pixels,
        "image_projection_proxy": image_projection_proxy,
        "image_encoder_proxy": image_encoder_proxy,
        "image_total_time_proxy": image_total_proxy,
        "image_activation_bytes_proxy": activation_bytes,
        "image_activation_gib_proxy": gib(activation_bytes),
    }


def make_rows(args: argparse.Namespace) -> list[dict]:
    if len(args.channels) == 1:
        pairs: Iterable[tuple[int, int]] = (
            (length, args.channels[0]) for length in args.lengths
        )
    elif len(args.channels) == len(args.lengths):
        pairs = zip(args.lengths, args.channels)
    else:
        raise ValueError(
            "Provide either one --channels value shared by all lengths or one "
            "channel count for every --lengths value."
        )
    return [
        complexity_row(
            sequence_length=length,
            channels=channels,
            batch_size=args.batch_size,
            grid=args.grid,
            max_inner=args.max_inner,
            embedding_dim=args.embedding_dim,
            kernel_size=args.kernel_size,
            image_depth=args.image_depth,
            dtype_bytes=args.dtype_bytes,
        )
        for length, channels in pairs
    ]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", nargs="+", type=int, required=True)
    parser.add_argument(
        "--channels",
        nargs="+",
        type=int,
        required=True,
        help="One shared M, or one M for every T in --lengths.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument("--max-inner", type=parse_optional_int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--kernel-size", type=int, default=9)
    parser.add_argument("--image-depth", type=int, default=2)
    parser.add_argument("--dtype-bytes", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=here / "complexity_results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = make_rows(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "aim_complexity_estimates.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "status": "analytical_estimate_not_measured_runtime",
        "notation": {
            "B": "batch size",
            "M": "number of time-series channels",
            "T": "sequence length",
            "S": "patch-grid count per image axis",
            "q": "min(floor(T/S), max_inner), or floor(T/S) when max_inner=None",
            "d": "image embedding dimension",
            "k": "depthwise-convolution kernel size",
            "L": "image encoder depth",
        },
        "formulae": {
            "aim_generation_time": "O(B*M*T^2)",
            "aim_output_memory": "O(B*M*T^2)",
            "float32_aim_bytes_per_sample": "4*M*T^2",
            "image_branch_time": "O(B*S^2*q^2*[M*d + L*(d*k^2 + d^2)])",
            "image_activation_memory": "O(B*S^2*q^2*d)",
        },
        "interpretation": (
            "When max_inner=None, S^2*q^2 is approximately T^2. Limiting "
            "max_inner bounds the internal image-branch resolution, but it does "
            "not remove the O(M*T^2) cost of materializing the AIM input."
        ),
        "rows": rows,
    }
    json_path = args.output_dir / "aim_complexity_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    for row in rows:
        print(
            f"T={row['sequence_length_T']} M={row['channels_M']} "
            f"q={row['effective_q']} AIM/sample={row['aim_raw_gib_per_sample']:.6f} GiB "
            f"image-activation/batch~={row['image_activation_gib_proxy']:.6f} GiB"
        )


if __name__ == "__main__":
    main()
