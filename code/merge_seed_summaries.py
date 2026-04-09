#!/usr/bin/env python3
"""Merge per-seed summary.csv files into a single summary_master.csv file.

Expected directory layout:
    outputs_root/
        seed_1/summary.csv
        seed_2/summary.csv
        ...

Each summary.csv is expected to contain one row per scenario.
The script adds a seed column inferred from the parent directory name when
the file does not already contain one.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import pandas as pd


def infer_seed(path: Path) -> int:
    match = re.search(r"(\d+)", path.parent.name)
    if not match:
        raise ValueError(f"Could not infer seed from directory name: {path.parent.name}")
    return int(match.group(1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge per-seed summary.csv files into summary_master.csv."
    )
    parser.add_argument(
        "--inputs-root",
        required=True,
        help="Root directory containing seed_* subdirectories."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the merged CSV file."
    )
    parser.add_argument(
        "--pattern",
        default="seed_*/summary.csv",
        help="Glob pattern relative to inputs root (default: seed_*/summary.csv)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs_root = Path(args.inputs_root)
    files = sorted(inputs_root.glob(args.pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matched pattern '{args.pattern}' under {inputs_root}"
        )

    frames = []
    for file_path in files:
        df = pd.read_csv(file_path)
        if "seed" not in df.columns:
            df.insert(0, "seed", infer_seed(file_path))
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["seed", "scenario"]).reset_index(drop=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Saved merged summary to: {output_path}")


if __name__ == "__main__":
    main()
