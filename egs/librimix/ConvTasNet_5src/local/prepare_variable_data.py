"""Merge Libri2Mix and Libri3Mix metadata CSVs into unified 5-source CSVs.

Creates merged CSVs with columns:
    mixture_ID, mixture_path, source_1_path, ..., source_5_path, length

Missing sources (e.g. source_3..5 for 2-mix rows) are NaN,
which VariableLibriMix handles by filling with silence.
"""
import os
import argparse
import pandas as pd


LIBRIMIX_ROOT = "/home/mkeller/data/librimix"
SUBSETS = ["train-360", "dev", "test"]
TASKS = {
    "sep_clean": "mix_clean",
    "sep_noisy": "mix_both",
}
MAX_SOURCES = 5


def merge_subset(subset, task_suffix, out_dir):
    """Merge Libri2Mix and Libri3Mix CSVs for one subset."""
    dfs = []
    for n_src in [2, 3]:
        csv_path = os.path.join(
            LIBRIMIX_ROOT,
            f"Libri{n_src}Mix",
            "wav8k",
            "min",
            "metadata",
            f"mixture_{subset}_{task_suffix}.csv",
        )
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping")
            continue
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} rows from {csv_path}")
        dfs.append(df)

    if not dfs:
        print(f"No data found for subset {subset}, skipping")
        return

    merged = pd.concat(dfs, ignore_index=True)

    # Ensure all source columns exist up to MAX_SOURCES
    for i in range(1, MAX_SOURCES + 1):
        col = f"source_{i}_path"
        if col not in merged.columns:
            merged[col] = float("nan")

    # Reorder columns
    cols = ["mixture_ID", "mixture_path"]
    cols += [f"source_{i}_path" for i in range(1, MAX_SOURCES + 1)]
    cols += ["length"]
    merged = merged[cols]

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"mixture_{task_suffix}.csv")
    merged.to_csv(out_path, index=False)
    print(f"  Wrote {len(merged)} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=str, default="sep_clean",
        choices=list(TASKS.keys()),
        help="Task type",
    )
    parser.add_argument(
        "--out_root", type=str, default="data/wav8k/min",
        help="Output root directory for merged CSVs",
    )
    args = parser.parse_args()

    task_suffix = TASKS[args.task]

    for subset in SUBSETS:
        print(f"Processing subset: {subset}")
        out_dir = os.path.join(args.out_root, subset)
        merge_subset(subset, task_suffix, out_dir)


if __name__ == "__main__":
    main()
