# File: scripts/prepare_dataset.py
"""
CLI script for preparing a labeled train/val/test dataset
from promoter and negative sequence files.

Example usage (Windows, from the project root):

    py scripts/prepare_dataset.py ^
        --promoters data/raw/promoters.txt ^
        --negatives data/raw/negatives.txt ^
        --output-dir data/processed ^
        --loader-type plain ^
        --train-size 0.8 ^
        --val-size 0.1 ^
        --test-size 0.1 ^
        --random-state 42 ^
        --prefix epd_demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is on sys.path so that `promoter_classifier` can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from promoter_classifier.data import (
    DatasetSplitConfig,
    prepare_sequences_to_csv,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare train/val/test CSV files from promoter and negative sequence files."
    )

    parser.add_argument(
        "--promoters",
        type=str,
        required=True,
        help="Path to the promoter sequences file (plain text or FASTA).",
    )
    parser.add_argument(
        "--negatives",
        type=str,
        required=True,
        help="Path to the negative (non-promoter) sequences file (plain text or FASTA).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory where train/val/test CSV files will be saved (default: data/processed).",
    )
    parser.add_argument(
        "--loader-type",
        type=str,
        choices=["plain", "fasta"],
        default="plain",
        help="Type of input files: 'plain' (one sequence per line) or 'fasta' (default: plain).",
    )

    parser.add_argument(
        "--train-size",
        type=float,
        default=0.8,
        help="Fraction of data to use for training (default: 0.8).",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation (default: 0.1).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Fraction of data to use for testing (default: 0.1).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for splitting the dataset (default: 42).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="dataset",
        help="Prefix for the output CSV files (default: 'dataset').",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point for the dataset preparation CLI."""
    args = parse_args()

    promoters_path = Path(args.promoters)
    negatives_path = Path(args.negatives)
    output_dir = Path(args.output_dir)

    # Build split configuration
    split_config = DatasetSplitConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # Run the high-level pipeline
    train_df, val_df, test_df = prepare_sequences_to_csv(
        promoters_path=promoters_path,
        negatives_path=negatives_path,
        output_dir=output_dir,
        loader_type=args.loader_type,
        split_config=split_config,
        output_prefix=args.prefix,
    )

    total = len(train_df) + len(val_df) + len(test_df)

    print("=== Dataset preparation completed ===")
    print(f"Promoters file: {promoters_path}")
    print(f"Negatives file: {negatives_path}")
    print(f"Output directory: {output_dir}")
    print()
    print(f"Total sequences: {total}")
    print(f"  Train: {len(train_df)}")
    print(f"  Val:   {len(val_df)}")
    print(f"  Test:  {len(test_df)}")
    print()
    print("Output CSV files:")
    print(f"  {output_dir / (args.prefix + '_train.csv')}")
    print(f"  {output_dir / (args.prefix + '_val.csv')}")
    print(f"  {output_dir / (args.prefix + '_test.csv')}")


if __name__ == "__main__":
    main()