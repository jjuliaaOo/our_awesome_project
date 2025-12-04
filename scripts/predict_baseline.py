# File: scripts/predict_baseline.py
"""
CLI script for predicting promoter sequences using a trained baseline k-mer logistic regression model.

Example usage (Windows, from the project root):

    py scripts/predict_baseline.py ^
        --model models/baseline_model.joblib ^
        --input data/processed/dataset_test.csv ^
        --output results/predictions.csv ^
        --threshold 0.5
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

import joblib
import pandas as pd
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model prediction."""
    parser = argparse.ArgumentParser(
        description="Predict promoter sequences using a trained baseline k-mer logistic regression model."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained model (.joblib file).",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input CSV file with 'sequence' column.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path where the output CSV with predictions will be saved.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold for converting probabilities to binary predictions (default: 0.5).",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point for the prediction CLI."""
    args = parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    model = joblib.load(args.model)

    # Load input data
    print(f"Loading input data from {args.input}...")
    input_df = pd.read_csv(args.input)

    if "sequence" not in input_df.columns:
        raise ValueError("Input CSV must contain a 'sequence' column.")

    sequences = input_df["sequence"].tolist()
    n_sequences = len(sequences)
    print(f"Processing {n_sequences} sequences...")

    # Predict
    probas = model.predict_proba(sequences)[:, 1]  # Probability of positive class
    predictions = (probas >= args.threshold).astype(int)

    # Create output dataframe
    output_df = pd.DataFrame({
        "sequence": sequences,
        "proba": probas,
        "prediction": predictions,
    })

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")

    # Print summary statistics
    n_positive = predictions.sum()
    n_negative = n_sequences - n_positive
    print(f"\n=== Prediction Summary ===")
    print(f"Total sequences: {n_sequences}")
    print(f"Predicted positive (promoter): {n_positive}")
    print(f"Predicted negative (non-promoter): {n_negative}")


if __name__ == "__main__":
    main()