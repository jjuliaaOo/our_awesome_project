"""
CLI script for end-to-end training of the baseline k-mer logistic regression model,
including loading CSVs, training, evaluation, and saving results.

Example usage (Windows, from the project root):

    py scripts/train_baseline.py ^
        --train data/processed/dataset_train.csv ^
        --val data/processed/dataset_val.csv ^
        --test data/processed/dataset_test.csv ^
        --k 3 ^
        --normalize ^
        --output-model models/baseline_model.joblib ^
        --output-metrics results/baseline_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src/ is on sys.path so that `promoter_classifier` can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from promoter_classifier.models import KmerLogisticBaseline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model training."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate the baseline k-mer logistic regression model."
    )

    parser.add_argument(
        "--train",
        type=str,
        required=True,
        help="Path to the training CSV file with columns ['sequence', 'label'].",
    )
    parser.add_argument(
        "--val",
        type=str,
        required=True,
        help="Path to the validation CSV file with columns ['sequence', 'label'].",
    )
    parser.add_argument(
        "--test",
        type=str,
        required=True,
        help="Path to the test CSV file with columns ['sequence', 'label'].",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Size of k-mers to use as features (default: 3).",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize k-mer frequencies to frequencies instead of raw counts.",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="models/baseline_model.joblib",
        help="Path where the trained model will be saved (default: models/baseline_model.joblib).",
    )
    parser.add_argument(
        "--output-metrics",
        type=str,
        default="results/baseline_metrics.json",
        help="Path where evaluation metrics will be saved as JSON (default: results/baseline_metrics.json).",
    )

    return parser.parse_args()


def compute_metrics(model: KmerLogisticBaseline, df: pd.DataFrame) -> dict:
    """
    Compute accuracy and ROC-AUC for a given dataset.

    Args:
        model: Trained KmerLogisticBaseline model.
        df: DataFrame with 'sequence' and 'label' columns.

    Returns:
        Dictionary with 'accuracy' and 'roc_auc' keys.
    """
    sequences = df["sequence"].tolist()
    y_true = df["label"].values

    # Predictions
    y_pred = model.predict(sequences)
    accuracy = accuracy_score(y_true, y_pred)

    # ROC-AUC (handle case where only one class is present)
    try:
        y_proba = model.predict_proba(sequences)[:, 1]  # Probability of positive class
        roc_auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        # If only one class is present, ROC-AUC cannot be computed
        roc_auc = None

    return {
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
    }


def main() -> None:
    """Entry point for the training CLI."""
    args = parse_args()

    # Load datasets
    print("Loading datasets...")
    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.val)
    test_df = pd.read_csv(args.test)

    # Validate columns
    required_columns = ["sequence", "label"]
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"{name} dataset is missing columns: {missing_cols}")

    # Initialize and train model
    print(f"Training model with k={args.k}, normalize={args.normalize}...")
    model = KmerLogisticBaseline(k=args.k, normalize=args.normalize)
    model.fit(train_df["sequence"].tolist(), train_df["label"].tolist())

    # Evaluate model
    print("Evaluating model...")
    train_metrics = compute_metrics(model, train_df)
    val_metrics = compute_metrics(model, val_df)
    test_metrics = compute_metrics(model, test_df)

    # Print results
    print("\n=== Model Evaluation Results ===")
    print(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
    if train_metrics['roc_auc'] is not None:
        print(f"Train ROC-AUC:  {train_metrics['roc_auc']:.4f}")
    else:
        print("Train ROC-AUC:  N/A (only one class present)")
        
    print(f"Val Accuracy:   {val_metrics['accuracy']:.4f}")
    if val_metrics['roc_auc'] is not None:
        print(f"Val ROC-AUC:    {val_metrics['roc_auc']:.4f}")
    else:
        print("Val ROC-AUC:    N/A (only one class present)")
        
    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    if test_metrics['roc_auc'] is not None:
        print(f"Test ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    else:
        print("Test ROC-AUC:   N/A (only one class present)")

    # Save model
    output_model_path = Path(args.output_model)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_model_path)
    print(f"\nModel saved to: {output_model_path}")

    # Save metrics
    metrics = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }
    output_metrics_path = Path(args.output_metrics)
    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {output_metrics_path}")


if __name__ == "__main__":
    main()