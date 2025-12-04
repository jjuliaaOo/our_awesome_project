# File: tests/test_predict_baseline.py
"""Tests for the predict_baseline.py script."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import joblib

from promoter_classifier.models import KmerLogisticBaseline


def test_predict_baseline_runs(tmp_path):
    """Test that the predict_baseline script runs end-to-end without errors."""
    # Create a small synthetic dataset and train a model
    sequences = ["AAAA", "CCCC", "AAAA", "CCCC"]
    labels = [1, 0, 1, 0]
    
    # Train a small model
    model = KmerLogisticBaseline(k=1, normalize=True)
    model.fit(sequences, labels)
    
    # Save model to temporary directory
    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)
    
    # Create input CSV
    input_csv = tmp_path / "input.csv"
    df_input = pd.DataFrame({
        "sequence": ["AAAA", "CCCC", "GGGG"],
    })
    df_input.to_csv(input_csv, index=False)
    
    # Define output path
    output_csv = tmp_path / "output.csv"
    
    # Run prediction script
    cmd = [
        sys.executable,
        "scripts/predict_baseline.py",
        "--model", str(model_path),
        "--input", str(input_csv),
        "--output", str(output_csv),
        "--threshold", "0.5",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    assert result.returncode == 0, f"Script failed with stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    
    # Check output file exists
    assert output_csv.exists()
    
    # Check output CSV has correct columns and number of rows
    df_output = pd.read_csv(output_csv)
    expected_columns = {"sequence", "proba", "prediction"}
    assert set(df_output.columns) == expected_columns
    assert len(df_output) == len(df_input)