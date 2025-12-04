"""Tests for the train_baseline.py script."""

import subprocess
import sys
from pathlib import Path


def test_train_baseline_runs(tmp_path):
    """Test that the train_baseline script runs end-to-end without errors."""
    # Create small synthetic CSVs
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    test_csv = tmp_path / "test.csv"

    import pandas as pd
    df_train = pd.DataFrame({
        "sequence": ["AAAA", "CCCC", "AAAA", "CCCC"],
        "label": [1, 0, 1, 0],
    })
    df_val = pd.DataFrame({
        "sequence": ["AAAA", "CCCC"],
        "label": [1, 0],
    })
    df_test = pd.DataFrame({
        "sequence": ["AAAA", "CCCC"],
        "label": [1, 0],
    })

    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)
    df_test.to_csv(test_csv, index=False)

    # Import training script
    cmd = [
        sys.executable,
        "scripts/train_baseline.py",
        "--train", str(train_csv),
        "--val", str(val_csv),
        "--test", str(test_csv),
        "--k", "1",
        "--normalize",
        "--output-model", str(tmp_path / "model.joblib"),
        "--output-metrics", str(tmp_path / "metrics.json"),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    assert result.returncode == 0, f"Script failed with stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    # Check output files
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "metrics.json").exists()