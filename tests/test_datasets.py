"""Tests for the promoter_classifier.data.datasets module."""

import pandas as pd
import pytest
from pathlib import Path

from promoter_classifier.data import (
    DatasetSplitConfig,
    build_labeled_dataframe,
    split_dataframe,
)


def test_build_labeled_dataframe_basic():
    """Test build_labeled_dataframe with basic inputs."""
    promoters = ["AAA", "CCC"]
    negatives = ["GGG", "TTT", "ACGT"]
    
    df = build_labeled_dataframe(promoters, negatives)
    
    # Check basic properties
    assert len(df) == len(promoters) + len(negatives)
    assert set(df["label"].unique()) == {0, 1}
    assert len(df[df["label"] == 1]) == len(promoters)
    assert len(df[df["label"] == 0]) == len(negatives)
    assert set(df["class_name"].unique()) == {"promoter", "negative"}
    assert len(df[df["class_name"] == "promoter"]) == len(promoters)
    assert len(df[df["class_name"] == "negative"]) == len(negatives)


def test_split_dataframe_sizes_and_disjoint():
    """Test split_dataframe produces correct sizes and disjoint splits."""
    # Create a larger dataset for more reliable splitting
    promoters = ["A" * 20] * 12  # 12 promoter sequences
    negatives = ["T" * 20] * 8   # 8 negative sequences
    df = build_labeled_dataframe(promoters, negatives)
    
    config = DatasetSplitConfig(train_size=0.6, val_size=0.2, test_size=0.2, random_state=123)
    
    train_df, val_df, test_df = split_dataframe(df, config, stratify=True)
    
    # Check that all rows are used exactly once
    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    
    # Check that sets of indices are disjoint
    train_indices = set(train_df.index)
    val_indices = set(val_df.index)
    test_indices = set(test_df.index)
    
    assert len(train_indices & val_indices) == 0
    assert len(train_indices & test_indices) == 0
    assert len(val_indices & test_indices) == 0
    
    # Check that all splits have positive size
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0


def test_split_dataframe_invalid_config():
    """Test that invalid configurations raise ValueError."""
    # Create a simple dataset
    promoters = ["AAA", "CCC"]
    negatives = ["GGG", "TTT"]
    df = build_labeled_dataframe(promoters, negatives)
    
    # Test config where fractions don't sum to 1.0
    invalid_config = DatasetSplitConfig(train_size=0.5, val_size=0.5, test_size=0.5)
    
    with pytest.raises(ValueError, match="Split sizes must sum to 1.0"):
        split_dataframe(df, invalid_config, stratify=True)
    
    # Test config with negative sizes
    negative_config = DatasetSplitConfig(train_size=-0.5, val_size=0.5, test_size=1.0)
    
    with pytest.raises(ValueError, match="All split sizes must be positive"):
        split_dataframe(df, negative_config, stratify=True)


def test_dataset_split_config_validate():
    """Test DatasetSplitConfig.validate method directly."""
    # Valid config
    valid_config = DatasetSplitConfig(train_size=0.8, val_size=0.1, test_size=0.1)
    valid_config.validate()  # Should not raise
    
    # Invalid configs
    with pytest.raises(ValueError, match="All split sizes must be positive"):
        invalid_config = DatasetSplitConfig(train_size=0, val_size=0.5, test_size=0.5)
        invalid_config.validate()
    
    with pytest.raises(ValueError, match="Split sizes must sum to 1.0"):
        invalid_config = DatasetSplitConfig(train_size=0.5, val_size=0.5, test_size=0.5)
        invalid_config.validate()


def test_save_splits_to_csv(tmp_path):
    """Test save_splits_to_csv function."""
    from promoter_classifier.data.datasets import save_splits_to_csv
    
    # Create simple DataFrames
    train_df = pd.DataFrame({"sequence": ["AAAA"], "label": [1]})
    val_df = pd.DataFrame({"sequence": ["CCCC"], "label": [1]})
    test_df = pd.DataFrame({"sequence": ["GGGG"], "label": [0]})
    
    # Save splits
    save_splits_to_csv(train_df, val_df, test_df, tmp_path, "test_dataset")
    
    # Check that files were created
    assert (tmp_path / "test_dataset_train.csv").exists()
    assert (tmp_path / "test_dataset_val.csv").exists()
    assert (tmp_path / "test_dataset_test.csv").exists()
    
    # Check content
    loaded_train = pd.read_csv(tmp_path / "test_dataset_train.csv")
    assert len(loaded_train) == 1
    assert loaded_train.iloc[0]["sequence"] == "AAAA"
    assert loaded_train.iloc[0]["label"] == 1