"""Tests for the promoter_classifier.data.pipeline module."""

from pathlib import Path

import pandas as pd
import pytest

from promoter_classifier.data import (
    DatasetSplitConfig,
    prepare_sequences_to_csv,
)


def test_prepare_sequences_to_csv_plain(tmp_path):
    """Test prepare_sequences_to_csv with plain text files."""
    # Create temporary plain-text files
    promoters_file = tmp_path / "promoters.txt"
    negatives_file = tmp_path / "negatives.txt"
    
    # Write promoter sequences
    promoters_content = "AAA\nCCC\nGGG"
    promoters_file.write_text(promoters_content)
    
    # Write negative sequences
    negatives_content = "TTT\nACGT\nACAC\nGTGT"
    negatives_file.write_text(negatives_content)
    
    # Define output directory
    output_dir = tmp_path / "processed"
    
    # Define a split config
    config = DatasetSplitConfig(train_size=0.5, val_size=0.25, test_size=0.25, random_state=123)
    
    # Call the function
    train_df, val_df, test_df = prepare_sequences_to_csv(
        promoters_path=promoters_file,
        negatives_path=negatives_file,
        output_dir=output_dir,
        loader_type="plain",
        split_config=config,
        output_prefix="test_dataset",
    )
    
    # Assertions
    # Check that returned objects are pandas DataFrames
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    
    # Check that the total number of rows equals the number of sequences in both files combined
    total_input_sequences = 3 + 4  # 3 promoters + 4 negatives
    assert len(train_df) + len(val_df) + len(test_df) == total_input_sequences
    
    # Check that the indices of the three splits are disjoint
    train_indices = set(train_df.index)
    val_indices = set(val_df.index)
    test_indices = set(test_df.index)
    
    assert len(train_indices & val_indices) == 0
    assert len(train_indices & test_indices) == 0
    assert len(val_indices & test_indices) == 0
    
    # Check that all labels are in {0, 1}
    all_labels = set(pd.concat([train_df, val_df, test_df])["label"].unique())
    assert all_labels == {0, 1}
    
    # Check that the output directory contains the expected files
    assert (output_dir / "test_dataset_train.csv").exists()
    assert (output_dir / "test_dataset_val.csv").exists()
    assert (output_dir / "test_dataset_test.csv").exists()


def test_prepare_sequences_to_csv_fasta(tmp_path):
    """Test prepare_sequences_to_csv with FASTA files."""
    # Create temporary FASTA files
    promoters_file = tmp_path / "promoters.fasta"
    negatives_file = tmp_path / "negatives.fasta"
    
    # Write promoter sequences in FASTA format
    promoters_content = ">seq1\nAAA\n>seq2\nCCC"
    promoters_file.write_text(promoters_content)
    
    # Write negative sequences in FASTA format
    negatives_content = ">seq1\nTTT\n>seq2\nACGT\n>seq3\nACAC"
    negatives_file.write_text(negatives_content)
    
    # Define output directory
    output_dir = tmp_path / "processed"
    
    # Call the function
    train_df, val_df, test_df = prepare_sequences_to_csv(
        promoters_path=promoters_file,
        negatives_path=negatives_file,
        output_dir=output_dir,
        loader_type="fasta",
        output_prefix="fasta_test",
    )
    
    # Assertions
    # Check that returned objects are pandas DataFrames
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    
    # Check that the total number of rows equals the number of sequences in both files combined
    total_input_sequences = 2 + 3  # 2 promoters + 3 negatives
    assert len(train_df) + len(val_df) + len(test_df) == total_input_sequences
    
    # Check that all labels are in {0, 1}
    all_labels = set(pd.concat([train_df, val_df, test_df])["label"].unique())
    assert all_labels == {0, 1}
    
    # Check that the output directory contains the expected files
    assert (output_dir / "fasta_test_train.csv").exists()
    assert (output_dir / "fasta_test_val.csv").exists()
    assert (output_dir / "fasta_test_test.csv").exists()


def test_prepare_sequences_to_csv_invalid_loader_type():
    """Test that invalid loader_type raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported loader_type"):
        prepare_sequences_to_csv(
            promoters_path="dummy",
            negatives_path="dummy",
            output_dir="dummy",
            loader_type="invalid_type"
        )


def test_prepare_sequences_to_csv_default_config(tmp_path):
    """Test prepare_sequences_to_csv with default config."""
    # Create temporary plain-text files
    promoters_file = tmp_path / "promoters.txt"
    negatives_file = tmp_path / "negatives.txt"
    
    # Write sequences
    promoters_file.write_text("AAA\nCCC")
    negatives_file.write_text("TTT\nACGT")
    
    # Define output directory
    output_dir = tmp_path / "processed"
    
    # Call the function without providing a split_config
    train_df, val_df, test_df = prepare_sequences_to_csv(
        promoters_path=promoters_file,
        negatives_path=negatives_file,
        output_dir=output_dir,
        loader_type="plain",
        output_prefix="default_config_test",
    )
    
    # Assertions
    # Check that returned objects are pandas DataFrames
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    
    # Check that the total number of rows equals the number of sequences in both files combined
    total_input_sequences = 2 + 2  # 2 promoters + 2 negatives
    assert len(train_df) + len(val_df) + len(test_df) == total_input_sequences
    
    # Check that the output directory contains the expected files
    assert (output_dir / "default_config_test_train.csv").exists()
    assert (output_dir / "default_config_test_val.csv").exists()
    assert (output_dir / "default_config_test_test.csv").exists()