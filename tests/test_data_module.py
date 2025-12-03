"""Tests for the promoter_classifier.data module."""

import pandas as pd
import pytest
from pathlib import Path

from promoter_classifier.data import (
    RawDatasetPaths,
    EPDDatasetPaths,
    load_fasta,
    load_promoter_table,
    load_plain_sequences,
    clean_sequence,
    is_valid_sequence,
)


def test_clean_sequence_basic():
    """Test clean_sequence function with various inputs."""
    # Test basic sequence
    assert clean_sequence("ACGT") == "ACGT"
    
    # Test lowercase and invalid characters
    assert clean_sequence("acgtnxyz") == "ACGT"
    
    # Test mixed case and invalid characters
    assert clean_sequence("AaCcGgTtNnXxYyZz") == "AACCGGTT"
    
    # Test empty string
    assert clean_sequence("") == ""


def test_is_valid_sequence():
    """Test is_valid_sequence function."""
    # Test valid sequences
    assert is_valid_sequence("ACGT") is True
    assert is_valid_sequence("AAAA") is True
    assert is_valid_sequence("CCCC") is True
    assert is_valid_sequence("GGGG") is True
    assert is_valid_sequence("TTTT") is True
    assert is_valid_sequence("") is True  # Empty string is valid
    
    # Test invalid sequences
    assert is_valid_sequence("ACGTN") is False
    assert is_valid_sequence("acgtn") is False  # Lowercase is converted to uppercase
    assert is_valid_sequence("ACGTX") is False
    assert is_valid_sequence("ACGT1") is False


def test_load_fasta_simple(tmp_path):
    """Test load_fasta function with a simple FASTA file."""
    # Create a temporary FASTA file
    fasta_content = """>seq1
ACGT
>seq2
ACGTNNacgt"""
    
    fasta_file = tmp_path / "test.fasta"
    fasta_file.write_text(fasta_content)
    
    # Load sequences
    sequences = load_fasta(fasta_file)
    
    # Check results
    assert isinstance(sequences, list)
    assert len(sequences) == 2
    assert sequences[0] == "ACGT"
    assert sequences[1] == "ACGTNNACGT"  # Only stripped and uppercased, not cleaned


def test_load_promoter_table_csv(tmp_path):
    """Test load_promoter_table function with a valid CSV file."""
    # Create a temporary CSV file
    csv_content = """sequence,label
ATCGATCG,1
GGCCGGCC,0
TATATATA,1"""
    
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(csv_content)
    
    # Load the table
    df = load_promoter_table(csv_file)
    
    # Check results
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 2)
    assert list(df.columns) == ["sequence", "label"]
    assert list(df["sequence"]) == ["ATCGATCG", "GGCCGGCC", "TATATATA"]
    assert list(df["label"]) == [1, 0, 1]


def test_load_promoter_table_nonexistent_file():
    """Test that load_promoter_table raises FileNotFoundError for nonexistent files."""
    with pytest.raises(FileNotFoundError):
        load_promoter_table("nonexistent_file.csv")


def test_load_promoter_table_wrong_extension(tmp_path):
    """Test that load_promoter_table raises ValueError for non-CSV files."""
    # Create a temporary TXT file
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("some content")
    
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_promoter_table(txt_file)


def test_raw_dataset_paths_creation(tmp_path):
    """Test RawDatasetPaths creation and directory setup."""
    raw_dir = tmp_path / "raw"
    interim_dir = tmp_path / "interim"
    processed_dir = tmp_path / "processed"
    
    paths = RawDatasetPaths(raw_dir, interim_dir, processed_dir)
    
    # Check that paths are correctly set
    assert paths.raw_dir == raw_dir
    assert paths.interim_dir == interim_dir
    assert paths.processed_dir == processed_dir
    
    # Check that directories are created
    assert raw_dir.exists()
    assert interim_dir.exists()
    assert processed_dir.exists()


def test_epd_dataset_paths_creation(tmp_path):
    """Test EPDDatasetPaths creation."""
    promoters_path = tmp_path / "promoters.csv"
    negatives_path = tmp_path / "negatives.csv"
    metadata_path = tmp_path / "metadata.csv"
    
    paths = EPDDatasetPaths(promoters_path, negatives_path, metadata_path)
    
    # Check that paths are correctly set
    assert paths.promoters_path == promoters_path
    assert paths.negatives_path == negatives_path
    assert paths.metadata_path == metadata_path
    
    # Check that parent directories are created
    assert promoters_path.parent.exists()
    assert negatives_path.parent.exists()
    assert metadata_path.parent.exists()


def test_epd_dataset_paths_without_metadata(tmp_path):
    """Test EPDDatasetPaths creation without metadata."""
    promoters_path = tmp_path / "promoters.csv"
    negatives_path = tmp_path / "negatives.csv"
    
    paths = EPDDatasetPaths(promoters_path, negatives_path)
    
    # Check that paths are correctly set
    assert paths.promoters_path == promoters_path
    assert paths.negatives_path == negatives_path
    assert paths.metadata_path is None


def test_load_plain_sequences_basic(tmp_path):
    """Test load_plain_sequences function with a simple text file."""
    # Create a temporary text file
    text_content = "acgtn\n\nACGT\nxxxacgtYYY"
    
    text_file = tmp_path / "test.txt"
    text_file.write_text(text_content)
    
    # Load sequences with cleaning
    sequences = load_plain_sequences(text_file, clean=True)
    
    # Check results
    assert isinstance(sequences, list)
    assert len(sequences) == 3  # Empty line should be ignored
    assert sequences[0] == "ACGT"  # "acgtn" -> "ACGT" (cleaned)
    assert sequences[1] == "ACGT"  # "ACGT" -> "ACGT" (no change)
    assert sequences[2] == "ACGT"  # "xxxacgtYYY" -> "ACGT" (cleaned)
    
    # Load sequences without cleaning
    sequences_uncleaned = load_plain_sequences(text_file, clean=False)
    
    # Check results
    assert isinstance(sequences_uncleaned, list)
    assert len(sequences_uncleaned) == 3  # Empty line should be ignored
    assert sequences_uncleaned[0] == "ACGTN"  # "acgtn" -> "ACGTN" (only uppercased)
    assert sequences_uncleaned[1] == "ACGT"   # "ACGT" -> "ACGT" (no change)
    assert sequences_uncleaned[2] == "XXXACGTYYY"  # "xxxacgtYYY" -> "XXXACGTYYY" (only uppercased)


def test_load_plain_sequences_file_not_found():
    """Test that load_plain_sequences raises FileNotFoundError for nonexistent files."""
    with pytest.raises(FileNotFoundError):
        load_plain_sequences("nonexistent_file.txt")