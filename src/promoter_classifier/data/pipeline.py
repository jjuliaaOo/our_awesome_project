"""Pipeline utilities for preparing promoter classification datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

from .loader import load_fasta, load_plain_sequences
from .datasets import (
    DatasetSplitConfig,
    build_labeled_dataframe,
    split_dataframe,
    save_splits_to_csv,
)


def prepare_sequences_to_csv(
    promoters_path: Union[str, Path],
    negatives_path: Union[str, Path],
    output_dir: Union[str, Path],
    loader_type: str = "plain",
    split_config: Optional[DatasetSplitConfig] = None,
    output_prefix: str = "dataset",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    High-level helper to build a labeled dataset from promoter and negative
    sequence files, split it into train/val/test, and save the splits as CSV.

    Depending on `loader_type`, uses either plain-text or FASTA loaders.

    Args:
        promoters_path: Path to a file containing promoter sequences.
        negatives_path: Path to a file containing negative (non-promoter) sequences.
        output_dir: Directory where train/val/test CSV files will be stored.
        loader_type: Either "plain" or "fasta" to choose the appropriate loader.
        split_config: Optional DatasetSplitConfig; if None, a default config is used.
        output_prefix: Prefix for the resulting CSV files (e.g. "epd_dataset").

    Returns:
        Tuple of (train_df, val_df, test_df) as pandas DataFrames.

    Raises:
        ValueError: If an unsupported loader_type is provided.
    """
    # Convert paths to Path objects
    promoters_path = Path(promoters_path)
    negatives_path = Path(negatives_path)
    output_dir = Path(output_dir)
    
    # Choose the loader
    if loader_type == "plain":
        # use load_plain_sequences with clean=True
        promoters = load_plain_sequences(promoters_path, clean=True)
        negatives = load_plain_sequences(negatives_path, clean=True)
    elif loader_type == "fasta":
        promoters = load_fasta(promoters_path)
        negatives = load_fasta(negatives_path)
    else:
        raise ValueError(f"Unsupported loader_type: {loader_type!r}. Expected 'plain' or 'fasta'.")

    # Build a labeled DataFrame
    df = build_labeled_dataframe(promoters, negatives)
    
    # Use default config if none provided
    if split_config is None:
        split_config = DatasetSplitConfig()
        
    # Split the dataframe
    train_df, val_df, test_df = split_dataframe(df, split_config, stratify=True)
    
    # Save the splits to CSV
    save_splits_to_csv(train_df, val_df, test_df, base_dir=output_dir, prefix=output_prefix)
    
    return train_df, val_df, test_df