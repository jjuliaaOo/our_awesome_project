"""Dataset utilities for building labeled DataFrames and splitting datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DatasetSplitConfig:
    """
    Configuration for splitting a labeled dataset into train/validation/test parts.
    Sizes are relative fractions that must sum to 1.0.
    """
    train_size: float = 0.8
    val_size: float = 0.1
    test_size: float = 0.1
    random_state: int = 42

    def validate(self) -> None:
        """
        Validate that the split fractions are positive and sum to 1.0 (within a small tolerance).
        Raise ValueError if the configuration is invalid.
        """
        if self.train_size <= 0 or self.val_size <= 0 or self.test_size <= 0:
            raise ValueError("All split sizes must be positive")
        
        total = self.train_size + self.val_size + self.test_size
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split sizes must sum to 1.0, got {total}")


def build_labeled_dataframe(promoters: List[str], negatives: List[str]) -> pd.DataFrame:
    """
    Build a pandas DataFrame with labeled promoter and negative sequences.
    
    Args:
        promoters: List of promoter sequences
        negatives: List of negative sequences
        
    Returns:
        DataFrame with columns: "sequence", "label", "class_name"
    """
    # Create DataFrame for promoters
    promoter_df = pd.DataFrame({
        "sequence": promoters,
        "label": 1,
        "class_name": "promoter"
    })
    
    # Create DataFrame for negatives
    negative_df = pd.DataFrame({
        "sequence": negatives,
        "label": 0,
        "class_name": "negative"
    })
    
    # Concatenate and reset index
    df = pd.concat([promoter_df, negative_df], ignore_index=True)
    return df


def split_dataframe(
    df: pd.DataFrame,
    config: DatasetSplitConfig,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train/validation/test sets.
    
    Args:
        df: Input DataFrame to split
        config: Configuration for splitting
        stratify: Whether to perform stratified splits based on labels
        
    Returns:
        Tuple of (train_df, val_df, test_df)
        
    Raises:
        ValueError: If the configuration is invalid
    """
    # Validate the configuration
    config.validate()
    
    # Determine stratification parameter
    stratify_param = df["label"] if stratify else None
    
    # First split: train and temp
    try:
        train_df, temp_df = train_test_split(
            df, 
            train_size=config.train_size, 
            stratify=stratify_param, 
            random_state=config.random_state
        )
    except ValueError:
        # Fallback: no stratification for small or imbalanced datasets
        train_df, temp_df = train_test_split(
            df, 
            train_size=config.train_size, 
            stratify=None, 
            random_state=config.random_state
        )
    
    # Calculate relative fraction for validation in temp set
    temp_total = config.val_size + config.test_size
    if temp_total > 0:
        val_fraction = config.val_size / temp_total
    else:
        val_fraction = 0.0
    
    # Second split: val and test
    if len(temp_df) > 1 and 0 < val_fraction < 1.0:
        stratify_param_temp = temp_df["label"] if stratify else None

        try:
            val_df, test_df = train_test_split(
                temp_df,
                train_size=val_fraction,
                stratify=stratify_param_temp,
                random_state=config.random_state,
            )
        except ValueError:
            # Fallback: no stratification for very small or imbalanced datasets
            val_df, test_df = train_test_split(
                temp_df,
                train_size=val_fraction,
                stratify=None,
                random_state=config.random_state,
            )
    else:
        # For tiny temp sets, keep everything as test and leave val empty
        val_df = temp_df.iloc[:0].copy()
        test_df = temp_df.copy()
    
    return train_df, val_df, test_df


def save_splits_to_csv(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_dir: Union[str, Path],
    prefix: str = "dataset"
) -> None:
    """
    Save train/validation/test DataFrames to CSV files.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        base_dir: Base directory to save files
        prefix: Prefix for filenames
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(base_dir / f"{prefix}_train.csv", index=False)
    val_df.to_csv(base_dir / f"{prefix}_val.csv", index=False)
    test_df.to_csv(base_dir / f"{prefix}_test.csv", index=False)