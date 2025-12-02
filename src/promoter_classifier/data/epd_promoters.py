"""Module for handling EPD promoter/negative dataset paths and CSV-based loader."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd


@dataclass
class EPDDatasetPaths:
    """
    Container for file paths related to the EPD-based promoter classification dataset.
    """
    promoters_path: Path
    negatives_path: Path
    metadata_path: Optional[Path] = None
    
    def __post_init__(self):
        """Convert inputs to Path objects."""
        self.promoters_path = Path(self.promoters_path)
        self.negatives_path = Path(self.negatives_path)
        if self.metadata_path is not None:
            self.metadata_path = Path(self.metadata_path)
            
        # Ensure parent directories exist
        self.promoters_path.parent.mkdir(parents=True, exist_ok=True)
        self.negatives_path.parent.mkdir(parents=True, exist_ok=True)
        if self.metadata_path is not None:
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)


def load_promoter_table(path: Union[str, Path]) -> pd.DataFrame:
    """
    Simple helper to load EPD-derived promoter tables from CSV files.
    
    Args:
        path: Path to the CSV file
        
    Returns:
        DataFrame containing the promoter data
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file format is not supported
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
        
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Expected a .csv file.")