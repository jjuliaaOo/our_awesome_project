"""Data loading and preprocessing functions for promoter classification."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union
import os


@dataclass
class RawDatasetPaths:
    """Paths to raw, interim, and processed data directories."""
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    
    def __post_init__(self):
        """Ensure all directories exist."""
        self.raw_dir = Path(self.raw_dir)
        self.interim_dir = Path(self.interim_dir)
        self.processed_dir = Path(self.processed_dir)
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)


def load_fasta(path: Union[str, Path]) -> List[str]:
    """
    Load sequences from a FASTA file.
    
    Args:
        path: Path to the FASTA file
        
    Returns:
        List of sequences (headers are ignored)
    """
    path = Path(path)
    sequences = []
    current_sequence = ""
    
    with open(path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                # If we have a previous sequence, save it
                if current_sequence:
                    sequences.append(current_sequence.upper())
                    current_sequence = ""
            else:
                # Append sequence line
                current_sequence += line
                
        # Don't forget the last sequence
        if current_sequence:
            sequences.append(current_sequence.upper())
            
    return sequences


def clean_sequence(seq: str) -> str:
    """
    Clean a DNA sequence by keeping only valid nucleotides.
    
    Args:
        seq: DNA sequence to clean
        
    Returns:
        Cleaned sequence containing only A/C/G/T characters
    """
    return ''.join(char for char in seq.upper() if char in 'ACGT')


def is_valid_sequence(seq: str) -> bool:
    """
    Check if a sequence contains only valid nucleotides.
    
    Args:
        seq: DNA sequence to validate
        
    Returns:
        True if sequence contains only A/C/G/T, False otherwise
    """
    return all(char in 'ACGT' for char in seq.upper())


def load_plain_sequences(path: Union[str, Path], clean: bool = True) -> List[str]:
    """
    Load DNA sequences from a plain text file, one sequence per line.

    - Ignore empty lines.
    - Strip whitespace from each line.
    - Convert sequences to uppercase.
    - If `clean` is True, use `clean_sequence()` to keep only A/C/G/T.

    Args:
        path: Path to the plain text file.
        clean: Whether to apply `clean_sequence()` to each sequence.

    Returns:
        List of processed sequences.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    sequences = []
    with open(path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  # Skip empty lines
                seq = line.upper()
                if clean:
                    seq = clean_sequence(seq)
                sequences.append(seq)
    
    return sequences