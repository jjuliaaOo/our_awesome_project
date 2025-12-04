# File: src/promoter_classifier/transformers/dnabert_dataset.py
"""DNABERT dataset implementation for promoter classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Any, Dict


@dataclass
class DNABERTConfig:
    """Configuration for DNABERT model training."""
    model_name: str = "dnabert"
    max_length: int = 256
    batch_size: int = 16
    lr: float = 2e-5
    num_epochs: int = 3
    random_state: int = 42

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.lr <= 0:
            raise ValueError("lr must be positive")


class PromoterDNABERTDataset:
    """
    Dataset class for promoter classification using DNABERT.
    
    This dataset wraps DNA sequences and their labels for use with DNABERT-style models.
    It uses a generic tokenizer interface that should be compatible with HuggingFace tokenizers.
    
    The dataset returns dictionaries with the following keys:
    - "input_ids": Tokenized sequence IDs
    - "attention_mask": Attention mask for the sequence
    - "labels": Classification label (0 or 1)
    """

    def __init__(
        self,
        sequences: List[str],
        labels: List[int],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        """
        Initialize the dataset.
        
        Args:
            sequences: List of DNA sequences
            labels: List of binary labels (0 or 1)
            tokenizer: HF-compatible tokenizer
            max_length: Maximum sequence length for tokenization
        """
        if len(sequences) != len(labels):
            raise ValueError("Length of sequences and labels must match")
            
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single example from the dataset.
        
        Args:
            idx: Index of the example to retrieve
            
        Returns:
            Dictionary with keys "input_ids", "attention_mask", and "labels"
        """
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        tokenized = self.tokenizer(
            sequence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": int(label),
        }