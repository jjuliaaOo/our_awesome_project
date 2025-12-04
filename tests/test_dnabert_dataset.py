# File: tests/test_dnabert_dataset.py
"""Tests for the DNABERT dataset implementation."""

import pytest

from promoter_classifier.transformers import DNABERTConfig, PromoterDNABERTDataset


class FakeTokenizer:
    """Fake tokenizer for testing purposes."""
    
    def __call__(self, text, max_length, padding, truncation, return_tensors=None):
        # For testing, just produce deterministic dummy ids based on text length
        length = len(text)
        input_ids = [length] * max_length
        attention_mask = [1] * max_length
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def test_dnabert_config_validate_ok():
    """Test that DNABERTConfig validates correctly with default values."""
    config = DNABERTConfig()
    # Should not raise an exception
    config.validate()


def test_dnabert_dataset_basic():
    """Test basic functionality of PromoterDNABERTDataset."""
    sequences = ["ACGT", "AAAA"]
    labels = [1, 0]
    tokenizer = FakeTokenizer()
    
    ds = PromoterDNABERTDataset(sequences, labels, tokenizer=tokenizer, max_length=8)
    
    assert len(ds) == 2
    
    item0 = ds[0]
    assert set(item0.keys()) == {"input_ids", "attention_mask", "labels"}
    assert isinstance(item0["labels"], int)
    assert len(item0["input_ids"]) == 8
    assert len(item0["attention_mask"]) == 8


def test_dnabert_dataset_mismatched_lengths():
    """Test that PromoterDNABERTDataset raises ValueError for mismatched sequences/labels."""
    sequences = ["ACGT"]
    labels = [1, 0]  # Different length than sequences
    
    with pytest.raises(ValueError, match="Length of sequences and labels must match"):
        PromoterDNABERTDataset(sequences, labels, tokenizer=FakeTokenizer(), max_length=8)


def test_dnabert_config_validation_errors():
    """Test that DNABERTConfig validation catches invalid parameters."""
    # Test invalid max_length
    config = DNABERTConfig(max_length=-1)
    with pytest.raises(ValueError, match="max_length must be positive"):
        config.validate()
        
    # Test invalid batch_size
    config = DNABERTConfig(batch_size=0)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        config.validate()
        
    # Test invalid num_epochs
    config = DNABERTConfig(num_epochs=-5)
    with pytest.raises(ValueError, match="num_epochs must be positive"):
        config.validate()
        
    # Test invalid lr
    config = DNABERTConfig(lr=-1e-5)
    with pytest.raises(ValueError, match="lr must be positive"):
        config.validate()