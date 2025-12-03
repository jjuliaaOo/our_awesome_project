# File: tests/test_kmer_features.py
import numpy as np
import pytest

from promoter_classifier.features import (
    ALPHABET,
    build_kmer_vocabulary,
    sequence_to_kmer_counts,
    sequences_to_kmer_matrix,
)


def test_build_kmer_vocabulary_k1():
    vocab = build_kmer_vocabulary(1, alphabet=ALPHABET)
    # Should contain exactly the alphabet
    assert set(vocab.keys()) == set(ALPHABET)
    assert len(vocab) == len(ALPHABET)


def test_sequence_to_kmer_counts_simple():
    vocab = build_kmer_vocabulary(2, alphabet=ALPHABET)
    seq = "ACGT"
    counts = sequence_to_kmer_counts(seq, k=2, vocab=vocab)

    # AC, CG, GT are present; number of 2-mers is len(seq) - 1 = 3
    assert counts.sum() == pytest.approx(3.0)

    # Check that specific kmers are counted once
    idx_ac = vocab["AC"]
    idx_cg = vocab["CG"]
    idx_gt = vocab["GT"]
    assert counts[idx_ac] == pytest.approx(1.0)
    assert counts[idx_cg] == pytest.approx(1.0)
    assert counts[idx_gt] == pytest.approx(1.0)


def test_sequences_to_kmer_matrix_shape_and_normalization():
    sequences = ["AAAA", "ACGT"]
    X, vocab = sequences_to_kmer_matrix(sequences, k=1, alphabet=ALPHABET, normalize=True)

    assert X.shape == (2, len(vocab))

    # Each row should sum to 1.0 if there was at least one k-mer
    row_sums = X.sum(axis=1)
    assert row_sums[0] == pytest.approx(1.0)
    assert row_sums[1] == pytest.approx(1.0)


def test_sequences_to_kmer_matrix_empty_sequence():
    sequences = [""]
    X, vocab = sequences_to_kmer_matrix(sequences, k=2, alphabet=ALPHABET, normalize=True)

    # No k-mers can be formed, so the row should be all zeros
    assert X.shape[0] == 1
    assert np.allclose(X[0], 0.0)


def test_build_kmer_vocabulary_invalid_k():
    with pytest.raises(ValueError, match="k must be positive"):
        build_kmer_vocabulary(0, alphabet=ALPHABET)
        
    with pytest.raises(ValueError, match="k must be positive"):
        build_kmer_vocabulary(-1, alphabet=ALPHABET)


def test_build_kmer_vocabulary_empty_alphabet():
    with pytest.raises(ValueError, match="alphabet must not be empty"):
        build_kmer_vocabulary(2, alphabet=[])


def test_sequence_to_kmer_counts_edge_cases():
    vocab = build_kmer_vocabulary(2, alphabet=ALPHABET)
    
    # Empty sequence
    counts = sequence_to_kmer_counts("", k=2, vocab=vocab)
    assert np.allclose(counts, 0.0)
    
    # Sequence shorter than k
    counts = sequence_to_kmer_counts("A", k=2, vocab=vocab)
    assert np.allclose(counts, 0.0)
    
    # Invalid k
    counts = sequence_to_kmer_counts("ACGT", k=0, vocab=vocab)
    assert np.allclose(counts, 0.0)


def test_sequences_to_kmer_matrix_no_normalization():
    sequences = ["AAAA", "CCCC"]
    X, vocab = sequences_to_kmer_matrix(sequences, k=1, alphabet=ALPHABET, normalize=False)
    
    assert X.shape == (2, len(vocab))
    
    # For "AAAA", the count of "A" should be 4
    idx_a = vocab["A"]
    assert X[0, idx_a] == pytest.approx(4.0)
    
    # For "CCCC", the count of "C" should be 4
    idx_c = vocab["C"]
    assert X[1, idx_c] == pytest.approx(4.0)