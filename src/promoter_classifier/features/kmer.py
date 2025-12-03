# File: src/promoter_classifier/features/kmer.py
from __future__ import annotations

from itertools import product
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

# Default DNA alphabet
ALPHABET: Tuple[str, ...] = ("A", "C", "G", "T")


def build_kmer_vocabulary(
    k: int,
    alphabet: Sequence[str] = ALPHABET,
) -> Dict[str, int]:
    """
    Build a mapping from k-mer string to column index.

    Args:
        k: Length of the k-mers to generate.
        alphabet: Collection of allowed symbols (e.g. ['A', 'C', 'G', 'T']).

    Returns:
        A dict mapping each possible k-mer (e.g. 'AAA', 'AAC', ...) to a unique integer index.

    Raises:
        ValueError: If k <= 0 or alphabet is empty.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not alphabet:
        raise ValueError("alphabet must not be empty")

    vocab: Dict[str, int] = {}
    index = 0
    for letters in product(alphabet, repeat=k):
        kmer = "".join(letters)
        vocab[kmer] = index
        index += 1
    return vocab


def sequence_to_kmer_counts(
    seq: str,
    k: int,
    vocab: Mapping[str, int],
) -> np.ndarray:
    """
    Convert a single sequence into a k-mer count vector.

    Args:
        seq: Input sequence (e.g. DNA string).
        k: Length of k-mers.
        vocab: Mapping from k-mer string to column index.

    Returns:
        A 1D numpy array of length len(vocab) with integer counts for each k-mer.
    """
    counts = np.zeros(len(vocab), dtype=np.float32)
    seq = (seq or "").upper()

    if k <= 0 or len(seq) < k:
        return counts

    for i in range(len(seq) - k + 1):
        kmer = seq[i : i + k]
        # Only count k-mers that exist in the vocabulary
        idx = vocab.get(kmer)
        if idx is not None:
            counts[idx] += 1.0
    return counts


def sequences_to_kmer_matrix(
    sequences: Sequence[str],
    k: int,
    alphabet: Sequence[str] = ALPHABET,
    normalize: bool = False,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Build a 2D feature matrix from a list of sequences using k-mer counts.

    Args:
        sequences: Iterable of input sequences.
        k: Length of k-mers.
        alphabet: Alphabet for k-mers (default: DNA A/C/G/T).
        normalize: If True, convert counts to per-sequence frequencies.

    Returns:
        A tuple of (X, vocab) where:
            - X is a numpy array of shape (n_sequences, n_kmers),
            - vocab is the k-mer -> column index mapping used.
    """
    vocab = build_kmer_vocabulary(k=k, alphabet=alphabet)
    n = len(sequences)
    d = len(vocab)

    X = np.zeros((n, d), dtype=np.float32)

    for i, seq in enumerate(sequences):
        counts = sequence_to_kmer_counts(seq, k=k, vocab=vocab)
        if normalize:
            total = counts.sum()
            if total > 0:
                counts = counts / total
        X[i, :] = counts

    return X, vocab