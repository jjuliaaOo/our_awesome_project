# File: src/promoter_classifier/models/baseline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from promoter_classifier.features import (
    ALPHABET,
    sequences_to_kmer_matrix,
)


@dataclass
class KmerLogisticBaseline:
    """
    Simple baseline classifier that uses k-mer frequency features
    and scikit-learn's LogisticRegression for binary promoter classification.
    """
    k: int = 3
    normalize: bool = True
    alphabet: Sequence[str] = ALPHABET
    C: float = 1.0
    max_iter: int = 1000
    random_state: Optional[int] = 42

    def __post_init__(self) -> None:
        # Underlying sklearn model
        self._model: Optional[LogisticRegression] = None
        self._vocab_: Optional[dict] = None

    def _ensure_fitted(self) -> None:
        if self._model is None or self._vocab_ is None:
            raise RuntimeError("Model is not fitted yet. Call 'fit' first.")

    def fit(self, sequences: Sequence[str], y: Sequence[int]) -> "KmerLogisticBaseline":
        """
        Fit the baseline model on raw sequences and binary labels.

        Args:
            sequences: Iterable of DNA sequences.
            y: Iterable of binary labels (0/1).

        Returns:
            self
        """
        X, vocab = sequences_to_kmer_matrix(
            sequences,
            k=self.k,
            alphabet=self.alphabet,
            normalize=self.normalize,
        )
        self._vocab_ = vocab

        clf = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        clf.fit(X, np.asarray(y, dtype=np.int32))

        self._model = clf
        return self

    def predict_proba(self, sequences: Sequence[str]) -> np.ndarray:
        """
        Predict class probabilities for new sequences.

        Args:
            sequences: Iterable of DNA sequences.

        Returns:
            Array of shape (n_samples, 2) with class probabilities [P(y=0), P(y=1)].
        """
        self._ensure_fitted()
        assert self._model is not None
        assert self._vocab_ is not None

        from promoter_classifier.features import sequence_to_kmer_counts  # local import

        n = len(sequences)
        d = len(self._vocab_)
        X = np.zeros((n, d), dtype=np.float32)

        for i, seq in enumerate(sequences):
            counts = sequence_to_kmer_counts(seq, k=self.k, vocab=self._vocab_)
            if self.normalize:
                total = counts.sum()
                if total > 0:
                    counts = counts / total
            X[i, :] = counts

        return self._model.predict_proba(X)

    def predict(self, sequences: Sequence[str]) -> np.ndarray:
        """
        Predict binary class labels for new sequences.

        Args:
            sequences: Iterable of DNA sequences.

        Returns:
            Array of shape (n_samples,) with predicted labels 0/1.
        """
        proba = self.predict_proba(sequences)
        return (proba[:, 1] >= 0.5).astype(np.int32)