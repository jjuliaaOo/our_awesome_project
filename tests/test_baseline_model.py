# File: tests/test_baseline_model.py
import numpy as np
import pytest

from promoter_classifier.models import KmerLogisticBaseline


def test_baseline_fit_and_predict_shapes():
    sequences = ["AAAA", "CCCC", "AAAA", "CCCC"]
    y = [1, 0, 1, 0]

    model = KmerLogisticBaseline(k=1, normalize=True, max_iter=200)
    model.fit(sequences, y)

    proba = model.predict_proba(sequences)
    preds = model.predict(sequences)

    assert proba.shape == (len(sequences), 2)
    assert preds.shape == (len(sequences),)
    # Probabilities per sample should sum to ~1
    row_sums = proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)


def test_baseline_overfits_simple_dataset():
    # Simple synthetic dataset where "AAAA" is always class 1 and "CCCC" always class 0
    sequences = ["AAAA", "AAAA", "CCCC", "CCCC"]
    y = np.array([1, 1, 0, 0], dtype=np.int32)

    model = KmerLogisticBaseline(k=1, normalize=True, max_iter=500)
    model.fit(sequences, y)

    preds = model.predict(sequences)
    accuracy = (preds == y).mean()

    assert accuracy >= 0.99


def test_baseline_predict_before_fit_raises_error():
    model = KmerLogisticBaseline()
    with pytest.raises(RuntimeError, match="Model is not fitted yet"):
        model.predict(["AAAA"])


def test_baseline_predict_proba_before_fit_raises_error():
    model = KmerLogisticBaseline()
    with pytest.raises(RuntimeError, match="Model is not fitted yet"):
        model.predict_proba(["AAAA"])


def test_baseline_with_different_k_values():
    sequences = ["AAAA", "AAAC", "CCCC", "CCCG"]
    y = [1, 1, 0, 0]

    # Test with k=2
    model = KmerLogisticBaseline(k=2, normalize=True, max_iter=300)
    model.fit(sequences, y)

    proba = model.predict_proba(sequences)
    preds = model.predict(sequences)

    assert proba.shape == (len(sequences), 2)
    assert preds.shape == (len(sequences),)


def test_baseline_empty_sequences():
    sequences = ["", "AAAA", "", "CCCC"]
    y = [1, 1, 0, 0]

    model = KmerLogisticBaseline(k=1, normalize=True, max_iter=200)
    model.fit(sequences, y)

    proba = model.predict_proba(sequences)
    preds = model.predict(sequences)

    assert proba.shape == (len(sequences), 2)
    assert preds.shape == (len(sequences),)