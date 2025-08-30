"""Test that temperature scaling + class-prior balancing yields softer predictions."""
import numpy as np
import torch

from pipeline.tasks import torch as torch_lib  # reuse same torch


def _softmax(x):
    x = torch.tensor(x, dtype=torch.float32)
    return torch.softmax(x, dim=-1).numpy()


def test_temperature_softening():
    logits = np.array([[10.0, 0.5, -1.0]])
    probs_T1 = _softmax(logits / 1.0)[0]
    probs_T5 = _softmax(logits / 5.0)[0]

    assert probs_T5[0] < probs_T1[0], "Higher temperature should lower peak prob"
    np.testing.assert_allclose(probs_T5.sum(), 1.0, atol=1e-6)


def test_class_prior_balancing():
    logits = np.array([[2.0, 1.0, 0.5]])
    probs = _softmax(logits)[0]
    priors = np.array([0.2, 0.5, 0.3])
    balanced = probs / priors
    balanced /= balanced.sum()

    assert balanced[0] > probs[0], "Prob of under-represented class should increase"
    np.testing.assert_allclose(balanced.sum(), 1.0, atol=1e-6)
