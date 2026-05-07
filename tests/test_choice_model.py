import numpy as np
import pytest

import kickscore as ks


def _model() -> ks.ChoiceModel:
    model = ks.ChoiceModel(num_samples=64, random_state=12)
    kernel = ks.kernel.Constant(1.0)
    for name in ["A", "B", "C"]:
        model.add_item(name, kernel=kernel)
    return model


def test_choice_probabilities_are_normalized_before_fit():
    model = _model()
    model.fit()
    probs = model.probabilities(["A", "B", "C"], t=0.0)
    assert np.allclose(sum(probs), 1.0)
    assert np.allclose(probs, [1 / 3, 1 / 3, 1 / 3], atol=0.03)


def test_choice_model_learns_frequent_winner():
    model = _model()
    for i in range(12):
        model.observe(["A", "B", "C"], winner="A", t=float(i))
    for i in range(12, 16):
        model.observe(["A", "B", "C"], winner="B", t=float(i))
    assert model.fit(tol=1e-4, max_iter=50)
    probs = model.probabilities(["A", "B", "C"], t=16.0, integrate=False)
    assert probs[0] > probs[1] > probs[2]
    assert np.allclose(sum(probs), 1.0)


def test_choice_model_validates_observations_and_method():
    model = _model()
    with pytest.raises(ValueError, match="winner"):
        model.observe(["A", "B", "C"], winner="D", t=0.0)
    model.observe(["A", "B", "C"], winner="A", t=1.0)
    with pytest.raises(ValueError, match="chronological"):
        model.observe(["A", "B", "C"], winner="A", t=0.0)
    with pytest.raises(ValueError, match="method='kl'"):
        model.fit(method="ep")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="temperature"):
        ks.ChoiceModel(temperature=0.0)


def test_choice_temperature_controls_softmax_sharpness():
    model = _model()
    for i in range(12):
        model.observe(["A", "B", "C"], winner="A", t=float(i))
    assert model.fit(tol=1e-4, max_iter=50)

    model.temperature = 0.5
    cold_probs = model.probabilities(["A", "B", "C"], t=12.0, integrate=False)
    model.temperature = 2.0
    hot_probs = model.probabilities(["A", "B", "C"], t=12.0, integrate=False)

    assert cold_probs[0] > hot_probs[0]
    assert cold_probs[-1] < hot_probs[-1]
    assert np.allclose(sum(cold_probs), 1.0)
    assert np.allclose(sum(hot_probs), 1.0)
