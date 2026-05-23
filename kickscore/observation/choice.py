from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ..item import Item
from .observation import Observation


def _logsumexp(xs: NDArray[np.float64], axis: int | None = None) -> NDArray[np.float64] | float:
    max_xs = np.max(xs, axis=axis, keepdims=True)
    out = max_xs + np.log(np.sum(np.exp(xs - max_xs), axis=axis, keepdims=True))
    out = np.squeeze(out, axis=axis)
    if axis is None:
        return float(out)
    return out


def _softmax(xs: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = xs - np.max(xs, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)


def _normal_draws(
    num_samples: int, num_choices: int, random_state: int | None
) -> NDArray[np.float64]:
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    rng = np.random.default_rng(random_state)
    half = num_samples // 2
    draws = rng.standard_normal((half, num_choices))
    if num_samples % 2 == 0:
        return np.vstack((draws, -draws))
    return np.vstack((draws, -draws, np.zeros((1, num_choices))))


def _choice_expectations(
    mean: NDArray[np.float64],
    var: NDArray[np.float64],
    winner: int,
    eps: NDArray[np.float64],
    temperature: float,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    z = mean[None, :] + np.sqrt(np.maximum(var, 0.0))[None, :] * eps
    scaled_z = z / temperature
    probs = _softmax(scaled_z)
    exp_ll = float(mean[winner] / temperature - np.mean(_logsumexp(scaled_z, axis=1)))
    pbar = np.mean(probs, axis=0)
    curv = np.mean(probs * (1.0 - probs), axis=0)
    grad = -pbar
    grad[winner] += 1.0
    return exp_ll, grad / temperature, curv / (temperature * temperature)


class PlackettLuceObservation(Observation):
    def __init__(
        self,
        alternatives,
        winner: int,
        t: float,
        num_samples: int = 128,
        random_state: int | None = None,
        temperature: float = 1.0,
    ):
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if len(alternatives) < 2:
            raise ValueError("need at least two alternatives per choice observation")
        if winner < 0 or winner >= len(alternatives):
            raise ValueError("winner index is outside of the choice set")

        self._M = len(alternatives)
        self._winner = winner
        self._temperature = temperature
        self._eps = _normal_draws(num_samples, self._M, random_state)
        self.t = t
        self._exp_ll = 0

        self._alternatives = []

        for alt in alternatives:
            stored_alt = []
            for item, coeff, term_t in alt:
                idx = item.fitter.add_sample(float(term_t))
                stored_alt.append((item, float(coeff), idx))
            self._alternatives.append(stored_alt)

    def kl_update(self, lr: float = 0.3) -> float:
        mean = np.zeros(self._M)
        var = np.zeros(self._M)

        for a, alt in enumerate(self._alternatives):
            for item, coeff, idx in alt:
                mean[a] += coeff * item.fitter.ms[idx]
                var[a] += coeff * coeff * item.fitter.vs[idx]

        exp_ll, grad, tau_utility = _choice_expectations(
            mean, var, self._winner, self._eps, self._temperature
        )

        for a, alt in enumerate(self._alternatives):
            for item, coeff, idx in alt:
                item_mean = item.fitter.ms[idx]
                x = coeff * coeff * tau_utility[a]
                n = coeff * grad[a] + x * item_mean

                item.fitter.xs[idx] = (1.0 - lr) * item.fitter.xs[idx] + lr * x
                item.fitter.ns[idx] = (1.0 - lr) * item.fitter.ns[idx] + lr * n

        diff = abs(self._exp_ll - exp_ll)
        self._exp_ll = exp_ll
        return diff

    @staticmethod
    def probability(
        alternatives,
        t: float,
        num_samples: int = 128,
        random_state: int | None = None,
        integrate: bool = True,
        temperature: float = 1.0,
    ) -> tuple[float, ...]:
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        mean = np.zeros(len(alternatives))
        var = np.zeros(len(alternatives))

        for a, alt in enumerate(alternatives):
            for item, coeff, term_t in alt:
                ms, vs = item.predict(np.array([float(term_t)]))
                mean[a] += coeff * ms[0]
                var[a] += coeff * coeff * vs[0]

        if integrate:
            eps = _normal_draws(num_samples, len(alternatives), random_state)
            z = mean[None, :] + np.sqrt(np.maximum(var, 0.0))[None, :] * eps
            probs = np.mean(_softmax(z / temperature), axis=0)
        else:
            probs = _softmax(mean[None, :] / temperature)[0]

        return tuple(float(p) for p in probs)
