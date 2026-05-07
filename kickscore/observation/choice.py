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
    return exp_shifted / np.sum(exp_shifted, ahttps://github.com/aneveu23/toineelo/pull/2/conflict?name=kickscore%252Fmodel.py&ancestor_oid=ab7531c9e2e68401464e2659cf2c2d6d59c2fb2d&base_oid=380b100e77d5168e932593bba7c6430a3989fa15&head_oid=a5f1669c8993b8a6999d9ff5a6f44d86293cc826xis=1, keepdims=True)


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
    """Top-1 Plackett-Luce / conditional-logit observation over a choice set.

    The KL/CVI update uses a diagonal Gaussian pseudo-observation per alternative.
    Expectations over posterior uncertainty are approximated with fixed antithetic
    Monte Carlo draws so repeated variational iterations are deterministic.
    """

    def __init__(
        self,
        elems: Sequence[tuple[Item, float]],
        winner: int,
        t: float,
        num_samples: int = 128,
        random_state: int | None = None,
        temperature: float = 1.0,
    ):
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if len(elems) < 2:
            raise ValueError("need at least two alternatives per choice observation")
        if winner < 0 or winner >= len(elems):
            raise ValueError("winner index is outside of the choice set")
        super().__init__(elems, t)
        self._winner = winner
        self._temperature = temperature
        self._eps = _normal_draws(num_samples, self._M, random_state)

    def match_moments(self, mean_cav: float, var_cav: float) -> tuple[float, float, float]:
        raise NotImplementedError("Plackett-Luce observations only support method='kl'")

    def cvi_expectations(self, mean: float, var: float) -> tuple[float, float, float]:
        raise NotImplementedError("use kl_update() for vector-valued Plackett-Luce updates")

    def ep_update(self, lr: float = 1.0) -> float:
        raise NotImplementedError("Plackett-Luce observations only support method='kl'")

    def kl_update(self, lr: float = 0.3) -> float:
        mean = np.zeros(self._M)
        var = np.zeros(self._M)
        for i in range(self._M):
            item = self._items[i]
            idx = self._indices[i]
            coeff = self._coeffs[i]
            mean[i] = coeff * item.fitter.ms[idx]
            var[i] = coeff * coeff * item.fitter.vs[idx]

        exp_ll, grad, tau_utility = _choice_expectations(
            mean, var, self._winner, self._eps, self._temperature
        )

        for i in range(self._M):
            item = self._items[i]
            idx = self._indices[i]
            coeff = self._coeffs[i]
            item_mean = item.fitter.ms[idx]
            x = coeff * coeff * tau_utility[i]
            n = coeff * grad[i] + x * item_mean
            item.fitter.xs[idx] = (1.0 - lr) * item.fitter.xs[idx] + lr * x
            item.fitter.ns[idx] = (1.0 - lr) * item.fitter.ns[idx] + lr * n

        diff = abs(self._exp_ll - exp_ll)
        self._exp_ll = exp_ll
        return diff

    @staticmethod
    def probability(
        elems: Sequence[tuple[Item, float]],
        t: float,
        num_samples: int = 128,
        random_state: int | None = None,
        integrate: bool = True,
        temperature: float = 1.0,
    ) -> tuple[float, ...]:
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        mean = np.zeros(len(elems))
        var = np.zeros(len(elems))
        ts = np.array([t])
        for i, (item, coeff) in enumerate(elems):
            ms, vs = item.predict(ts)
            mean[i] = coeff * ms[0]
            var[i] = coeff * coeff * vs[0]
        if integrate:
            eps = _normal_draws(num_samples, len(elems), random_state)
            z = mean[None, :] + np.sqrt(np.maximum(var, 0.0))[None, :] * eps
            probs = np.mean(_softmax(z / temperature), axis=0)
        else:
            probs = _softmax(mean[None, :] / temperature)[0]
        return tuple(float(p) for p in probs)
