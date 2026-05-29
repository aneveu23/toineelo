from math import log

import numba
import numpy as np
from numpy.typing import NDArray

from ..kernel import Kernel
from .fitter import Fitter


@numba.jit(nopython=True)
def _fit(
    ts: NDArray,
    ms: NDArray,
    vs: NDArray,
    ns: NDArray,
    xs: NDArray,
    h: NDArray,
    I: NDArray,
    A: NDArray,
    Q: NDArray,
    m_p: NDArray,
    P_p: NDArray,
    m_f: NDArray,
    P_f: NDArray,
    m_s: NDArray,
    P_s: NDArray,
) -> None:
    # Forward pass: Kalman filter.
    for i in range(len(ts)):
        if i > 0:
            m_p[i] = np.dot(A[i - 1], m_f[i - 1])
            P_p[i] = np.dot(np.dot(A[i - 1], P_f[i - 1]), A[i - 1].T) + Q[i - 1]

        # Modified equations using natural pseudo-observation params xs/ns.
        k = np.dot(P_p[i], h) / (1 + xs[i] * np.dot(np.dot(h, P_p[i]), h))
        m_f[i] = m_p[i] + k * (ns[i] - xs[i] * np.dot(h, m_p[i]))

        # Joseph-form covariance update.
        Z = I - np.outer(xs[i] * k, h)
        P_f[i] = np.dot(np.dot(Z, P_p[i]), Z.T) + xs[i] * np.outer(k, k)

    # Backward pass: RTS smoother.
    for i in range(len(ts) - 1, -1, -1):
        if i == len(ts) - 1:
            m_s[i] = m_f[i]
            P_s[i] = P_f[i]
        else:
            G = np.linalg.solve(P_p[i + 1], np.dot(A[i], P_f[i])).T
            m_s[i] = m_f[i] + np.dot(G, m_s[i + 1] - m_p[i + 1])
            P_s[i] = P_f[i] + np.dot(np.dot(G, P_s[i + 1] - P_p[i + 1]), G.T)

        ms[i] = np.dot(h, m_s[i])
        vs[i] = np.dot(np.dot(h, P_s[i]), h)


class RecursiveFitter(Fitter):
    """
    Recursive GP fitter.

    Important implementation detail:
    observations may add sample locations in arbitrary order. That is especially
    important for shared age curves, where samples are ages, not chronological
    game times.

    Public arrays self.ts/self.ms/self.vs/self.ns/self.xs remain in insertion
    order so Observation objects can keep using the integer indices returned by
    add_sample(). Internally, this fitter sorts sample locations before running
    the Kalman filter / RTS smoother.
    """

    def __init__(self, kernel: Kernel):
        super().__init__(kernel)

        m = kernel.order
        self._h = kernel.measurement_vector
        self._I = np.eye(m)

        # Internal sorted representation used by the Kalman/RTS recursions.
        self._sort_idx = np.zeros(0, dtype=int)
        self._inv_sort_idx = np.zeros(0, dtype=int)
        self._ts_sorted = np.zeros(0)

        self._A = np.zeros((0, m, m))
        self._Q = np.zeros((0, m, m))

        self._m_p = np.zeros((0, m))
        self._P_p = np.zeros((0, m, m))
        self._m_f = np.zeros((0, m))
        self._P_f = np.zeros((0, m, m))
        self._m_s = np.zeros((0, m))
        self._P_s = np.zeros((0, m, m))

    def _rebuild_sorted_state(self) -> None:
        """
        Rebuild all internal state-space arrays in sorted sample-location order.

        This is what makes the recursive fitter order invariant. The public
        sample indices do not change; only the internal Kalman ordering changes.
        """
        n = len(self.ts)
        m = self.kernel.order

        if n == 0:
            self._sort_idx = np.zeros(0, dtype=int)
            self._inv_sort_idx = np.zeros(0, dtype=int)
            self._ts_sorted = np.zeros(0)

            self._A = np.zeros((0, m, m))
            self._Q = np.zeros((0, m, m))

            self._m_p = np.zeros((0, m))
            self._P_p = np.zeros((0, m, m))
            self._m_f = np.zeros((0, m))
            self._P_f = np.zeros((0, m, m))
            self._m_s = np.zeros((0, m))
            self._P_s = np.zeros((0, m, m))
            return

        # Stable sort is useful when there are repeated ages/game times.
        self._sort_idx = np.argsort(self.ts, kind="mergesort")
        self._inv_sort_idx = np.empty(n, dtype=int)
        self._inv_sort_idx[self._sort_idx] = np.arange(n)
        self._ts_sorted = self.ts[self._sort_idx]

        self._A = np.zeros((n, m, m))
        self._Q = np.zeros((n, m, m))

        self._m_p = np.array(
            [self.kernel.state_mean(float(t)) for t in self._ts_sorted]
        )
        self._P_p = np.array(
            [self.kernel.state_cov(float(t)) for t in self._ts_sorted]
        )

        self._m_f = self._m_p.copy()
        self._P_f = self._P_p.copy()
        self._m_s = self._m_p.copy()
        self._P_s = self._P_p.copy()

        for i in range(1, n):
            t_prev = float(self._ts_sorted[i - 1])
            t_cur = float(self._ts_sorted[i])
            self._A[i - 1] = self.kernel.transition(t_prev, t_cur)
            self._Q[i - 1] = self.kernel.noise_cov(t_prev, t_cur)

    def allocate(self) -> None:
        """
        Allocate pending samples.

        Public arrays are still in insertion order because observations store
        the indices returned by add_sample(). Internal arrays are rebuilt in
        sorted order so recursive fitting is valid for both chronological
        player curves and non-chronological age curves.
        """
        n_new = len(self.ts_new)
        if n_new == 0:
            return

        ts_new = np.asarray(self.ts_new, dtype=float)
        prior_ms = self.kernel.mean(ts_new)
        
        self.ts = np.concatenate((self.ts, ts_new))
        self.ms = np.concatenate((self.ms, prior_ms))
        self.vs = np.concatenate((self.vs, self.kernel.k_diag(ts_new)))
        self.ns = np.concatenate((self.ns, np.zeros(n_new)))
        self.xs = np.concatenate((self.xs, np.zeros(n_new)))

        self.ts_new = list()

        self._rebuild_sorted_state()

    def fit(self) -> None:
        if not self.is_allocated:
            raise RuntimeError("new data since last call to `allocate()`")

        n = len(self.ts)
        if n == 0:
            self.is_fitted = True
            return

        if len(self._sort_idx) != n:
            self._rebuild_sorted_state()

        order = self._sort_idx

        # Gather pseudo-observation params in sorted order.
        ts_sorted = self._ts_sorted
        ns_sorted = self.ns[order].copy()
        xs_sorted = self.xs[order].copy()

        # These are overwritten by _fit in sorted order.
        ms_sorted = np.zeros(n)
        vs_sorted = np.zeros(n)

        _fit(
            ts=ts_sorted,
            ms=ms_sorted,
            vs=vs_sorted,
            ns=ns_sorted,
            xs=xs_sorted,
            h=self._h,
            I=self._I,
            A=self._A,
            Q=self._Q,
            m_p=self._m_p,
            P_p=self._P_p,
            m_f=self._m_f,
            P_f=self._P_f,
            m_s=self._m_s,
            P_s=self._P_s,
        )

        # Scatter posterior marginals back to public insertion-order arrays.
        # This keeps observation indices valid.
        self.ms[order] = ms_sorted
        self.vs[order] = vs_sorted

        self.is_fitted = True

    @property
    def ep_log_likelihood_contrib(self) -> float:
        if not self.is_fitted:
            raise RuntimeError("new data since last call to `fit()`")

        if len(self.ts) == 0:
            return 0.0

        order = self._sort_idx
        h = self._h
        m_p, P_p = self._m_p, self._P_p
        ns = self.ns[order]
        xs = self.xs[order]

        val = 0.0

        for i in range(len(order)):
            o = h.dot(m_p[i])
            v = h.dot(P_p[i]).dot(h)

            val += -0.5 * (
                log(xs[i] * v + 1.0)
                + (-(ns[i] ** 2) * v - 2 * ns[i] * o + xs[i] * o**2)
                / (xs[i] * v + 1.0)
            )

        return val

    @property
    def kl_log_likelihood_contrib(self) -> float:
        if not self.is_fitted:
            raise RuntimeError("new data since last call to `fit()`")

        if len(self.ts) == 0:
            return 0.0

        order = self._sort_idx
        h = self._h
        ns = self.ns[order]
        xs = self.xs[order]

        val = 0.0

        for i in range(len(order)):
            # Marginal predictive distribution.
            mp = np.dot(h, self._m_p[i])
            vp = np.dot(h, np.dot(self._P_p[i], h))

            # Marginal smoothed distribution.
            ms = np.dot(h, self._m_s[i])
            vs = np.dot(h, np.dot(self._P_s[i], h))

            val += -0.5 * (
                log(xs[i] * vp + 1.0)
                + xs[i] * (mp * mp - ms * ms - vs)
                - 2.0 * ns[i] * (mp - ms)
                - (xs[i] * mp - ns[i]) ** 2 / (1.0 / vp + xs[i])
            )

        return val

    def predict(self, ts: NDArray) -> tuple[NDArray, NDArray]:
        if not self.is_fitted:
            raise RuntimeError("new data since last call to `fit()`")

        ts = np.asarray(ts, dtype=float)

        if len(self.ts) == 0:
            return (self.kernel.mean(ts), self.kernel.k_diag(ts))

        ts_train = self._ts_sorted
        ms = np.zeros(len(ts))
        vs = np.zeros(len(ts))

        h = self._h
        m_p, P_p = self._m_p, self._P_p
        m_f, P_f = self._m_f, self._P_f
        m_s, P_s = self._m_s, self._P_s

        locations = np.searchsorted(ts_train, ts)

        for i, nxt in enumerate(locations):
            t = float(ts[i])

            if nxt == len(ts_train):
                # Prediction point is after the last training location.
                A = self.kernel.transition(float(ts_train[-1]), t)
                Q = self.kernel.noise_cov(float(ts_train[-1]), t)

                pred_state_mean = np.dot(A, m_s[-1])
                pred_state_cov = A.dot(P_s[-1]).dot(A.T) + Q

                ms[i] = h.dot(pred_state_mean)
                vs[i] = h.dot(pred_state_cov).dot(h)

            else:
                j = nxt - 1

                if j < 0:
                    # Prediction point is before or at the first training location.
                    pred_state_mean = self.kernel.state_mean(t)
                    pred_state_cov = self.kernel.state_cov(t)
                else:
                    # Predict from the left neighbor.
                    A_left = self.kernel.transition(float(ts_train[j]), t)
                    Q_left = self.kernel.noise_cov(float(ts_train[j]), t)

                    pred_state_mean = A_left.dot(m_f[j])
                    pred_state_cov = A_left.dot(P_f[j]).dot(A_left.T) + Q_left

                # Smooth using the right neighbor.
                A_right = self.kernel.transition(t, float(ts_train[j + 1]))
                G = np.linalg.solve(
                    P_p[j + 1],
                    A_right.dot(pred_state_cov),
                ).T

                smoothed_state_mean = pred_state_mean + G.dot(
                    m_s[j + 1] - m_p[j + 1]
                )
                smoothed_state_cov = pred_state_cov + G.dot(
                    P_s[j + 1] - P_p[j + 1]
                ).dot(G.T)

                ms[i] = h.dot(smoothed_state_mean)
                vs[i] = h.dot(smoothed_state_cov).dot(h)

        return (ms, vs)
