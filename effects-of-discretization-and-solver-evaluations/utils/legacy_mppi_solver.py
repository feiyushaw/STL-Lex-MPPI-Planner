
"""Reference legacy MPPI optimizer used only for fixed-budget comparisons.

This module is intentionally isolated from the production planner. It preserves
the historical MPPI update rule while using the same effective control horizon
(K-1), input bounds, Lex-STL evaluator, and pre-generated standard-normal
latents as the IGO benchmark.
"""

import math
import time

import numpy as np


class LegacyMPPISolver:
    """Faithful reference MPPI baseline for offline benchmark experiments."""

    def __init__(
        self,
        cost_functions,
        x0,
        K,
        system,
        n_iterations,
        n_samples,
        cov,
        lamb,
        gamma,
        input_clipping,
        pregenerated_standard_samples,
    ):
        self.cost_functions = cost_functions
        self.x0 = float(x0)
        self.K = int(K)
        self.control_horizon = self.K - 1
        self.sys = system
        self.n_iterations = int(n_iterations)
        self.n_samples = int(n_samples)
        self.cov = float(cov)
        self.lamb = float(lamb)
        self.gamma = float(gamma)
        self.input_clipping = bool(input_clipping)
        self.standard_samples = np.asarray(
            pregenerated_standard_samples, dtype=np.float64
        )

        if self.K <= 1:
            raise ValueError("K must be greater than one")
        if self.n_iterations <= 0 or self.n_samples <= 0:
            raise ValueError("MPPI iteration and sample counts must be positive")
        if self.cov <= 0.0 or self.lamb <= 0.0:
            raise ValueError("MPPI covariance and temperature must be positive")
        expected = (self.n_iterations, self.n_samples, self.sys.m, self.control_horizon)
        if self.standard_samples.shape != expected:
            raise ValueError(
                f"standard samples must have shape {expected}, "
                f"got {self.standard_samples.shape}"
            )
        if not np.all(np.isfinite(self.standard_samples)):
            raise ValueError("standard samples contain non-finite values")

        self.power_sums = self._precompute_power_sums()
        self.best_overall_lex_cost = float("inf")
        self.best_overall_u = None
        self.total_rollouts = 0
        self.evaluated_transitions = 0

    def _precompute_power_sums(self):
        bits = [
            math.ceil(math.log2(cf.nof_intervals + 1))
            for cf in self.cost_functions
        ]
        return [1 << int(sum(bits[i + 1:])) for i in range(len(bits))]

    def _beta(self, iteration):
        return math.sqrt(self.gamma ** iteration)

    def _evaluate(self, y):
        rule_costs = [
            int(cf.evaluate_trajectory(y)) for cf in self.cost_functions
        ]
        return sum(
            rule_costs[i] * self.power_sums[i]
            for i in range(len(rule_costs))
        )

    def _rollout(self, u):
        x = np.empty((self.sys.n, self.K), dtype=np.float64)
        x[:, 0] = self.x0
        for k in range(self.control_horizon):
            x[:, k + 1] = self.sys.f(x[:, k], u[:, k])
        return x

    def _step(self, iteration, u):
        beta = self._beta(iteration)
        cov = beta * self.cov
        lamb = (beta * beta) * self.lamb
        sqrt_cov = math.sqrt(cov)

        z = self.standard_samples[iteration]
        eps = np.zeros(
            (self.n_samples, self.sys.m, self.K), dtype=np.float64
        )
        eps[:, :, :self.control_horizon] = z * sqrt_cov

        u_samples = u[None, :, :] + eps
        if self.input_clipping:
            clipped = np.clip(
                u_samples[:, :, :self.control_horizon],
                self.sys.u_min,
                self.sys.u_max,
            )
            u_samples[:, :, :self.control_horizon] = clipped
            eps[:, :, :self.control_horizon] = (
                clipped - u[None, :, :self.control_horizon]
            )

        x = np.empty(
            (self.n_samples, self.sys.n, self.K), dtype=np.float64
        )
        y = np.empty(
            (self.n_samples, self.sys.p, self.K), dtype=np.float64
        )
        total_cost = np.empty(self.n_samples, dtype=np.float64)
        lex_cost = np.empty(self.n_samples, dtype=np.float64)

        cov_inv = 1.0 / cov
        for n in range(self.n_samples):
            x_n = self._rollout(u_samples[n])
            x[n] = x_n
            y[n] = x_n
            correction = 0.0
            for k in range(self.control_horizon):
                correction += (
                    lamb
                    * float(u[:, k] @ np.array([[cov_inv]]) @ eps[n, :, k])
                )
            lex = self._evaluate(y[n])
            total_cost[n] = correction + lex
            lex_cost[n] = lex

        self.total_rollouts += self.n_samples
        self.evaluated_transitions += self.n_samples * self.control_horizon

        psi = float(np.min(total_cost))
        weights = np.exp(-(total_cost - psi) / lamb)
        weights /= np.sum(weights)

        u[:, :self.control_horizon] += np.einsum(
            "n,nmk->mk", weights, eps[:, :, :self.control_horizon]
        )

        best_idx = int(np.argmin(lex_cost))
        if lex_cost[best_idx] < self.best_overall_lex_cost:
            self.best_overall_lex_cost = float(lex_cost[best_idx])
            self.best_overall_u = u_samples[best_idx].copy()

        return u, y, best_idx

    def solve(self):
        u = np.zeros((self.sys.m, self.K), dtype=np.float64)
        t_start = time.perf_counter()
        for iteration in range(self.n_iterations):
            u, _, _ = self._step(iteration, u)

        solve_time = time.perf_counter() - t_start
        best_u = self.best_overall_u if self.best_overall_u is not None else u
        final_x = self._rollout(best_u)
        final_cost = self._evaluate(final_x)
        return {
            "final_cost": int(final_cost),
            "best_cost": int(self.best_overall_lex_cost),
            "solve_time_sec": float(solve_time),
            "total_rollouts": self.total_rollouts,
            "evaluated_transitions": self.evaluated_transitions,
            "pruned_rollouts": 0,
        }
