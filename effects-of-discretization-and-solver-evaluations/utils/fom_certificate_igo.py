"""Certificate-guided implicit-FOM IGO search.

This is the primal side paired with SemiLagrangianFOMCertificate. It follows
the paper's Section V-B ordering:

1. build the certificate independently;
2. freeze the certificate during primal search;
3. sample u = m + A s;
4. roll out the dynamics and accumulate running cost;
5. apply the corrected certificate prefix bound against a feasible incumbent;
6. discard pruned prefixes;
7. rank completed rollouts and perform the xNES natural-gradient update.

The implementation is intentionally separate from the legacy Lex-STL evaluator:
the certificate and the primal must describe the same Bolza objective before
certificate pruning is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from .fom_certificate import SemiLagrangianFOMCertificate


@dataclass(frozen=True)
class IGOIterationReport:
    iteration: int
    best_cost: float
    incumbent_cost: float
    evaluated_rollouts: int
    pruned_rollouts: int
    infeasible_rollouts: int


class CertificateGuidedIGOSolver:
    """Implicit-FOM IGO/xNES solver for a Bolza certificate problem."""

    def __init__(
        self,
        certificate: SemiLagrangianFOMCertificate,
        x0,
        horizon_steps: int,
        n_samples: int = 1000,
        n_iterations: int = 20,
        block_steps: int = 5,
        elite_quantile: float = 0.05,
        lr_mean: float = 1.0,
        lr_cov: float = 0.25,
        initial_std: float = 0.5,
        random_seed: int | None = None,
        pruning_tolerance: float = 0.0,
    ) -> None:
        if not certificate._built:
            raise ValueError("certificate must be built before starting IGO")
        self.certificate = certificate
        self.problem = certificate.problem
        self.x0 = np.asarray(x0, dtype=float).reshape(-1)
        self.horizon_steps = int(horizon_steps)
        self.n_samples = int(n_samples)
        self.n_iterations = int(n_iterations)
        self.block_steps = int(block_steps)
        self.elite_quantile = float(elite_quantile)
        self.lr_mean = float(lr_mean)
        self.lr_cov = float(lr_cov)
        self.pruning_tolerance = float(pruning_tolerance)

        controls = np.asarray(self.problem.controls, dtype=float)
        if controls.ndim == 1:
            controls = controls[:, None]
        if controls.ndim != 2:
            raise ValueError("controls must be a [n_controls, u_dim] array")
        self.control_bounds = np.column_stack(
            (np.min(controls, axis=0), np.max(controls, axis=0))
        )
        self.u_dim = controls.shape[1]
        self.state_dim = self.x0.size
        if self.horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive")
        if self.n_samples <= 0 or self.n_iterations <= 0:
            raise ValueError("sample and iteration counts must be positive")
        if self.block_steps <= 0:
            raise ValueError("block_steps must be positive")
        if not (0.0 < self.elite_quantile <= 1.0):
            raise ValueError("elite_quantile must be in (0, 1]")
        if self.lr_mean < 0.0 or self.lr_cov < 0.0:
            raise ValueError("IGO learning rates must be non-negative")
        if initial_std <= 0.0:
            raise ValueError("initial_std must be positive")

        self.rng = np.random.default_rng(random_seed)
        self.mean = np.zeros((self.u_dim, self.horizon_steps), dtype=float)
        std = np.full(self.u_dim, float(initial_std), dtype=float)
        self.blocks = [
            (s, min(self.horizon_steps, s + self.block_steps))
            for s in range(0, self.horizon_steps, self.block_steps)
        ]
        self.factors = []
        for s, e in self.blocks:
            dim = (e - s) * self.u_dim
            A = np.zeros((dim, dim), dtype=float)
            for k in range(e - s):
                for d in range(self.u_dim):
                    A[k * self.u_dim + d, k * self.u_dim + d] = std[d]
            self.factors.append(A)

        self.incumbent_cost = np.inf
        self.incumbent_control = None
        self.reports: list[IGOIterationReport] = []
        self.pruned_rollouts_total = 0
        self.evaluated_rollouts_total = 0
        self.infeasible_rollouts_total = 0

    def _sample_latents(self) -> np.ndarray:
        s = self.rng.standard_normal(
            (self.n_samples, self.u_dim, self.horizon_steps)
        )
        half = self.n_samples // 2
        if half:
            s[half : 2 * half] = -s[:half]
        return s

    def _controls_from_latents(self, latents: np.ndarray) -> np.ndarray:
        u = np.broadcast_to(
            self.mean[None, :, :],
            (self.n_samples, self.u_dim, self.horizon_steps),
        ).copy()
        for (s, e), A in zip(self.blocks, self.factors):
            dim = (e - s) * self.u_dim
            latent = latents[:, :, s:e].reshape(self.n_samples, dim)
            perturb = latent @ A.T
            u[:, :, s:e] += perturb.reshape(
                self.n_samples, self.u_dim, e - s
            )

        # The paper uses an admissible control set. Projection is performed
        # only on the primal control, never on the whitened latent samples.
        u = np.clip(
            u,
            self.control_bounds[:, 0][None, :, None],
            self.control_bounds[:, 1][None, :, None],
        )
        return u

    def _rollout_batch(
        self, controls: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        states = np.zeros(
            (self.n_samples, self.state_dim, self.horizon_steps + 1),
            dtype=float,
        )
        states[:, :, 0] = self.x0
        prefix = np.zeros(self.n_samples, dtype=float)
        active = np.ones(self.n_samples, dtype=bool)
        pruned = np.zeros(self.n_samples, dtype=bool)
        infeasible = np.zeros(self.n_samples, dtype=bool)

        for k in range(self.horizon_steps):
            idx = np.flatnonzero(active)
            if idx.size == 0:
                break

            xk = states[idx, :, k]
            uk = controls[idx, :, k]
            dyn = self.problem.dynamics(xk, uk)
            states[idx, :, k + 1] = xk + self.problem.dt * dyn

            stage = np.asarray(
                self.problem.running_cost(xk, uk), dtype=float
            ).reshape(-1)
            prefix[idx] += self.problem.dt * stage

            next_x = states[idx, :, k + 1]
            admissible = np.asarray(
                self.problem.admissible(next_x), dtype=bool
            )
            bad = idx[~admissible]
            if bad.size:
                active[bad] = False
                infeasible[bad] = True

            idx = idx[admissible]
            if idx.size == 0:
                continue

            if np.isfinite(self.incumbent_cost) and k < self.horizon_steps - 1:
                remaining_time = self.problem.dt * (
                    self.horizon_steps - (k + 1)
                )
                # Query all active prefixes in one vectorized certificate pass.
                # The scalar path performs a separate interpolation for every
                # sample and horizon step, which is a major Python overhead at
                # the paper benchmark's 10,000-rollout budget.
                bounds = self.certificate.prefix_bound_batch(
                    prefix[idx],
                    states[idx, :, k + 1],
                    remaining_time,
                )
                prune_local = bounds > (
                    self.incumbent_cost + self.pruning_tolerance
                )
                prune_idx = idx[prune_local]
                active[prune_idx] = False
                pruned[prune_idx] = True

        # Only rollouts that remained active through the full horizon have
        # a complete Bolza cost. Pruned prefixes are not trajectories and must
        # never enter incumbent selection or the xNES elite population.
        final_idx = np.flatnonzero(active)
        total = np.full(self.n_samples, np.inf, dtype=float)
        if final_idx.size:
            terminal = np.asarray(
                self.problem.terminal_cost(
                    states[final_idx, :, self.horizon_steps]
                ),
                dtype=float,
            ).reshape(-1)
            total[final_idx] = prefix[final_idx] + terminal
        completed_count = int(final_idx.size)
        return (
            states,
            total,
            pruned,
            int(np.count_nonzero(infeasible)),
            completed_count,
        )

    def _update_distribution(
        self, latents: np.ndarray, total_cost: np.ndarray
    ) -> int:
        finite = np.flatnonzero(np.isfinite(total_cost))
        if finite.size == 0:
            return -1

        n_elite = max(
            1,
            min(
                finite.size,
                int(np.ceil(self.elite_quantile * self.n_samples)),
            ),
        )
        order = finite[
            np.argsort(total_cost[finite], kind="stable")[:n_elite]
        ]
        weights = np.full(n_elite, 1.0 / n_elite, dtype=float)

        for block_idx, (s, e) in enumerate(self.blocks):
            A = self.factors[block_idx]
            dim = (e - s) * self.u_dim
            sb = latents[order, :, s:e].reshape(n_elite, dim)

            mean_grad = np.sum(weights[:, None] * sb, axis=0)
            covariance_grad = (
                (sb.T * weights) @ sb - np.eye(dim)
            )
            self.mean[:, s:e] += (
                self.lr_mean * (A @ mean_grad)
            ).reshape(self.u_dim, e - s)

            covariance_grad = 0.5 * (
                covariance_grad + covariance_grad.T
            )
            new_A = A @ expm(
                0.5 * self.lr_cov * covariance_grad
            )
            if not np.all(np.isfinite(new_A)):
                raise FloatingPointError(
                    "IGO covariance update produced non-finite values"
                )
            self.factors[block_idx] = new_A

        self.mean = np.clip(
            self.mean,
            self.control_bounds[:, 0][:, None],
            self.control_bounds[:, 1][:, None],
        )
        return int(order[0])

    def solve(self):
        best_cost = np.inf
        best_control = None

        for iteration in range(self.n_iterations):
            latents = self._sample_latents()
            controls = self._controls_from_latents(latents)
            states, costs, pruned, infeasible_count, active_count = (
                self._rollout_batch(controls)
            )

            finite = np.flatnonzero(np.isfinite(costs))
            if finite.size:
                local = finite[np.argmin(costs[finite])]
                if costs[local] < best_cost:
                    best_cost = float(costs[local])
                    best_control = controls[local].copy()
                if costs[local] < self.incumbent_cost:
                    self.incumbent_cost = float(costs[local])
                    self.incumbent_control = controls[local].copy()

            self.pruned_rollouts_total += int(np.count_nonzero(pruned))
            self.evaluated_rollouts_total += int(active_count)
            self.infeasible_rollouts_total += int(infeasible_count)

            self._update_distribution(latents, costs)
            self.reports.append(
                IGOIterationReport(
                    iteration=iteration,
                    best_cost=best_cost,
                    incumbent_cost=self.incumbent_cost,
                    evaluated_rollouts=int(finite.size),
                    pruned_rollouts=int(np.count_nonzero(pruned)),
                    infeasible_rollouts=int(infeasible_count),
                )
            )

        return best_control, best_cost, self.reports


__all__ = [
    "CertificateGuidedIGOSolver",
    "IGOIterationReport",
]
