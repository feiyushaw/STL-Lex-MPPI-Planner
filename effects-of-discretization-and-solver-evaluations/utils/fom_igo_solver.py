"""
FOM-IGO solver for STL-Lex-MPPI-Planner.

The historical MPPISolver interface is retained for compatibility, but the
active optimizer is IGO/xNES:

    u = m + A s,  s ~ N(0,I)
    rank/quantile selection
    m <- m + eta_m A E_elite[s]
    A_b <- A_b exp(eta_A/2 E_elite[s_b s_b^T - I])

No MPPI exponential temperature weighting or MPPI perturbation correction is
used by the optimizer.
"""

import math
import time

import numpy as np
from scipy.linalg import expm


class FOMIGOSolver:
    """Implicit-FOM IGO solver with block-diagonal xNES search."""

    def __init__(
        self,
        cost_functions: list,
        x0,
        K,
        system,
        n_iterations: int = 10,
        n_samples_constant: int = 500,
        n_samples_cosine: int = 950,
        n_samples_cosine_min: int = 50,
        cov=np.array([]),
        lamb: float = 2.0,
        gamma: float = 0.6,
        beta_shrinking_method: str = "exponential",
        beta_min: float = 1e-6,
        samples_shrinking_method: str = "constant",
        input_clipping: bool = False,
        random_seed=None,
        verbose: bool = True,
        pregenerated_eps=None,
        elite_quantile: float = 0.05,
        block_steps: int = 5,
        lr_mean: float = 1.0,
        lr_cov: float = 0.25,
        enable_certificate_pruning: bool = True,
    ):
        self.cost_functions = cost_functions
        self.x0 = float(x0)
        self.K = int(K)
        self.sys = system

        # Legacy constructor parameters are accepted for compatibility only.
        self.n_iterations = int(n_iterations)
        self.n_samples_constant = int(n_samples_constant)
        self.n_samples_cosine = int(n_samples_cosine)
        self.n_samples_cosine_min = int(n_samples_cosine_min)
        self.cov = (
            np.asarray(cov, dtype=np.float64)
            if getattr(cov, "size", 0) > 0
            else np.array([[0.5]], dtype=np.float64)
        )
        self.lamb = lamb
        self.gamma = gamma
        self.beta_shrinking_method = beta_shrinking_method
        self.beta_min = beta_min
        self.samples_shrinking_method = samples_shrinking_method
        self.input_clipping = bool(input_clipping)
        self.random_seed = random_seed
        self.verbose = verbose
        self.pregenerated_eps = pregenerated_eps

        self.elite_quantile = float(elite_quantile)
        self.block_steps = int(block_steps)
        self.lr_mean = float(lr_mean)
        self.lr_cov = float(lr_cov)
        self.enable_certificate_pruning = bool(enable_certificate_pruning)

        self.u_dim = int(self.sys.m)
        self.x_dim = int(self.sys.n)
        self.y_dim = int(self.sys.p)
        self.control_horizon = self.K - 1

        if self.K <= 1:
            raise ValueError("K must be greater than one")
        if self.n_iterations < 0:
            raise ValueError("n_iterations must be non-negative")
        if self.n_samples_constant <= 0 or self.n_samples_cosine <= 0 or self.n_samples_cosine_min <= 0:
            raise ValueError("IGO sample counts must be positive")

        if not (0.0 < self.elite_quantile <= 1.0):
            raise ValueError("elite_quantile must be in (0, 1]")
        if self.block_steps <= 0:
            raise ValueError("block_steps must be positive")
        if self.lr_mean < 0.0 or self.lr_cov < 0.0:
            raise ValueError("IGO learning rates must be non-negative")
        if self.cov.ndim != 2 or self.cov.shape != (self.u_dim, self.u_dim):
            raise ValueError("covariance must be square and match the input dimension")
        if not np.allclose(self.cov, np.diag(np.diag(self.cov))):
            raise ValueError("IGO requires a diagonal initial covariance")
        if np.any(np.diag(self.cov) <= 0.0):
            raise ValueError("initial covariance must be positive")

        self._rng = (
            np.random.default_rng()
            if random_seed is None
            else np.random.default_rng(random_seed)
        )

        self.best_overall_lex_cost = float("inf")
        self.best_overall_u = None
        self.power_sums = self.precompute_power_sums()

        self.blocks = [
            (s, min(self.control_horizon, s + self.block_steps))
            for s in range(0, self.control_horizon, self.block_steps)
        ]

        self._rules_by_time = {}
        self._rule_bounds = []
        for r_idx, cf in enumerate(self.cost_functions):
            t_step = int(cf.time)
            self._rules_by_time.setdefault(t_step, []).append((r_idx, cf))
            iv_lows = np.array([iv[0][0] for iv in cf.intervals], dtype=np.float64)
            iv_highs = np.array([iv[0][1] for iv in cf.intervals], dtype=np.float64)
            self._rule_bounds.append(
                (t_step, cf.direction, cf.lb, cf.ub, iv_lows, iv_highs, self.power_sums[r_idx])
            )

        self._pruned_steps_total = 0
        self._pruned_rollouts_total = 0
        self._total_rollouts = 0
        self._evaluated_steps_total = 0
        self._validate_pregenerated_samples()

        if self.verbose and self.n_iterations > 0:
            self.print_total_samples_planned()

    def print_total_samples_planned(self):
        total = sum(
            self.calculate_number_of_samples(i)
            for i in range(self.n_iterations)
        )
        print(f"Total planned samples (FOM-IGO): {total}")

    def precompute_power_sums(self):
        bits = [math.ceil(math.log2(c.nof_intervals + 1)) for c in self.cost_functions]
        return [1 << int(sum(bits[i + 1:])) for i in range(len(bits))]

    def calculate_beta(self, iteration):
        # Compatibility-only API. IGO carries covariance in A and has no beta schedule.
        return 1.0

    def calculate_number_of_samples(self, iteration):
        if iteration < 0 or iteration >= self.n_iterations:
            raise ValueError("iteration is outside the configured IGO horizon")
        if self.samples_shrinking_method == "constant":
            return self.n_samples_constant
        if self.samples_shrinking_method == "cosine":
            if self.n_iterations <= 1:
                return self.n_samples_cosine
            ratio = 0.5 * (
                1.0 - math.cos(math.pi * iteration / (self.n_iterations - 1))
            )
            return round(
                self.n_samples_cosine
                - (self.n_samples_cosine - self.n_samples_cosine_min) * ratio
            )
        if self.samples_shrinking_method == "linear":
            if self.n_iterations <= 1:
                return self.n_samples_cosine
            ratio = iteration / (self.n_iterations - 1)
            return round(
                self.n_samples_cosine
                - (self.n_samples_cosine - self.n_samples_cosine_min) * ratio
            )
        raise ValueError(
            f"Unknown samples shrinking method: {self.samples_shrinking_method}"
        )

    def _init_block_factors(self):
        std = np.sqrt(np.diag(self.cov))
        A_blocks = []
        for s, e in self.blocks:
            dim_b = (e - s) * self.u_dim
            A_b = np.zeros((dim_b, dim_b), dtype=np.float64)
            for k in range(e - s):
                for d in range(self.u_dim):
                    A_b[k * self.u_dim + d, k * self.u_dim + d] = std[d]
            A_blocks.append(A_b)
        return A_blocks

    @staticmethod
    def _vectorized_interval_index(x_vals, iv_lows, iv_highs):
        x_vals = np.asarray(x_vals, dtype=np.float64)
        in_iv = (
            (x_vals[:, None] >= iv_lows[None, :])
            & (x_vals[:, None] <= iv_highs[None, :])
        )
        found = np.any(in_iv, axis=1)
        result = np.argmax(in_iv, axis=1).astype(np.float64)
        if np.any(~found):
            # Never turn an out-of-range state into the best (zero-violation) bin.
            result[~found] = float(len(iv_lows) - 1)
        return result

    def _evaluate_future_certificate(self, k_step, x_curr):
        """
        Conservative lower bound for the 1D pointwise-rule benchmark.

        For x[k+1] = x[k] + dt*u[k], u in [u_min,u_max], a positive rule is
        minimized at the lower reachable endpoint and a negative rule at the
        upper endpoint. The result is a lower bound for the additive rule
        penalties used by this synthetic evaluator.
        """
        x_curr = np.asarray(x_curr, dtype=np.float64)
        v_cert = np.zeros_like(x_curr)
        dt = float(self.sys.dt)

        for t_step, direction, lb, ub, iv_lows, iv_highs, p_weight in self._rule_bounds:
            if t_step <= k_step:
                continue
            rem_steps = t_step - k_step
            if direction == "pos":
                x_best = x_curr + rem_steps * dt * self.sys.u_min
                mask = x_best > lb
            else:
                x_best = x_curr + rem_steps * dt * self.sys.u_max
                mask = x_best < ub
            if np.any(mask):
                idx = self._vectorized_interval_index(
                    x_best[mask], iv_lows, iv_highs
                )
                v_cert[mask] += idx * p_weight
        return v_cert

    def _validate_pregenerated_samples(self):
        if self.pregenerated_eps is None:
            return
        arr = np.asarray(self.pregenerated_eps, dtype=np.float64)
        if arr.ndim != 4:
            raise ValueError(
                "pregenerated_eps must have shape [iteration, sample, u_dim, K-1]"
            )
        if arr.shape[2:] != (self.u_dim, self.control_horizon):
            raise ValueError(
                "pregenerated_eps shape does not match [*, u_dim, K-1]"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("pregenerated_eps contains non-finite values")
        self.pregenerated_eps = arr

    def _latent_samples(self, iteration_idx, n_samples):
        if self.pregenerated_eps is not None:
            if iteration_idx >= self.pregenerated_eps.shape[0]:
                raise ValueError(
                    "pregenerated_eps has fewer iterations than requested"
                )
            if self.pregenerated_eps.shape[1] < n_samples:
                raise ValueError(
                    "pregenerated_eps has fewer samples than requested"
                )
            return self.pregenerated_eps[iteration_idx, :n_samples].copy()

        s = self._rng.standard_normal((n_samples, self.u_dim, self.control_horizon))
        half = n_samples // 2
        if half > 0:
            s[:half] = self._rng.standard_normal((half, self.u_dim, self.control_horizon))
            s[half:2 * half] = -s[:half]
        return s

    def solve(self):
        record_y_opt = []
        record_y = []
        record_best_sample_idx = []
        record_cost = []
        record_beta = []
        record_n_samples = []

        u = np.zeros((self.u_dim, self.K), dtype=np.float64)
        A_blocks = self._init_block_factors()

        _, initial_y = self.forward_rollout(u)
        self.best_overall_u = u.copy()
        self.best_overall_lex_cost = float(self.evaluate(initial_y))

        t_start = time.time()

        for i in range(self.n_iterations):
            if self.verbose:
                print(
                    f"Iteration: {i + 1}/{self.n_iterations} "
                    f"(FOM-IGO) ----------------------------"
                )

            n_samples = self.calculate_number_of_samples(i)
            self._total_rollouts += n_samples
            (
                u,
                A_blocks,
                y_opt,
                y,
                best_idx,
                cost,
            ) = self.fom_igo_step(i, u, A_blocks, n_samples)

            record_n_samples.append(n_samples)
            record_beta.append(1.0)
            record_y_opt.append(y_opt)
            record_y.append(y)
            record_best_sample_idx.append(best_idx)
            record_cost.append(cost)

        solve_time = time.time() - t_start

        final_u = u
        final_x, final_y = self.forward_rollout(final_u)
        final_cost = self.evaluate(final_y)

        if (
            self.best_overall_u is not None
            and self.best_overall_lex_cost < final_cost
        ):
            final_u = self.best_overall_u.copy()
            final_x, final_y = self.forward_rollout(final_u)
            final_cost = self.evaluate(final_y)

        return (
            final_x,
            final_u,
            final_cost,
            solve_time,
            record_y_opt,
            record_y,
            record_best_sample_idx,
            record_cost,
            self.forward_rollout(self.best_overall_u)[0],
            self.best_overall_lex_cost,
            record_n_samples,
            record_beta,
        )

    def fom_igo_step(self, iteration_idx, u, A_blocks, n_samples):
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        s_samples = self._latent_samples(iteration_idx, n_samples)

        eps = np.zeros((n_samples, self.u_dim, self.K), dtype=np.float64)
        for (s_idx, e_idx), A_b in zip(self.blocks, A_blocks):
            dim_b = (e_idx - s_idx) * self.u_dim
            s_slice = s_samples[:, :, s_idx:e_idx].reshape(n_samples, dim_b)
            eps[:, :, s_idx:e_idx] = (
                s_slice @ A_b.T
            ).reshape(n_samples, self.u_dim, e_idx - s_idx)

        u_samples = u[None, :, :] + eps
        if self.input_clipping:
            u_samples = np.clip(
                u_samples, self.sys.u_min, self.sys.u_max
            )

        x = np.zeros((n_samples, self.x_dim, self.K), dtype=np.float64)
        x[:, :, 0] = self.x0
        lex_cost = np.zeros(n_samples, dtype=np.float64)
        active = np.ones(n_samples, dtype=bool)

        dt = float(self.sys.dt)
        incumbent = self.best_overall_lex_cost

        for k in range(self.K - 1):
            if not np.any(active):
                break

            act_idx = np.flatnonzero(active)
            self._evaluated_steps_total += int(act_idx.size)
            x[act_idx, :, k + 1] = (
                x[act_idx, :, k] + dt * u_samples[act_idx, :, k]
            )
            k_next = k + 1

            if k_next in self._rules_by_time:
                x_curr = x[act_idx, 0, k_next]
                for r_idx, _cf in self._rules_by_time[k_next]:
                    (
                        _,
                        _,
                        _,
                        _,
                        iv_lows,
                        iv_highs,
                        p_weight,
                    ) = self._rule_bounds[r_idx]
                    r_cost = self._vectorized_interval_index(
                        x_curr, iv_lows, iv_highs
                    )
                    lex_cost[act_idx] += r_cost * p_weight

            if (
                self.enable_certificate_pruning
                and np.isfinite(incumbent)
                and k_next < self.K - 1
            ):
                x_curr = x[act_idx, 0, k_next]
                lower_bound = (
                    lex_cost[act_idx]
                    + self._evaluate_future_certificate(k_next, x_curr)
                )
                prune_mask = lower_bound > incumbent
                keep_required = max(
                    1,
                    int(math.ceil(self.elite_quantile * n_samples)),
                )
                if np.count_nonzero(~prune_mask) >= keep_required:
                    pruned_global = act_idx[prune_mask]
                    if pruned_global.size:
                        active[pruned_global] = False
                        lex_cost[pruned_global] = lower_bound[prune_mask]
                        self._pruned_rollouts_total += int(pruned_global.size)
                        self._pruned_steps_total += int(
                            pruned_global.size * (self.K - 1 - k_next)
                        )
                        # Complete state rollout for visualization compatibility.
                        for rem_k in range(k_next, self.K - 1):
                            x[pruned_global, :, rem_k + 1] = (
                                x[pruned_global, :, rem_k]
                                + dt * u_samples[pruned_global, :, rem_k]
                            )

        y = x.copy()
        reg_cost = 1e-6 * np.sum(
            u_samples[:, :, :-1] ** 2, axis=(1, 2)
        )
        total_cost = lex_cost + reg_cost

        # Lex-STL key is primary; effort is only a deterministic tie-breaker.
        order = np.lexsort((total_cost, lex_cost))
        best_idx = int(order[0])

        best_candidate_lex = float(lex_cost[best_idx])
        if best_candidate_lex < self.best_overall_lex_cost:
            self.best_overall_lex_cost = best_candidate_lex
            self.best_overall_u = u_samples[best_idx].copy()

        n_elite = max(
            1,
            min(
                n_samples,
                int(math.ceil(self.elite_quantile * n_samples)),
            ),
        )
        elite_indices = order[:n_elite]
        w = np.full(n_elite, 1.0 / n_elite, dtype=np.float64)

        for block_idx, (s_idx, e_idx) in enumerate(self.blocks):
            A_b = A_blocks[block_idx]
            dim_b = (e_idx - s_idx) * self.u_dim
            s_elites = s_samples[
                elite_indices, :, s_idx:e_idx
            ].reshape(n_elite, dim_b)

            grad_delta = np.sum(w[:, None] * s_elites, axis=0)
            grad_M = (
                (s_elites.T * w) @ s_elites
                - np.eye(dim_b)
            )

            u[:, s_idx:e_idx] += (
                self.lr_mean * (A_b @ grad_delta)
            ).reshape(self.u_dim, e_idx - s_idx)

            symmetric_grad = 0.5 * (grad_M + grad_M.T)
            A_new = A_b @ expm(
                0.5 * self.lr_cov * symmetric_grad
            )
            if not np.all(np.isfinite(A_new)):
                raise FloatingPointError(
                    "xNES covariance update produced non-finite values"
                )
            A_blocks[block_idx] = A_new

        if self.input_clipping:
            u = np.clip(u, self.sys.u_min, self.sys.u_max)

        _, y_opt = self.forward_rollout(u)
        return u, A_blocks, y_opt, y, best_idx, total_cost

    def forward_rollout(self, u):
        x = np.empty((self.x_dim, self.K), dtype=np.float64)
        x[:, 0] = self.x0
        for k in range(self.K - 1):
            x[:, k + 1] = self.sys.f(x[:, k], u[:, k])
        return x, x

    def get_diagnostics(self):
        """Return deterministic rollout/pruning accounting for benchmark use."""
        total_possible_transitions = self._total_rollouts * self.control_horizon
        return {
            "total_rollouts": int(self._total_rollouts),
            "pruned_rollouts": int(self._pruned_rollouts_total),
            "evaluated_transitions": int(self._evaluated_steps_total),
            "pruned_transitions": int(self._pruned_steps_total),
            "rollout_effective_rate": (
                1.0 - self._pruned_rollouts_total / self._total_rollouts
                if self._total_rollouts
                else 1.0
            ),
            "transition_effective_rate": (
                self._evaluated_steps_total / total_possible_transitions
                if total_possible_transitions
                else 1.0
            ),
            "transition_reduction_rate": (
                1.0 - self._evaluated_steps_total / total_possible_transitions
                if total_possible_transitions
                else 0.0
            ),
        }

    def calculate_path_cost(self, x, y, w, u, lamb, cov_inv):
        # Legacy API. MPPI path-integral correction is intentionally ignored.
        lex_cost = self.evaluate(y)
        secondary = 1e-6 * np.sum(np.asarray(u)[:, :-1] ** 2)
        return lex_cost + secondary, lex_cost

    def evaluate(self, y):
        rule_costs = [
            int(cost_func.evaluate_trajectory(y))
            for cost_func in self.cost_functions
        ]
        return sum(
            rule_costs[i] * self.power_sums[i]
            for i in range(len(rule_costs))
        )


# Existing imports remain valid, but the backend is IGO/xNES.
MPPISolver = FOMIGOSolver
