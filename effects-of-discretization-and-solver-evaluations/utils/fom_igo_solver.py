"""
FOM Certificate-Guided Block-Diagonal xNES (IGO) Solver for STL-Lex-MPPI-Planner.
Drop-in replacement for `utils/mppi_solver.py` based on:
Wei et al., "Featurized Occupation Measures for Structured Global Search in Numerical Optimal Control" (arXiv:2603.16231v3).
"""

import math
import time
import numpy as np
from scipy.linalg import expm


class FOMIGOSolver:
    """
    Implicit FOM with Block-Diagonal xNES natural gradient updates (Section V-B)
    and Corollary 1 + Proposition 4 cumulative lower-bound prefix pruning (Section III-B).
    """

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
        elite_quantile: float = 0.08,
        block_steps: int = 3,
        lr_mean: float = 1.0,
        lr_cov: float = 0.30,
        enable_certificate_pruning: bool = True,
    ):
        self.cost_functions = cost_functions
        self.x0 = x0
        self.K = K
        self.sys = system

        # Standard MPPI-compatible parameters
        self.n_iterations = n_iterations
        self.n_samples_constant = n_samples_constant
        self.n_samples_cosine = n_samples_cosine
        self.n_samples_cosine_min = n_samples_cosine_min
        self.cov = cov if getattr(cov, "size", 0) > 0 else np.array([[0.5]])
        self.lamb = lamb
        self.gamma = gamma
        self.beta_shrinking_method = beta_shrinking_method
        self.beta_min = beta_min
        self.samples_shrinking_method = samples_shrinking_method
        self.input_clipping = input_clipping
        self.random_seed = random_seed
        self.verbose = verbose
        self.pregenerated_eps = pregenerated_eps

        # FOM-IGO (Block-xNES + Certificate Pruning) hyperparameters
        self.elite_quantile = elite_quantile
        self.block_steps = block_steps
        self.lr_mean = lr_mean
        self.lr_cov = lr_cov
        self.enable_certificate_pruning = enable_certificate_pruning

        if self.random_seed is not None:
            np.random.seed(random_seed)

        self.best_overall_lex_cost = float("inf")
        self.best_overall_u = np.zeros((self.sys.m, self.K))

        # Precompute bit-shifted lexicographical weights
        self.power_sums = self.precompute_power_sums()

        # Partition control horizon K into block-diagonal slices
        self.u_dim = self.sys.m
        self.blocks = [
            (s, min(self.K, s + self.block_steps))
            for s in range(0, self.K, self.block_steps)
        ]

        # Pre-index cost functions by time step for step-wise prefix pruning
        self._rules_by_time = {}
        self._rule_bounds = []
        for r_idx, cf in enumerate(self.cost_functions):
            t_step = int(cf.time)
            self._rules_by_time.setdefault(t_step, []).append((r_idx, cf))
            iv_lows = np.array([iv[0][0] for iv in cf.intervals], dtype=np.float64)
            iv_highs = np.array([iv[0][1] for iv in cf.intervals], dtype=np.float64)
            self._rule_bounds.append((t_step, cf.direction, cf.lb, cf.ub, iv_lows, iv_highs, self.power_sums[r_idx]))

        self._pruned_steps_total = 0
        self._evaluated_steps_total = 0

        if self.verbose and self.n_iterations > 0:
            self.print_total_samples_planned()

    def print_total_samples_planned(self):
        total_planned_samples = sum(
            self.calculate_number_of_samples(i) for i in range(self.n_iterations)
        )
        print(f"Total planned samples (FOM-IGO): {total_planned_samples}")

    def precompute_power_sums(self):
        bits = [math.ceil(math.log2(c.nof_intervals + 1)) for c in self.cost_functions]
        return [1 << int(sum(bits[i + 1 :])) for i in range(len(bits))]

    def calculate_beta(self, iteration):
        if self.beta_shrinking_method == "exponential":
            return math.sqrt(self.gamma**iteration)
        elif self.beta_shrinking_method == "cosine":
            if self.n_iterations <= 1:
                return self.beta_min
            ratio = 0.5 * (1 - math.cos(math.pi * iteration / (self.n_iterations - 1)))
            return 1.0 - (1.0 - self.beta_min) * ratio
        else:
            raise ValueError(
                f"Unknown beta shrinking method: {self.beta_shrinking_method}"
            )

    def calculate_number_of_samples(self, iteration):
        if self.samples_shrinking_method == "constant":
            return self.n_samples_constant
        elif self.samples_shrinking_method == "cosine":
            if self.n_iterations <= 1:
                return self.n_samples_cosine
            ratio = 0.5 * (1 - math.cos(math.pi * iteration / (self.n_iterations - 1)))
            return round(
                self.n_samples_cosine
                - (self.n_samples_cosine - self.n_samples_cosine_min) * ratio
            )
        else:
            raise ValueError(
                f"Unknown samples shrinking method: {self.samples_shrinking_method}"
            )

    def _ init_block_factors(self):
        init_std = np.sqrt(np.diag(self.cov)) if self.cov.ndim == 2 else np.sqrt(self.cov)
        base_std = float(init_std[0]) if np.size(init_std) > 0 else 1.0
        A_blocks = []
        for s, e in self.blocks:
            dim_b = (e - s) * self.u_dim
            A_blocks.append(np.eye(dim_b, dtype=np.float64) * base_std)
        return A_blocks

    @staticmethod
    def _vectorized_interval_index(x_vals: np.ndarray, iv_lows: np.ndarray, iv_highs: np.ndarray) -> np.ndarray:
        """Vectorized evaluation of CostFunction.evaluate_trajectory on 1D array x_vals."""
        in_iv = (x_vals[:, None] >= iv_lows[None, :]) & (x_vals[:, None] <= iv_highs[None, :])
        return np.argmax(in_iv, axis=1).astype(np.float64)

    def _evaluate_future_certificate(self, k_step: int, x_curr: np.ndarray) -> np.ndarray:
        """
        Analytical HJB reachability subsolution certificate v_psi(k, x_k) (Corollary 1 & Proposition 4).
        Computes guaranteed minimum future lexicographic violation over reachable cone [x_k + dt*u_min, x_k + dt*u_max].
        """
        v_cert = np.zeros_like(x_curr, dtype=np.float64)
        dt = self.sys.dt
        u_min = self.sys.u_min
        u_max = self.sys.u_max

        for t_step, direction, lb, ub, iv_lows, iv_highs, p_weight in self._rule_bounds:
            if t_step <= k_step:
                continue
            rem_steps = t_step - k_step
            if direction == "pos":
                # Best reachable state to minimize positive violation is x_curr + rem_steps * dt * u_min
                x_best_reach = x_curr + rem_steps * dt * u_min
                viol_mask = x_best_reach > lb
                if np.any(viol_mask):
                    v_cert[viol_mask] += (
                        self._vectorized_interval_index(x_best_reach[viol_mask], iv_lows, iv_highs)
                        * p_weight
                    )
            else:
                # Best reachable state to minimize negative violation is x_curr + rem_steps * dt * u_max
                x_best_reach = x_curr + rem_steps * dt * u_max
                viol_mask = x_best_reach < ub
                if np.any(viol_mask):
                    v_cert[viol_mask] += (
                        self._vectorized_interval_index(x_best_reach[viol_mask], iv_lows, iv_highs)
                        * p_weight
                    )
        return v_cert

    def solve(self):
        record_y_opt, record_y, record_best_sample_idx, record_cost = [], [], [], []
        record_beta, record_n_samples = [], []

        u = np.zeros((self.sys.m, self.K), dtype=np.float64)
        A_blocks = self._init_block_factors()

        t_start = time.time()

        for i in range(self.n_iterations):
            if self.verbose:
                print(
                    f"Iteration: {i + 1}/{self.n_iterations} (FOM-IGO) ----------------------------"
                )

            beta = self.calculate_beta(i)
            n_samples_current = self.calculate_number_of_samples(i)
            cov = beta * self.cov
            lamb = beta**2 * self.lamb

            u, A_blocks, y_opt, y, best_sample_idx, cost = self.fom_igo_step(
                i, u, A_blocks, beta, cov, lamb, n_samples_current
            )

            record_n_samples.append(n_samples_current)
            record_beta.append(beta)
            record_y_opt.append(y_opt)
            record_y.append(y)
            record_best_sample_idx.append(best_sample_idx)
            record_cost.append(cost)

        solve_time = time.time() - t_start

        if self.best_overall_u is None:
            self.best_overall_u = u.copy()

        best_overall_x, _ = self.forward_rollout(self.best_overall_u)

        # In FOM-IGO, ensure final control sequence is at least as good as the best elite found
        final_u = u
        final_x, final_y = self.forward_rollout(final_u)
        cost = self.evaluate(final_y)
        if self.best_overall_lex_cost < cost:
            final_u = self.best_overall_u.copy()
            final_x, final_y = best_overall_x, best_overall_x
            cost = self.best_overall_lex_cost

        return (
            final_x,
            final_u,
            cost,
            solve_time,
            record_y_opt,
            record_y,
            record_best_sample_idx,
            record_cost,
            best_overall_x,
            self.best_overall_lex_cost,
            record_n_samples,
            record_beta,
        )

    def fom_igo_step(self, iteration_idx, u, A_blocks, beta, cov, lamb, n_samples):
        x_dim = self.sys.n
        y_dim = self.sys.p
        u_dim = self.sys.m

        # 1. Generate whitened standard normal innovations s ~ N(0, I)
        if self.pregenerated_eps is not None:
            scale = math.sqrt(max(beta * float(self.cov[0, 0]), 1e-12))
            s_samples = self.pregenerated_eps[iteration_idx, :n_samples, :, :] / scale
        else:
            half_n = n_samples // 2
            s_half = np.random.standard_normal((half_n, u_dim, self.K))
            s_samples = np.concatenate([s_half, -s_half], axis=0)
            if s_samples.shape[0] < n_samples:
                s_extra = np.random.standard_normal((n_samples - s_samples.shape[0], u_dim, self.K))
                s_samples = np.concatenate([s_samples, s_extra], axis=0)

        # 2. Map whitened samples through block-diagonal covariance factors A_blocks
        eps = np.empty_like(s_samples)
        for (s_idx, e_idx), A_b in zip(self.blocks, A_blocks):
            dim_b = (e_idx - s_idx) * u_dim
            s_slice = s_samples[:, :, s_idx:e_idx].reshape(n_samples, dim_b)
            eps_slice = s_slice @ A_b.T
            eps[:, :, s_idx:e_idx] = eps_slice.reshape(n_samples, u_dim, e_idx - s_idx)

        u_samples = u[None, :, :] + eps
        if self.input_clipping:
            u_samples = np.clip(u_samples, self.sys.u_min, self.sys.u_max)
            eps = u_samples - u[None, :, :]

        # 3. Step-wise rollout with Corollary 1 + Proposition 4 lower-bound certificate pruning
        x = np.zeros((n_samples, x_dim, self.K), dtype=np.float64)
        x[:, :, 0] = self.x0
        lex_cost = np.zeros(n_samples, dtype=np.float64)
        active = np.ones(n_samples, dtype=bool)

        dt = self.sys.dt
        incumbent_bound = self.best_overall_lex_cost

        for k in range(self.K - 1):
            if not np.any(active):
                break
            act_idx = np.flatnonzero(active)
            self._evaluated_steps_total += act_idx.size

            # Forward step on active samples: x[k+1] = x[k] + dt * u[k]
            x[act_idx, :, k + 1] = x[act_idx, :, k] + dt * u_samples[act_idx, :, k]
            k_next = k + 1

            # Accumulate prefix lexicographic cost at time step k_next
            if k_next in self._rules_by_time:
                x_curr = x[act_idx, 0, k_next]
                for r_idx, cf in self._rules_by_time[k_next]:
                    _, _, _, _, iv_lows, iv_highs, p_weight = self._rule_bounds[r_idx]
                    r_cost = self._vectorized_interval_index(x_curr, iv_lows, iv_highs)
                    lex_cost[act_idx] += r_cost * p_weight

            # Corollary 1 Prefix Pruning Check: B_psi(k_next) = J_pre(k_next) + v_psi(k_next, x) > incumbent_bound
            if self.enable_certificate_pruning and np.isfinite(incumbent_bound) and (k_next < self.K - 1):
                x_curr = x[act_idx, 0, k_next]
                v_cert = self._evaluate_future_certificate(k_next, x_curr)
                b_psi = lex_cost[act_idx] + v_cert

                # Keep enough elites unpruned while pruning trajectories that cannot beat incumbent + tolerance
                prune_mask = b_psi > incumbent_bound
                if np.count_nonzero(~prune_mask) >= max(4, int(self.elite_quantile * n_samples)):
                    pruned_global = act_idx[prune_mask]
                    if pruned_global.size > 0:
                        active[pruned_global] = False
                        lex_cost[pruned_global] = b_psi[prune_mask] + 1e-3
                        self._pruned_steps_total += pruned_global.size * (self.K - 1 - k_next)
                        # Complete state rollout passively for visualization compatibility if needed
                        for rem_k in range(k_next, self.K - 1):
                            x[pruned_global, :, rem_k + 1] = (
                                x[pruned_global, :, rem_k] + dt * u_samples[pruned_global, :, rem_k]
                            )

        y = x.copy()

        # Secondary control regularization cost for tie-breaking among equal lexicographic costs
        reg_cost = 1e-6 * np.sum(u_samples[:, :, :-1] ** 2, axis=(1, 2))
        total_cost = lex_cost + reg_cost

        # Update best overall trajectory
        best_lex_sample = int(np.argmin(total_cost))
        if lex_cost[best_lex_sample] < self.best_overall_lex_cost or (
            lex_cost[best_lex_sample] == self.best_overall_lex_cost
            and total_cost[best_lex_sample]
            < self.best_overall_lex_cost + 1e-6 * np.sum(self.best_overall_u[:, :-1] ** 2)
        ):
            self.best_overall_lex_cost = float(lex_cost[best_lex_sample])
            self.best_overall_u = u_samples[best_lex_sample].copy()

        # 4. IGO Quantile-based Elite Selection & Block-Diagonal xNES Natural Gradient Update
        n_elite = max(2, int(round(self.elite_quantile * n_samples)))
        elite_indices = np.argsort(total_cost, kind="stable")[:n_elite]
        w_elite = np.full(n_elite, 1.0 / n_elite, dtype=np.float64)

        for idx_b, (s_idx, e_idx) in enumerate(self.blocks):
            dim_b = (e_idx - s_idx) * u_dim
            A_b = A_blocks[idx_b]

            # Recover effective whitened coordinates of clipped elite controls
            eps_elites_b = eps[elite_indices, :, s_idx:e_idx].reshape(n_elite, dim_b)
            s_elites_b = np.linalg.solve(A_b, eps_elites_b.T).T

            grad_delta = np.sum(w_elite[:, None] * s_elites_b, axis=0)
            grad_M = (s_elites_b.T * w_elite) @ s_elites_b - np.eye(dim_b)

            # Natural gradient updates in local coordinates
            delta_m = self.lr_mean * (A_b @ grad_delta)
            u[:, s_idx:e_idx] += delta_m.reshape(u_dim, e_idx - s_idx)
            A_blocks[idx_b] = A_b @ expm(0.5 * self.lr_cov * grad_M)

        if self.input_clipping:
            u = np.clip(u, self.sys.u_min, self.sys.u_max)

        _, y_opt = self.forward_rollout(u)
        return u, A_blocks, y_opt, y, best_lex_sample, total_cost

    def forward_rollout(self, u):
        """Performs an integration step of the system dynamics."""
        x = np.empty((self.sys.n, self.K))
        x[:, 0] = self.x0

        f = self.sys.f
        for k in range(self.K - 1):
            x[:, k + 1] = f(x[:, k], u[:, k])

        return x, x

    def calculate_path_cost(self, x, y, w, u, lamb, cov_inv):
        """Calculate the path costs (kept for backward compatibility)."""
        mppi_cost = 0
        for k in range(self.K - 1):
            mppi_cost += lamb * np.sum(u[:, k] @ cov_inv @ w[:, k])
        lex_cost = self.evaluate(y)
        return mppi_cost + lex_cost, lex_cost

    def evaluate(self, y):
        rule_costs = []
        for cost_func in self.cost_functions:
            rule_costs.append(int(cost_func.evaluate_trajectory(y)))

        c_bar = 0
        for i in range(len(rule_costs)):
            c_bar += rule_costs[i] * self.power_sums[i]

        return c_bar


# Drop-in alias for seamless compatibility with existing evaluation scripts
MPPISolver = FOMIGOSolver
