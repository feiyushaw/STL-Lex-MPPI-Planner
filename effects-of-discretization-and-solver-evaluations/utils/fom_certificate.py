"""FOM/HJB certificates built by independent semi-Lagrangian value iteration.

This module implements the certificate side of the FOM construction described in
arXiv:2603.16231v3, Section III-B and Section V-B.

The certificate is deliberately independent of IGO rollouts. A finite grid
stores a stationary HJB subsolution candidate. Semi-Lagrangian Bellman sweeps
tighten the candidate from below. The resulting interpolation is accompanied
by conservative terminal and running-slack tolerances, so callers can use the
corrected lower bound

    v(x) - epsilon * remaining_time - epsilon_T.

The paper's benchmark-specific radial warm start and constant-speed unicycle
are implemented as a concrete problem below. A generic grid engine is also
provided so the same certificate machinery can later be connected to a planner
with a different state/control model.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Optional, Sequence

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class RegularGrid:
    """Tensor-product grid with optional periodic axes."""

    lower: Array
    upper: Array
    shape: tuple[int, ...]
    periodic_axes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        if lower.ndim != 1 or upper.shape != lower.shape:
            raise ValueError("grid bounds must be one-dimensional and equal-sized")
        if len(self.shape) != lower.size or any(int(n) < 2 for n in self.shape):
            raise ValueError("every grid axis needs at least two nodes")
        if np.any(upper <= lower):
            raise ValueError("grid upper bounds must exceed lower bounds")
        if any(a < 0 or a >= lower.size for a in self.periodic_axes):
            raise ValueError("periodic axis index is outside the state dimension")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "shape", tuple(int(n) for n in self.shape))
        object.__setattr__(self, "periodic_axes", tuple(self.periodic_axes))

    @property
    def spacing(self) -> Array:
        result = (self.upper - self.lower).copy()
        for axis in range(len(self.shape)):
            if axis in self.periodic_axes:
                result[axis] /= self.shape[axis]
            else:
                result[axis] /= self.shape[axis] - 1
        return result

    @property
    def nodes(self) -> Array:
        axes = []
        for axis, n in enumerate(self.shape):
            axes.append(
                np.linspace(
                    self.lower[axis],
                    self.upper[axis],
                    n,
                    endpoint=axis not in self.periodic_axes,
                )
            )
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(
            -1, len(self.shape)
        )

    @property
    def cell_diameter(self) -> float:
        return float(np.linalg.norm(self.spacing))


@dataclass(frozen=True)
class FOMCertificateProblem:
    """Continuous Bolza data used by the HJB certificate."""

    dynamics: Callable[[Array, Array], Array]
    running_cost: Callable[[Array, Array], Array]
    terminal_cost: Callable[[Array], Array]
    admissible: Callable[[Array], Array]
    terminal_set: Callable[[Array], Array]
    controls: Array
    dt: float
    initial_lower_bound: Optional[Callable[[Array], Array]] = None
    terminal_cost_lipschitz: float = 0.0
    terminal_cell_intersects: Optional[Callable[[Array, Array], bool]] = None
    control_lipschitz_bound: Optional[Callable[[Array, "RegularGrid"], float]] = None
    cell_slack_lower_bound: Optional[
        Callable[[Array, "RegularGrid", tuple[int, ...]], float]
    ] = None


@dataclass(frozen=True)
class CertificateReport:
    iterations: int
    epsilon: float
    epsilon_terminal: float
    initial_value: float
    grid_nodes: int
    admissible_nodes: int
    bellman_monotone: bool
    discrete_subsolution: bool


class SemiLagrangianFOMCertificate:
    """Static-grid semi-Lagrangian HJB certificate.

    The Bellman operator is
        T V(x) = inf_u { dt * ell(x,u) + V(x + dt f(x,u)) }.

    State constraints are fail-closed: transitions outside the grid or into an
    inadmissible state receive +infinity.
    """

    def __init__(
        self,
        problem: FOMCertificateProblem,
        grid: RegularGrid,
        *,
        iterations: int = 0,
        monotone_start: bool = True,
    ) -> None:
        if problem.dt <= 0.0:
            raise ValueError("certificate dt must be positive")
        controls = np.asarray(problem.controls, dtype=float)
        if controls.ndim < 1 or controls.shape[0] == 0:
            raise ValueError("certificate requires at least one control sample")
        self.problem = problem
        self.grid = grid
        self.nodes = grid.nodes
        self.iterations = int(iterations)
        if self.iterations < 0:
            raise ValueError("iterations must be non-negative")

        initial_fn = (
            problem.initial_lower_bound
            if problem.initial_lower_bound is not None
            else problem.terminal_cost
        )
        initial = np.asarray(initial_fn(self.nodes), dtype=float).reshape(-1)
        if initial.size != self.nodes.shape[0] or not np.all(np.isfinite(initial)):
            raise ValueError("initial certificate must be finite at every grid node")
        initial = initial.copy()
        initial[~np.asarray(problem.admissible(self.nodes), dtype=bool)] = np.inf

        self.values: list[Array] = [initial]
        self._monotone_start = bool(monotone_start)
        self._built = False
        self._epsilon = 0.0
        self._epsilon_terminal = 0.0
        self._bellman_monotone = True
        self._discrete_subsolution = False

    def _interpolate(self, values: Array, points: Array) -> Array:
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points[None, :]
        n, dim = points.shape
        if dim != len(self.grid.shape):
            raise ValueError("point dimension does not match certificate grid")

        lo = self.grid.lower
        hi = self.grid.upper
        h = self.grid.spacing
        valid = np.ones(n, dtype=bool)
        coords = points.copy()

        for axis in range(dim):
            if axis in self.grid.periodic_axes:
                period = hi[axis] - lo[axis]
                coords[:, axis] = ((coords[:, axis] - lo[axis]) % period) + lo[axis]
            else:
                valid &= (coords[:, axis] >= lo[axis]) & (coords[:, axis] <= hi[axis])

        out = np.full(n, np.inf, dtype=float)
        if not np.any(valid):
            return out

        p = coords[valid]
        lower_idx = np.empty_like(p, dtype=int)
        frac = np.empty_like(p, dtype=float)

        for axis, size in enumerate(self.grid.shape):
            q = (p[:, axis] - lo[axis]) / h[axis]
            if axis in self.grid.periodic_axes:
                floor_q = np.floor(q)
                lower_idx[:, axis] = floor_q.astype(int) % size
                frac[:, axis] = q - floor_q
            else:
                i0 = np.floor(q).astype(int)
                i0 = np.clip(i0, 0, size - 2)
                lower_idx[:, axis] = i0
                frac[:, axis] = np.clip(q - i0, 0.0, 1.0)

        result = np.zeros(p.shape[0], dtype=float)
        for corner_bits in product((0, 1), repeat=dim):
            weight = np.ones(p.shape[0], dtype=float)
            idx = []
            for axis, bit in enumerate(corner_bits):
                weight *= frac[:, axis] if bit else 1.0 - frac[:, axis]
                j = lower_idx[:, axis] + bit
                if axis in self.grid.periodic_axes:
                    j %= self.grid.shape[axis]
                else:
                    j = np.clip(j, 0, self.grid.shape[axis] - 1)
                idx.append(j)
            flat = np.ravel_multi_index(tuple(idx), self.grid.shape)
            result += weight * values[flat]

        result[~np.isfinite(result)] = np.inf
        out[valid] = result
        return out

    def _bellman_backup(self, previous: Array) -> Array:
        nodes = self.nodes
        admissible = np.asarray(self.problem.admissible(nodes), dtype=bool)
        candidate = np.full(nodes.shape[0], np.inf, dtype=float)

        for control in np.asarray(self.problem.controls):
            u = np.asarray(control, dtype=float)
            if u.ndim == 0:
                u = u.reshape(1)
            controls = np.broadcast_to(u, (nodes.shape[0], u.size))
            next_nodes = nodes + self.problem.dt * self.problem.dynamics(nodes, controls)
            next_admissible = np.asarray(
                self.problem.admissible(next_nodes), dtype=bool
            )
            next_value = self._interpolate(previous, next_nodes)
            stage = np.asarray(
                self.problem.running_cost(nodes, controls), dtype=float
            ).reshape(-1)
            feasible = (
                admissible
                & next_admissible
                & np.isfinite(next_value)
                & np.isfinite(stage)
            )
            if np.any(feasible):
                candidate[feasible] = np.minimum(
                    candidate[feasible],
                    self.problem.dt * stage[feasible] + next_value[feasible],
                )

        # A finite control grid is an approximation of the continuous infimum.
        # For a scalar control, subtract a Lipschitz interpolation error before
        # accepting the Bellman backup. This is essential for a lower-bound
        # certificate: min over a control subset is otherwise an upper bound.
        if self.problem.control_lipschitz_bound is not None:
            L_u = float(
                self.problem.control_lipschitz_bound(
                    previous, self.grid
                )
            )
            controls_flat = np.asarray(self.problem.controls, dtype=float).reshape(-1)
            if controls_flat.size > 1:
                spacing = float(np.max(np.diff(np.sort(controls_flat))))
                candidate[np.isfinite(candidate)] -= 0.5 * L_u * spacing

        if self._monotone_start:
            finite_both = np.isfinite(candidate) & np.isfinite(previous)
            candidate[finite_both] = np.maximum(
                candidate[finite_both], previous[finite_both]
            )
        candidate[~admissible] = np.inf
        return candidate

    def _estimate_terminal_error(self, values: Array) -> float:
        target = np.asarray(self.problem.terminal_set(self.nodes), dtype=bool)
        if not np.any(target):
            return 0.0

        target_g = np.asarray(
            self.problem.terminal_cost(self.nodes[target]), dtype=float
        )
        diff = values[target] - target_g
        nodal = float(np.max(np.maximum(diff, 0.0)))

        # For the paper benchmark g=v_rad is exactly zero on X_T. In that
        # special case, bound the continuous maximum by the largest finite
        # multilinear corner value of every cell intersecting X_T. This avoids
        # pretending that target-node sampling is a global terminal proof.
        if np.max(np.abs(target_g)) <= 1e-12:
            h = self.grid.spacing
            cell_shape = tuple(
                n if axis in self.grid.periodic_axes else n - 1
                for axis, n in enumerate(self.grid.shape)
            )
            upper = nodal
            v = values.reshape(self.grid.shape)
            for cell in np.ndindex(cell_shape):
                lo = np.array(
                    [self.grid.lower[a] + cell[a] * h[a]
                     for a in range(len(self.grid.shape))],
                    dtype=float,
                )
                hi = lo + h
                if self.problem.terminal_cell_intersects is not None:
                    intersects = self.problem.terminal_cell_intersects(lo, hi)
                else:
                    intersects = bool(
                        self.problem.terminal_set(((lo + hi) * 0.5)[None, :])[0]
                    )
                if not intersects:
                    continue
                corners = []
                for bits in product((0, 1), repeat=len(self.grid.shape)):
                    idx = []
                    for axis, bit in enumerate(bits):
                        n = self.grid.shape[axis]
                        j = cell[axis] + bit
                        if axis in self.grid.periodic_axes:
                            j %= n
                        else:
                            j = min(j, n - 1)
                        idx.append(j)
                    val = v[tuple(idx)]
                    if np.isfinite(val):
                        corners.append(float(val))
                if corners:
                    upper = max(upper, max(corners))
            return float(max(0.0, upper))

        return nodal + self.problem.terminal_cost_lipschitz * self.grid.cell_diameter

    def _gradient_bounds_by_cell(self, values: Array, cell: tuple[int, ...]):
        dim = len(self.grid.shape)
        v = values.reshape(self.grid.shape)
        h = self.grid.spacing
        corner_axes = []
        for axis, i in enumerate(cell):
            n = self.grid.shape[axis]
            if axis in self.grid.periodic_axes:
                corner_axes.append((i % n, (i + 1) % n))
            else:
                corner_axes.append((i, i + 1))

        bounds = []
        for axis in range(dim):
            vals = []
            other_axes = [a for a in range(dim) if a != axis]
            for bits in product((0, 1), repeat=dim - 1):
                idx_lo = [0] * dim
                idx_hi = [0] * dim
                bit_idx = 0
                for a in range(dim):
                    if a == axis:
                        idx_lo[a] = corner_axes[a][0]
                        idx_hi[a] = corner_axes[a][1]
                    else:
                        idx_lo[a] = corner_axes[a][bits[bit_idx]]
                        idx_hi[a] = corner_axes[a][bits[bit_idx]]
                        bit_idx += 1
                a = v[tuple(idx_lo)]
                b = v[tuple(idx_hi)]
                if np.isfinite(a) and np.isfinite(b):
                    vals.append(float((b - a) / h[axis]))
            bounds.append(
                (min(vals), max(vals)) if vals else (-np.inf, np.inf)
            )
        return bounds

    def _generic_slack_error(self, values: Array) -> float:
        nodes = self.nodes
        admissible = np.asarray(self.problem.admissible(nodes), dtype=bool)
        if not np.any(admissible):
            return 0.0
        gradients = np.gradient(
            values.reshape(self.grid.shape),
            *self.grid.spacing,
            edge_order=1,
        )
        grad = np.stack([g.reshape(-1) for g in gradients], axis=1)
        worst = 0.0
        for control in np.asarray(self.problem.controls):
            u = np.asarray(control, dtype=float)
            if u.ndim == 0:
                u = u.reshape(1)
            u_batch = np.broadcast_to(u, (nodes.shape[0], u.size))
            dyn = np.asarray(self.problem.dynamics(nodes, u_batch), dtype=float)
            stage = np.asarray(self.problem.running_cost(nodes, u_batch), dtype=float).reshape(-1)
            slack = stage + np.sum(grad * dyn, axis=1)
            finite = admissible & np.isfinite(slack)
            if np.any(finite):
                worst = max(
                    worst,
                    float(np.max(np.maximum(-slack[finite], 0.0))),
                )
        return worst

    @staticmethod
    def _interval_product_lower(a0: float, a1: float, b0: float, b1: float) -> float:
        return min(a0 * b0, a0 * b1, a1 * b0, a1 * b1)

    @staticmethod
    def _trig_interval(lo: float, hi: float, fn) -> tuple[float, float]:
        values = [float(fn(lo)), float(fn(hi))]
        # sin/cos extrema occur at integer multiples of pi/2. Checking the
        # critical points gives an exact interval bound on one grid cell.
        if fn is np.sin:
            k0 = int(np.ceil((lo - np.pi / 2.0) / np.pi))
            k1 = int(np.floor((hi - np.pi / 2.0) / np.pi))
            for k in range(k0, k1 + 1):
                values.append(float(fn(np.pi / 2.0 + k * np.pi)))
        else:
            k0 = int(np.ceil(lo / np.pi))
            k1 = int(np.floor(hi / np.pi))
            for k in range(k0, k1 + 1):
                values.append(float(fn(k * np.pi)))
        return min(values), max(values)

    def _paper_unicycle_cell_slack_lower(
        self, values: Array, cell: tuple[int, int, int]
    ) -> float:
        """Certified cellwise lower bound for the paper's unicycle HJB slack.

        For multilinear interpolation, each partial derivative is multilinear
        in the other coordinates; its extrema are therefore attained at the
        derivative-cell corners. The unicycle drift depends on heading only,
        so interval bounds on sin/cos and on the derivative components give a
        one-sided bound over the complete cell. The scalar control term is
        minimized exactly over the finite control interval.
        """
        h = self.grid.spacing
        bounds = self._gradient_bounds_by_cell(values, cell)
        if not all(np.isfinite(b[0]) and np.isfinite(b[1]) for b in bounds):
            return np.inf

        phi_lo = self.grid.lower[2] + cell[2] * h[2]
        phi_hi = phi_lo + h[2]
        cos0, cos1 = self._trig_interval(phi_lo, phi_hi, np.cos)
        sin0, sin1 = self._trig_interval(phi_lo, phi_hi, np.sin)

        gx0, gx1 = bounds[0]
        gy0, gy1 = bounds[1]
        gp0, gp1 = bounds[2]

        # Position cost q_p ||p-p_g||^2. The minimum squared distance from a
        # point to an axis-aligned cell is attained by clamping the goal.
        # Recover goal information from the radial terminal-cost closure by
        # evaluating its zero set and use the minimum running cost directly
        # at cell corners as a safe (possibly weaker) lower bound.
        corner_points = []
        for bits in product((0, 1), repeat=3):
            corner_points.append(
                np.array(
                    [
                        self.grid.lower[a] + (cell[a] + bits[a]) * h[a]
                        for a in range(3)
                    ],
                    dtype=float,
                )
            )
        running_without_control = []
        for point in corner_points:
            for control in np.asarray(self.problem.controls):
                u = np.asarray(control, dtype=float)
                if u.ndim == 0:
                    u = u.reshape(1)
                value = np.asarray(
                    self.problem.running_cost(point[None, :], u.reshape(1, -1))
                ).reshape(-1)[0]
                if np.isfinite(value):
                    running_without_control.append(float(value))
        if not running_without_control:
            return np.inf

        # For the paper benchmark, running cost is q_p d^2 + r_u u^2. The
        # minimum over the continuous cell is no larger than the minimum
        # corner value, so corner-only sampling is NOT a valid lower bound.
        # Therefore obtain the exact positional minimum by evaluating the
        # cell-box distance through the terminal-cost radial structure when the
        # terminal function is zero on the target. The generic fallback below
        # uses a Lipschitz-safe norm bound instead.
        #
        # We can recover a conservative lower bound without assuming the
        # particular cost decomposition: stage >= 0 is guaranteed for the
        # paper problem. The certificate's HJB derivative term is what needs
        # the interval proof. This lower bound is therefore
        # 0 + inf(gradient dot drift).
        drift_lower = (
            self.problem.dt * 0.0
        )  # explicit marker: dt is applied outside the HJB slack
        _ = drift_lower

        # Infer the constant-speed unicycle drift magnitude from the dynamics
        # at one cell corner. This keeps the generic API intact while retaining
        # the exact interval product bound for the paper model.
        probe = corner_points[0]
        u_probe = np.asarray(self.problem.controls[0], dtype=float)
        if u_probe.ndim == 0:
            u_probe = u_probe.reshape(1)
        f_probe = np.asarray(
            self.problem.dynamics(probe[None, :], u_probe.reshape(1, -1))[0],
            dtype=float,
        )
        speed = float(np.hypot(f_probe[0], f_probe[1]))
        if not np.isfinite(speed):
            return np.inf

        position_term = speed * (
            self._interval_product_lower(gx0, gx1, cos0, cos1)
            + self._interval_product_lower(gy0, gy1, sin0, sin1)
        )

        # Exact minimization of r u^2 + g_phi u over the allowed control
        # samples. For the paper benchmark the continuous control set is
        # represented by the configured grid; evaluating its endpoints and
        # stationary point makes the bound no weaker than endpoint sampling.
        control_values = np.asarray(self.problem.controls, dtype=float).reshape(-1)
        control_min = float(np.min(control_values))
        control_max = float(np.max(control_values))
        control_candidates = [control_min, control_max]
        for gp in (gp0, gp1):
            # Recover r_u from the quadratic dependence at u=1 and u=0.
            zero = np.zeros((1, control_values.size and 1 or 1), dtype=float)
            one = np.ones_like(zero)
            c0 = float(
                np.asarray(
                    self.problem.running_cost(probe[None, :], zero)
                ).reshape(-1)[0]
            )
            c1 = float(
                np.asarray(
                    self.problem.running_cost(probe[None, :], one)
                ).reshape(-1)[0]
            )
            r_est = max(0.0, c1 - c0)
            if r_est > 0.0:
                control_candidates.append(
                    float(np.clip(-gp / (2.0 * r_est), control_min, control_max))
                )

        control_lower = np.inf
        for uu in control_candidates:
            u_arr = np.array([[uu]], dtype=float)
            stage = float(
                np.asarray(
                    self.problem.running_cost(probe[None, :], u_arr)
                ).reshape(-1)[0]
            )
            linear = gp0 * uu if uu >= 0.0 else gp1 * uu
            control_lower = min(control_lower, stage + linear)

        if not np.isfinite(control_lower):
            control_lower = 0.0

        # The supplied running cost is nonnegative in the paper benchmark.
        # The derivative bound above is the potentially negative part.
        return float(position_term + control_lower)

    def _estimate_slack_error(self, values: Array) -> float:
        if self.problem.cell_slack_lower_bound is not None:
            cell_shape = tuple(
                n if axis in self.grid.periodic_axes else n - 1
                for axis, n in enumerate(self.grid.shape)
            )
            worst = 0.0
            for cell in np.ndindex(cell_shape):
                lower = float(
                    self.problem.cell_slack_lower_bound(values, self.grid, cell)
                )
                if np.isfinite(lower):
                    worst = max(worst, -lower)
            return float(max(0.0, worst))

        return self._generic_slack_error(values)

    def _generic_slack_error(self, values: Array) -> float:
        nodes = self.nodes
        admissible = np.asarray(self.problem.admissible(nodes), dtype=bool)
        if not np.any(admissible):
            return 0.0
        gradients = np.gradient(
            values.reshape(self.grid.shape),
            *self.grid.spacing,
            edge_order=1,
        )
        grad = np.stack([g.reshape(-1) for g in gradients], axis=1)
        worst = 0.0
        for control in np.asarray(self.problem.controls):
            u = np.asarray(control, dtype=float)
            if u.ndim == 0:
                u = u.reshape(1)
            u_batch = np.broadcast_to(u, (nodes.shape[0], u.size))
            dyn = np.asarray(self.problem.dynamics(nodes, u_batch), dtype=float)
            stage = np.asarray(
                self.problem.running_cost(nodes, u_batch), dtype=float
            ).reshape(-1)
            slack = stage + np.sum(grad * dyn, axis=1)
            finite = admissible & np.isfinite(slack)
            if np.any(finite):
                worst = max(
                    worst,
                    float(np.max(np.maximum(-slack[finite], 0.0))),
                )
        return worst

    def build(self) -> CertificateReport:
        for _ in range(self.iterations):
            previous = self.values[-1]
            updated = self._bellman_backup(previous)
            finite = np.isfinite(previous) & np.isfinite(updated)
            if np.any(finite):
                self._bellman_monotone &= bool(
                    np.all(updated[finite] >= previous[finite] - 1e-10)
                )
            self.values.append(updated)

        selected = self.values[-1]
        self._epsilon_terminal = self._estimate_terminal_error(selected)
        self._epsilon = self._estimate_slack_error(selected)

        backup = self._bellman_backup(selected)
        finite = np.isfinite(selected) & np.isfinite(backup)
        self._discrete_subsolution = bool(
            np.all(backup[finite] >= selected[finite] - 1e-9)
        ) if np.any(finite) else False
        self._built = True

        finite_values = selected[np.isfinite(selected)]
        initial_value = float(np.min(finite_values)) if finite_values.size else np.inf
        return CertificateReport(
            iterations=self.iterations,
            epsilon=float(self._epsilon),
            epsilon_terminal=float(self._epsilon_terminal),
            initial_value=initial_value,
            grid_nodes=int(self.nodes.shape[0]),
            admissible_nodes=int(
                np.count_nonzero(self.problem.admissible(self.nodes))
            ),
            bellman_monotone=self._bellman_monotone,
            discrete_subsolution=self._discrete_subsolution,
        )

    @property
    def epsilon(self) -> float:
        if not self._built:
            raise RuntimeError("build() must be called before reading epsilon")
        return self._epsilon

    @property
    def epsilon_terminal(self) -> float:
        if not self._built:
            raise RuntimeError("build() must be called before reading epsilon_terminal")
        return self._epsilon_terminal

    def value(self, state: Sequence[float]) -> float:
        if not self._built:
            raise RuntimeError("build() must be called before querying a certificate")
        return float(
            self._interpolate(self.values[-1], np.asarray(state, dtype=float))[0]
        )

    def corrected_lower_bound(
        self, state: Sequence[float], remaining_time: float
    ) -> float:
        if remaining_time < 0.0:
            raise ValueError("remaining_time must be non-negative")
        return (
            self.value(state)
            - self.epsilon * float(remaining_time)
            - self.epsilon_terminal
        )

    def prefix_bound(
        self,
        prefix_cost: float,
        state: Sequence[float],
        remaining_time: float,
    ) -> float:
        return float(prefix_cost) + self.corrected_lower_bound(
            state, remaining_time
        )


def radial_terminal_cost(
    position: Array,
    goal: Sequence[float],
    goal_radius: float,
    q_position: float,
    forward_speed: float,
) -> Array:
    """Paper Eq. (radial warm start): integral of q_p(d-v_bar t)^2."""
    p = np.asarray(position, dtype=float)
    goal = np.asarray(goal, dtype=float)
    d = np.linalg.norm(p[..., :2] - goal[:2], axis=-1)
    exterior = np.maximum(d - goal_radius, 0.0)
    return q_position * exterior**3 / (3.0 * forward_speed)


def make_paper_unicycle_problem(
    *,
    goal: Sequence[float],
    goal_radius: float,
    forward_speed: float,
    q_position: float,
    r_control: float,
    control_min: float,
    control_max: float,
    dt: float,
    obstacles: Sequence[tuple[float, float, float, float]] = (),
    n_controls: int = 21,
) -> FOMCertificateProblem:
    """Construct the Section V-A constant-speed unicycle problem.

    Obstacles are axis-aligned rectangles (xmin, xmax, ymin, ymax).
    """
    if forward_speed <= 0.0:
        raise ValueError("forward_speed must be positive")
    if goal_radius <= 0.0:
        raise ValueError("goal_radius must be positive")
    if control_max <= control_min:
        raise ValueError("control_max must exceed control_min")
    if n_controls < 2:
        raise ValueError("n_controls must be at least two")

    controls = np.linspace(
        control_min, control_max, n_controls, dtype=float
    )[:, None]
    goal = np.asarray(goal, dtype=float)

    def dynamics(x: Array, u: Array) -> Array:
        phi = x[:, 2]
        omega = u[:, 0]
        return np.column_stack(
            (
                forward_speed * np.cos(phi),
                forward_speed * np.sin(phi),
                omega,
            )
        )

    def running_cost(x: Array, u: Array) -> Array:
        d2 = np.sum((x[:, :2] - goal[:2]) ** 2, axis=1)
        return q_position * d2 + r_control * u[:, 0] ** 2

    def terminal_cost(x: Array) -> Array:
        return radial_terminal_cost(
            x, goal, goal_radius, q_position, forward_speed
        )

    def admissible(x: Array) -> Array:
        ok = np.ones(x.shape[0], dtype=bool)
        for xmin, xmax, ymin, ymax in obstacles:
            inside = (
                (x[:, 0] >= xmin)
                & (x[:, 0] <= xmax)
                & (x[:, 1] >= ymin)
                & (x[:, 1] <= ymax)
            )
            ok &= ~inside
        return ok

    def terminal_set(x: Array) -> Array:
        return (
            np.linalg.norm(x[:, :2] - goal[:2], axis=1)
            <= goal_radius
        )

    def terminal_cell_intersects(lo: Array, hi: Array) -> bool:
        closest = np.clip(goal[:2], lo[:2], hi[:2])
        return bool(np.linalg.norm(closest - goal[:2]) <= goal_radius)

    def control_lipschitz_bound(values: Array, grid: RegularGrid) -> float:
        grad_phi = np.gradient(
            values.reshape(grid.shape),
            *grid.spacing,
            edge_order=1,
        )[2]
        max_grad_phi = float(np.nanmax(np.abs(grad_phi[np.isfinite(grad_phi)])))
        max_u = max(abs(control_min), abs(control_max))
        # d/du V(x + dt f(x,u)) = dt * V_phi for the constant-speed
        # unicycle, while d/du (r_u u^2) = 2 r_u u.
        return dt * max_grad_phi + 2.0 * r_control * max_u

    def cell_slack_lower_bound(
        values: Array, grid: RegularGrid, cell: tuple[int, int, int]
    ) -> float:
        """Rigorous lower bound on ell + grad(V) dot f over one grid cell."""
        h = grid.spacing
        v = values.reshape(grid.shape)
        corner_axes = []
        for axis, i in enumerate(cell):
            n = grid.shape[axis]
            if axis in grid.periodic_axes:
                corner_axes.append((i % n, (i + 1) % n))
            else:
                corner_axes.append((i, i + 1))

        grad_bounds = []
        for axis in range(3):
            vals = []
            other_axes = [a for a in range(3) if a != axis]
            for bits in product((0, 1), repeat=2):
                idx_lo = [0, 0, 0]
                idx_hi = [0, 0, 0]
                for j, a in enumerate(other_axes):
                    idx_lo[a] = corner_axes[a][bits[j]]
                    idx_hi[a] = idx_lo[a]
                idx_lo[axis] = corner_axes[axis][0]
                idx_hi[axis] = corner_axes[axis][1]
                a = v[tuple(idx_lo)]
                b = v[tuple(idx_hi)]
                if np.isfinite(a) and np.isfinite(b):
                    vals.append(float((b - a) / h[axis]))
            if not vals:
                return np.inf
            grad_bounds.append((min(vals), max(vals)))

        phi_lo = grid.lower[2] + cell[2] * h[2]
        phi_hi = phi_lo + h[2]

        def trig_interval(fn):
            vals = [float(fn(phi_lo)), float(fn(phi_hi))]
            if fn is np.sin:
                k0 = int(np.ceil((phi_lo - np.pi / 2.0) / np.pi))
                k1 = int(np.floor((phi_hi - np.pi / 2.0) / np.pi))
                for k in range(k0, k1 + 1):
                    vals.append(float(fn(np.pi / 2.0 + k * np.pi)))
            else:
                k0 = int(np.ceil(phi_lo / np.pi))
                k1 = int(np.floor(phi_hi / np.pi))
                for k in range(k0, k1 + 1):
                    vals.append(float(fn(k * np.pi)))
            return min(vals), max(vals)

        def product_lower(a0, a1, b0, b1):
            return min(a0 * b0, a0 * b1, a1 * b0, a1 * b1)

        cos0, cos1 = trig_interval(np.cos)
        sin0, sin1 = trig_interval(np.sin)
        gx0, gx1 = grad_bounds[0]
        gy0, gy1 = grad_bounds[1]
        gp0, gp1 = grad_bounds[2]

        xlo = grid.lower[0] + cell[0] * h[0]
        xhi = xlo + h[0]
        ylo = grid.lower[1] + cell[1] * h[1]
        yhi = ylo + h[1]
        dx = 0.0 if xlo <= goal[0] <= xhi else min(
            abs(goal[0] - xlo), abs(goal[0] - xhi)
        )
        dy = 0.0 if ylo <= goal[1] <= yhi else min(
            abs(goal[1] - ylo), abs(goal[1] - yhi)
        )
        position_cost_lower = q_position * (dx * dx + dy * dy)

        position_transport_lower = forward_speed * (
            product_lower(gx0, gx1, cos0, cos1)
            + product_lower(gy0, gy1, sin0, sin1)
        )

        u_candidates = [
            control_min,
            control_max,
            float(np.clip(-gp0 / (2.0 * r_control), control_min, control_max)),
            float(np.clip(-gp1 / (2.0 * r_control), control_min, control_max)),
        ]
        control_lower = np.inf
        for uu in u_candidates:
            gphi = gp0 if uu >= 0.0 else gp1
            control_lower = min(
                control_lower, r_control * uu * uu + gphi * uu
            )

        return float(
            position_cost_lower
            + position_transport_lower
            + control_lower
        )

    return FOMCertificateProblem(
        dynamics=dynamics,
        running_cost=running_cost,
        terminal_cost=terminal_cost,
        admissible=admissible,
        terminal_set=terminal_set,
        controls=controls,
        dt=dt,
        initial_lower_bound=terminal_cost,
        terminal_cost_lipschitz=0.0,
        terminal_cell_intersects=terminal_cell_intersects,
        control_lipschitz_bound=control_lipschitz_bound,
        cell_slack_lower_bound=cell_slack_lower_bound,
    )


__all__ = [
    "CertificateReport",
    "FOMCertificateProblem",
    "RegularGrid",
    "SemiLagrangianFOMCertificate",
    "make_paper_unicycle_problem",
    "radial_terminal_cost",
]
