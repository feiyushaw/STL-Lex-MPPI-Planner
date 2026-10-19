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
    initial_lower_bound_is_subsolution: bool = False
    global_value_lower_bound: float = 0.0
    running_cost_lower_bound: float = 0.0
    terminal_cost_lipschitz: Optional[float] = None
    terminal_cell_intersects: Optional[Callable[[Array, Array], bool]] = None
    terminal_cell_lower_bound: Optional[Callable[[Array, Array], float]] = None
    cell_is_fully_inadmissible: Optional[Callable[[Array, Array], bool]] = None
    outside_domain_lower_bound: Optional[Callable[[Array], Array]] = None
    outside_domain_is_subsolution: bool = False
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

    State constraints are fail-closed. Transitions into inadmissible states
    are excluded. By default, outside-grid and invalid-cell queries use the
    declared global value lower bound. A problem may instead supply a global
    lower-bound extension that is itself a subsolution outside the grid; its
    values anchor a narrow boundary strip and provide a continuous extension.
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
        if not np.isfinite(problem.global_value_lower_bound):
            raise ValueError("global_value_lower_bound must be finite")
        if not np.isfinite(problem.running_cost_lower_bound):
            raise ValueError("running_cost_lower_bound must be finite")
        if problem.terminal_cost_lipschitz is not None and (
            not np.isfinite(problem.terminal_cost_lipschitz)
            or problem.terminal_cost_lipschitz < 0.0
        ):
            raise ValueError("terminal_cost_lipschitz must be finite and non-negative")

        if problem.outside_domain_lower_bound is not None and not (
            problem.outside_domain_is_subsolution
        ):
            raise ValueError(
                "outside_domain_lower_bound requires an explicit "
                "outside_domain_is_subsolution declaration"
            )

        self._node_admissible = np.asarray(
            problem.admissible(self.nodes), dtype=bool
        ).reshape(self.grid.shape)
        self._safe_zero_mask = self._make_safe_zero_mask()
        self._boundary_anchor_mask = self._make_boundary_anchor_mask()

        initial_fn = (
            problem.initial_lower_bound
            if problem.initial_lower_bound is not None
            else problem.terminal_cost
        )
        initial = np.asarray(initial_fn(self.nodes), dtype=float).reshape(-1)
        if initial.size != self.nodes.shape[0] or not np.all(np.isfinite(initial)):
            raise ValueError("initial certificate must be finite at every grid node")
        initial = initial.copy()
        initial[~self._node_admissible.reshape(-1)] = np.inf
        if self.iterations > 0:
            initial = self._apply_safe_domain(initial)

        self.values: list[Array] = [initial]
        self._monotone_start = bool(monotone_start)
        self._built = False
        self._epsilon = 0.0
        self._epsilon_terminal = 0.0
        self._bellman_monotone = True
        self._discrete_subsolution = False

    def _make_safe_zero_mask(self) -> Array:
        """Mask admissible nodes adjacent to blocked/unknown states.

        When no certified outside-domain extension exists, the outer grid
        boundary is also pinned to the global lower bound. With a certified
        extension, the boundary is anchored to that extension instead, avoiding
        an artificial jump from zero to a large interior value.
        """
        shape = self.grid.shape
        dim = len(shape)
        invalid = ~self._node_admissible
        halo = np.zeros(shape, dtype=bool)

        for index in np.ndindex(shape):
            if not invalid[index]:
                continue
            for offsets in product((-1, 0, 1), repeat=dim):
                neighbor = []
                inside = True
                for axis, offset in enumerate(offsets):
                    j = index[axis] + offset
                    if axis in self.grid.periodic_axes:
                        j %= shape[axis]
                    elif j < 0 or j >= shape[axis]:
                        inside = False
                        break
                    neighbor.append(j)
                if inside:
                    halo[tuple(neighbor)] = True

        if self.problem.outside_domain_lower_bound is None:
            for axis in range(dim):
                if axis in self.grid.periodic_axes:
                    continue
                boundary = [slice(None)] * dim
                boundary[axis] = 0
                halo[tuple(boundary)] = True
                boundary[axis] = -1
                halo[tuple(boundary)] = True

        return halo & self._node_admissible

    def _make_boundary_anchor_mask(self) -> Array:
        """Anchor the outer two node layers to the certified extension."""
        shape = self.grid.shape
        mask = np.zeros(shape, dtype=bool)
        if self.problem.outside_domain_lower_bound is None:
            return mask

        for axis, size in enumerate(shape):
            if axis in self.grid.periodic_axes:
                continue
            low = [slice(None)] * len(shape)
            low[axis] = slice(0, min(2, size))
            mask[tuple(low)] = True
            high = [slice(None)] * len(shape)
            high[axis] = slice(max(0, size - 2), size)
            mask[tuple(high)] = True
        return mask & self._node_admissible

    def _evaluate_outside_domain_lower_bound(self, points: Array) -> Array:
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points[None, :]
        if self.problem.outside_domain_lower_bound is None:
            return np.full(
                points.shape[0], self.problem.global_value_lower_bound, dtype=float
            )
        values = np.asarray(
            self.problem.outside_domain_lower_bound(points), dtype=float
        ).reshape(-1)
        if values.size != points.shape[0] or not np.all(np.isfinite(values)):
            raise ValueError(
                "outside_domain_lower_bound must return one finite value per point"
            )
        return values

    def _apply_safe_domain(self, values: Array) -> Array:
        result = np.asarray(values, dtype=float).reshape(-1).copy()
        if np.any(self._boundary_anchor_mask):
            anchor = self._boundary_anchor_mask.reshape(-1)
            result[anchor] = self._evaluate_outside_domain_lower_bound(
                self.nodes[anchor]
            )
        result[self._safe_zero_mask.reshape(-1)] = (
            self.problem.global_value_lower_bound
        )
        result[~self._node_admissible.reshape(-1)] = np.inf
        return result

    def _cell_corner_indices(self, cell: tuple[int, ...]):
        for bits in product((0, 1), repeat=len(self.grid.shape)):
            idx = []
            for axis, bit in enumerate(bits):
                j = cell[axis] + bit
                if axis in self.grid.periodic_axes:
                    j %= self.grid.shape[axis]
                idx.append(j)
            yield tuple(idx)

    def _cell_corner_values(self, values: Array, cell: tuple[int, ...]) -> Array:
        shaped = np.asarray(values, dtype=float).reshape(self.grid.shape)
        return np.asarray(
            [shaped[idx] for idx in self._cell_corner_indices(cell)],
            dtype=float,
        )

    def _cell_corner_points(self, cell: tuple[int, ...]) -> Array:
        h = self.grid.spacing
        return np.asarray(
            [
                [
                    self.grid.lower[axis] + (cell[axis] + bit) * h[axis]
                    for axis, bit in enumerate(bits)
                ]
                for bits in product((0, 1), repeat=len(self.grid.shape))
            ],
            dtype=float,
        )

    def _cell_bounds(self, cell: tuple[int, ...]) -> tuple[Array, Array]:
        lo = self.grid.lower + np.asarray(cell, dtype=float) * self.grid.spacing
        return lo, lo + self.grid.spacing

    def _interpolate(self, values: Array, points: Array) -> Array:
        """Interpolate without renormalizing away non-finite corners.

        Zero-weight non-finite corners are ignored to avoid 0 * inf. If any
        positive-weight corner is non-finite, the query falls back to the
        declared global value lower bound instead of dropping that corner and
        renormalizing the remaining weights.
        """
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points[None, :]
        n, dim = points.shape
        if dim != len(self.grid.shape):
            raise ValueError("point dimension does not match certificate grid")

        lo = self.grid.lower
        hi = self.grid.upper
        h = self.grid.spacing
        coords = points.copy()
        valid = np.all(np.isfinite(coords), axis=1)

        for axis in range(dim):
            if axis in self.grid.periodic_axes:
                period = hi[axis] - lo[axis]
                coords[:, axis] = ((coords[:, axis] - lo[axis]) % period) + lo[axis]
            else:
                valid &= (coords[:, axis] >= lo[axis]) & (coords[:, axis] <= hi[axis])

        out = np.full(n, self.problem.global_value_lower_bound, dtype=float)
        outside = np.all(np.isfinite(coords), axis=1) & ~valid
        if np.any(outside) and self.problem.outside_domain_lower_bound is not None:
            out[outside] = self._evaluate_outside_domain_lower_bound(
                coords[outside]
            )
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
        invalid_cell = np.zeros(p.shape[0], dtype=bool)
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
            corner_val = values[flat]
            positive = weight > 0.0
            invalid_cell |= positive & ~np.isfinite(corner_val)
            finite = positive & np.isfinite(corner_val)
            result[finite] += weight[finite] * corner_val[finite]

        result[invalid_cell] = self.problem.global_value_lower_bound
        out[valid] = result
        return out

    def _bellman_backup(
        self,
        previous: Array,
        *,
        enforce_monotone: bool = True,
        apply_control_correction: bool = True,
    ) -> Array:
        nodes = self.nodes
        admissible = self._node_admissible.reshape(-1)
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

        # A finite control grid approximates the continuous control infimum
        # from above. Subtract a Lipschitz covering error before using it as a
        # lower approximation. If the bound is unavailable, fail conservatively.
        if apply_control_correction and self.problem.control_lipschitz_bound is not None:
            finite_candidate = np.isfinite(candidate)
            if np.any(finite_candidate):
                controls_array = np.asarray(self.problem.controls, dtype=float)
                if controls_array.ndim == 1:
                    scalar_controls = controls_array.reshape(-1)
                elif controls_array.ndim == 2 and controls_array.shape[1] == 1:
                    scalar_controls = controls_array[:, 0]
                else:
                    scalar_controls = np.asarray([], dtype=float)

                unique_controls = np.unique(scalar_controls)
                if unique_controls.size < 2:
                    candidate[finite_candidate] = (
                        self.problem.global_value_lower_bound
                    )
                else:
                    L_u = float(
                        self.problem.control_lipschitz_bound(previous, self.grid)
                    )
                    if not np.isfinite(L_u) or L_u < 0.0:
                        candidate[finite_candidate] = (
                            self.problem.global_value_lower_bound
                        )
                    else:
                        spacing = float(np.max(np.diff(unique_controls)))
                        candidate[finite_candidate] -= 0.5 * L_u * spacing
                        candidate[finite_candidate] = np.maximum(
                            candidate[finite_candidate],
                            self.problem.global_value_lower_bound,
                        )

        if enforce_monotone and self._monotone_start:
            finite_both = np.isfinite(candidate) & np.isfinite(previous)
            candidate[finite_both] = np.maximum(
                candidate[finite_both], previous[finite_both]
            )
        candidate[~admissible] = np.inf
        return self._apply_safe_domain(candidate)

    def _has_exact_initial_subsolution(self) -> bool:
        same_terminal_function = (
            self.problem.initial_lower_bound is None
            or self.problem.initial_lower_bound is self.problem.terminal_cost
        )
        return bool(
            self.problem.initial_lower_bound_is_subsolution
            and same_terminal_function
        )

    def _analytic_initial_bellman_backup(self) -> Array:
        """Evaluate the K=0 analytic function at successor states directly."""
        initial_fn = (
            self.problem.initial_lower_bound
            if self.problem.initial_lower_bound is not None
            else self.problem.terminal_cost
        )
        nodes = self.nodes
        admissible = self._node_admissible.reshape(-1)
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
            next_value = np.asarray(initial_fn(next_nodes), dtype=float).reshape(-1)
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
        return candidate

    def _terminal_cell_cost_lower_bound(
        self, lo: Array, hi: Array, cell: tuple[int, ...]
    ) -> float:
        if self.problem.terminal_cell_lower_bound is not None:
            return float(self.problem.terminal_cell_lower_bound(lo, hi))
        lipschitz = self.problem.terminal_cost_lipschitz
        if lipschitz is None or not np.isfinite(lipschitz):
            return -np.inf
        corner_costs = np.asarray(
            self.problem.terminal_cost(self._cell_corner_points(cell)),
            dtype=float,
        ).reshape(-1)
        if corner_costs.size == 0 or not np.all(np.isfinite(corner_costs)):
            return -np.inf
        return float(np.min(corner_costs) - lipschitz * self.grid.cell_diameter)

    def _estimate_terminal_error(self, values: Array) -> float:
        # The analytic radial warm start is both the terminal cost and a known
        # continuous HJB subsolution in the paper benchmark.
        if self.iterations == 0 and self._has_exact_initial_subsolution():
            return 0.0
        if self.problem.terminal_cell_intersects is None:
            return np.inf

        admissible = self._node_admissible.reshape(-1)
        in_target = np.asarray(self.problem.terminal_set(self.nodes), dtype=bool)
        target_nodes = admissible & in_target
        upper = 0.0
        if np.any(target_nodes):
            target_values = values[target_nodes]
            target_costs = np.asarray(
                self.problem.terminal_cost(self.nodes[target_nodes]), dtype=float
            ).reshape(-1)
            if (
                not np.all(np.isfinite(target_values))
                or not np.all(np.isfinite(target_costs))
            ):
                return np.inf
            upper = max(upper, float(np.max(target_values - target_costs)))

        cell_shape = tuple(
            n if axis in self.grid.periodic_axes else n - 1
            for axis, n in enumerate(self.grid.shape)
        )
        for cell in np.ndindex(cell_shape):
            lo, hi = self._cell_bounds(cell)
            if not self.problem.terminal_cell_intersects(lo, hi):
                continue

            corners = self._cell_corner_values(values, cell)
            if np.all(np.isfinite(corners)):
                value_upper = float(np.max(corners))
            else:
                finite_corners = corners[np.isfinite(corners)]
                if finite_corners.size == 0:
                    return np.inf
                if not np.allclose(
                    finite_corners,
                    self.problem.global_value_lower_bound,
                    rtol=0.0,
                    atol=1e-12,
                ):
                    return np.inf
                value_upper = float(self.problem.global_value_lower_bound)

            terminal_lower = self._terminal_cell_cost_lower_bound(lo, hi, cell)
            if not np.isfinite(terminal_lower):
                return np.inf
            upper = max(upper, value_upper - terminal_lower)

        return float(max(0.0, upper))

    def _estimate_slack_error(self, values: Array) -> float:
        if self.iterations == 0 and self._has_exact_initial_subsolution():
            return 0.0
        # Nodal finite differences are not a continuous HJB-slack certificate.
        # Without a cellwise one-sided bound, disable pruning for this problem.
        if self.problem.cell_slack_lower_bound is None:
            return np.inf

        cell_shape = tuple(
            n if axis in self.grid.periodic_axes else n - 1
            for axis, n in enumerate(self.grid.shape)
        )
        worst = 0.0
        for cell in np.ndindex(cell_shape):
            corners = self._cell_corner_values(values, cell)
            if not np.all(np.isfinite(corners)):
                finite_corners = corners[np.isfinite(corners)]
                if finite_corners.size == 0:
                    lo, hi = self._cell_bounds(cell)
                    if (
                        self.problem.cell_is_fully_inadmissible is not None
                        and self.problem.cell_is_fully_inadmissible(lo, hi)
                    ):
                        continue
                    return np.inf
                if not np.allclose(
                    finite_corners,
                    self.problem.global_value_lower_bound,
                    rtol=0.0,
                    atol=1e-12,
                ):
                    return np.inf
                # In this cell the interpolant is the global lower-bound
                # constant, so its slack is bounded by the declared inf ell.
                lower = float(self.problem.running_cost_lower_bound)
            else:
                lower = float(
                    self.problem.cell_slack_lower_bound(values, self.grid, cell)
                )
            if not np.isfinite(lower):
                return np.inf
            worst = max(worst, -lower)
        return float(max(0.0, worst))

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

        if self.iterations == 0 and self._has_exact_initial_subsolution():
            # Do not test a grid interpolation of the analytic warm start.
            # Evaluate the known continuous function at successor states.
            backup = self._analytic_initial_bellman_backup()
        else:
            backup = self._bellman_backup(
                selected,
                enforce_monotone=False,
                apply_control_correction=False,
            )
        finite = self._node_admissible.reshape(-1) & np.isfinite(selected)
        self._discrete_subsolution = bool(
            np.any(finite)
            and np.all(
                np.isfinite(backup[finite])
                & (backup[finite] >= selected[finite] - 1e-9)
            )
        )
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
        point = np.asarray(state, dtype=float).reshape(-1)
        if point.size != len(self.grid.shape):
            raise ValueError("state dimension does not match certificate grid")
        if not bool(self.problem.admissible(point[None, :])[0]):
            return np.inf
        if self.iterations == 0 and self._has_exact_initial_subsolution():
            initial_fn = (
                self.problem.initial_lower_bound
                if self.problem.initial_lower_bound is not None
                else self.problem.terminal_cost
            )
            value = np.asarray(initial_fn(point[None, :]), dtype=float).reshape(-1)
            return float(value[0]) if value.size else -np.inf
        return float(self._interpolate(self.values[-1], point)[0])

    def corrected_lower_bound(
        self, state: Sequence[float], remaining_time: float
    ) -> float:
        if remaining_time < 0.0:
            raise ValueError("remaining_time must be non-negative")
        value = self.value(state)
        # Non-finite means no usable certificate value, never an infinite
        # lower bound. Disable pruning for such a query.
        if not np.isfinite(value):
            return -np.inf
        if not np.isfinite(self.epsilon) or not np.isfinite(self.epsilon_terminal):
            return -np.inf
        return (
            value
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
    """Paper Eq. (24): integrate q_p d(t)^2 until reaching radius rho."""
    p = np.asarray(position, dtype=float)
    goal = np.asarray(goal, dtype=float)
    d = np.linalg.norm(p[..., :2] - goal[:2], axis=-1)
    exterior_cube = np.maximum(d**3 - goal_radius**3, 0.0)
    return q_position * exterior_cube / (3.0 * forward_speed)


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
    if q_position <= 0.0 or r_control <= 0.0:
        raise ValueError("q_position and r_control must be positive")
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
        if np.linalg.norm(closest - goal[:2]) > goal_radius:
            return False
        # Cells wholly contained in one obstacle contain no admissible terminal
        # states and do not participate in the terminal-slack maximum.
        for xmin, xmax, ymin, ymax in obstacles:
            if (
                lo[0] >= xmin and hi[0] <= xmax
                and lo[1] >= ymin and hi[1] <= ymax
            ):
                return False
        return True

    def terminal_cell_lower_bound(lo: Array, hi: Array) -> float:
        # The terminal cost is exactly zero on the target set.
        return 0.0

    def cell_is_fully_inadmissible(lo: Array, hi: Array) -> bool:
        # An entire tensor cell contained in one obstacle has no admissible
        # states, so the HJB slack constraint is not imposed on that cell.
        for xmin, xmax, ymin, ymax in obstacles:
            if (
                lo[0] >= xmin and hi[0] <= xmax
                and lo[1] >= ymin and hi[1] <= ymax
            ):
                return True
        return False

    def control_lipschitz_bound(values: Array, grid: RegularGrid) -> float:
        shaped = np.asarray(values, dtype=float).reshape(grid.shape)
        next_heading = np.roll(shaped, -1, axis=2)
        finite_edges = np.isfinite(shaped) & np.isfinite(next_heading)
        if not np.any(finite_edges):
            return np.inf
        edge_slopes = np.abs(next_heading - shaped) / grid.spacing[2]
        max_grad_phi = float(np.max(edge_slopes[finite_edges]))
        max_u = max(abs(control_min), abs(control_max))
        # d/du [dt*ell(x,u) + V(x + dt*f(x,u))]
        # is bounded by dt*(2*r_u*|u| + |dV/dphi|).
        return dt * (max_grad_phi + 2.0 * r_control * max_u)

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
            float(np.clip(0.0, control_min, control_max)),
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
        initial_lower_bound_is_subsolution=True,
        global_value_lower_bound=0.0,
        running_cost_lower_bound=0.0,
        terminal_cost_lipschitz=None,
        terminal_cell_intersects=terminal_cell_intersects,
        terminal_cell_lower_bound=terminal_cell_lower_bound,
        cell_is_fully_inadmissible=cell_is_fully_inadmissible,
        outside_domain_lower_bound=terminal_cost,
        outside_domain_is_subsolution=True,
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
