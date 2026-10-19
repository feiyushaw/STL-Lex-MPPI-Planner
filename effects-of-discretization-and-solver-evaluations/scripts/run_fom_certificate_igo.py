#!/usr/bin/env python3
"""Run a paper-style FOM certificate + IGO smoke study.

The geometry and grid resolution are configurable. This script intentionally
does not claim to reproduce the paper's exact figure geometry; it exercises the
same certificate construction, capacity sweep, corrected bound, and IGO
integration before the later MPPI comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.fom_certificate import (  # noqa: E402
    RegularGrid,
    SemiLagrangianFOMCertificate,
    make_paper_unicycle_problem,
)
from utils.fom_certificate_igo import CertificateGuidedIGOSolver  # noqa: E402


def benchmark_prefix_query_batch(
    certificate,
    states: np.ndarray,
    prefix_costs: np.ndarray,
    remaining_times: np.ndarray,
    *,
    repeats: int = 3,
) -> tuple[float, float, float]:
    """Measure scalar-vs-batch query latency and verify numerical equivalence."""
    scalar_result = None
    batch_result = None
    scalar_times = []
    batch_times = []
    for _ in range(repeats):
        started = time.perf_counter()
        scalar_result = np.asarray(
            [
                certificate.prefix_bound(float(cost), state, float(remaining))
                for cost, state, remaining in zip(
                    prefix_costs, states, remaining_times
                )
            ],
            dtype=float,
        )
        scalar_times.append(time.perf_counter() - started)

        started = time.perf_counter()
        batch_result = certificate.prefix_bound_batch(
            prefix_costs, states, remaining_times
        )
        batch_times.append(time.perf_counter() - started)

    if not np.allclose(
        scalar_result, batch_result, rtol=1e-10, atol=1e-10
    ):
        raise RuntimeError("scalar and batched certificate bounds disagree")
    scalar_seconds = float(min(scalar_times))
    batch_seconds = float(min(batch_times))
    speedup = scalar_seconds / batch_seconds if batch_seconds > 0.0 else float("inf")
    return scalar_seconds, batch_seconds, speedup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--certificate-iterations",
        nargs="+",
        type=int,
        default=[0, 50, 300],
    )
    parser.add_argument(
        "--certificate-refinement-budget",
        type=int,
        default=256,
        help="maximum cells to refine for the continuous HJB slack bound",
    )
    parser.add_argument(
        "--certificate-deep-refinement-budget",
        type=int,
        default=8,
        help="maximum already-refined cells to subdivide again at 4x4x4 resolution",
    )
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--n-iterations", type=int, default=20)
    parser.add_argument(
        "--grid-shape",
        nargs=3,
        type=int,
        default=[21, 21, 12],
        metavar=("NX", "NY", "NPHI"),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--query-benchmark-samples",
        type=int,
        default=1000,
        help="number of states for scalar-vs-batch query timing; set 0 to disable",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "fom_certificate_igo",
    )
    args = parser.parse_args()
    if args.certificate_refinement_budget < 0:
        parser.error("--certificate-refinement-budget must be non-negative")
    if args.certificate_deep_refinement_budget < 0:
        parser.error("--certificate-deep-refinement-budget must be non-negative")

    args.output.mkdir(parents=True, exist_ok=True)

    problem = make_paper_unicycle_problem(
        goal=(0.0, 0.0),
        goal_radius=0.5,
        forward_speed=1.0,
        q_position=1.0,
        r_control=0.1,
        control_min=-1.0,
        control_max=1.0,
        dt=0.05,
        # Static obstacles, as in Section V-A.
        obstacles=(
            (-1.0, 1.0, -0.4, 0.4),
            (-4.0, -2.5, 1.0, 3.5),
        ),
        n_controls=11,
    )
    grid = RegularGrid(
        lower=np.array([-5.0, -5.0, -np.pi]),
        upper=np.array([5.0, 5.0, np.pi]),
        shape=tuple(args.grid_shape),
        periodic_axes=(2,),
    )

    if args.query_benchmark_samples < 0:
        parser.error("--query-benchmark-samples must be non-negative")
    query_states = query_prefix = query_remaining = None
    if args.query_benchmark_samples:
        query_rng = np.random.default_rng(args.seed + 10000)
        query_states = query_rng.uniform(
            low=grid.lower,
            high=grid.upper,
            size=(args.query_benchmark_samples, len(grid.shape)),
        )
        query_prefix = query_rng.uniform(
            0.0, 100.0, size=args.query_benchmark_samples
        )
        query_remaining = query_rng.uniform(
            0.0, 40 * problem.dt, size=args.query_benchmark_samples
        )

    rows = []
    for capacity in args.certificate_iterations:
        certificate = SemiLagrangianFOMCertificate(
            problem,
            grid,
            iterations=capacity,
            refinement_budget=args.certificate_refinement_budget,
            deep_refinement_budget=args.certificate_deep_refinement_budget,
        )
        certificate_build_started = time.perf_counter()
        report = certificate.build()
        certificate_build_seconds = time.perf_counter() - certificate_build_started
        if query_states is not None:
            (
                prefix_query_scalar_seconds,
                prefix_query_batch_seconds,
                prefix_query_speedup,
            ) = benchmark_prefix_query_batch(
                certificate,
                query_states,
                query_prefix,
                query_remaining,
            )
        else:
            prefix_query_scalar_seconds = None
            prefix_query_batch_seconds = None
            prefix_query_speedup = None
        worst_cell_neighborhood = (
            certificate.cell_slack_neighborhood(radius=1)
            if report.epsilon_worst_cell is not None
            else []
        )
        obstacle_extension_mode = (
            "nearest_admissible_residual"
            if certificate._domain_baseline_nodes is not None
            else "global_lower_bound_fail_closed"
        )
        blocked_mask = ~certificate._node_admissible
        obstacle_extension_residual_abs_max = 0.0
        if certificate._domain_baseline_nodes is not None and np.any(blocked_mask):
            extended_values = certificate._effective_extension_values(
                certificate.values[-1]
            ).reshape(grid.shape)
            extension_residual = (
                extended_values - certificate._domain_baseline_nodes.reshape(grid.shape)
            )
            obstacle_extension_residual_abs_max = float(
                np.max(np.abs(extension_residual[blocked_mask]))
            )

        solver = CertificateGuidedIGOSolver(
            certificate,
            x0=np.array([4.0, -3.0, 0.0]),
            horizon_steps=40,
            n_samples=args.n_samples,
            n_iterations=args.n_iterations,
            block_steps=5,
            elite_quantile=0.05,
            lr_mean=1.0,
            lr_cov=0.25,
            initial_std=0.5,
            random_seed=args.seed,
        )
        igo_solve_started = time.perf_counter()
        _, best_cost, reports = solver.solve()
        igo_solve_seconds = time.perf_counter() - igo_solve_started
        rows.append(
            {
                "certificate_iterations": capacity,
                "obstacle_extension_mode": obstacle_extension_mode,
                "obstacle_extension_blocked_nodes": int(np.count_nonzero(blocked_mask)),
                "obstacle_extension_residual_abs_max": obstacle_extension_residual_abs_max,
                "certificate_build_seconds": certificate_build_seconds,
                "igo_solve_seconds": igo_solve_seconds,
                "completed_rollouts": solver.evaluated_rollouts_total,
                "prefix_query_scalar_seconds": prefix_query_scalar_seconds,
                "prefix_query_batch_seconds": prefix_query_batch_seconds,
                "prefix_query_speedup": prefix_query_speedup,
                "epsilon": report.epsilon,
                "epsilon_coarse": float(certificate._epsilon_coarse),
                "epsilon_coarse_worst_cell": (
                    list(certificate._epsilon_coarse_worst_cell)
                    if certificate._epsilon_coarse_worst_cell is not None
                    else None
                ),
                "epsilon_refined_worst_cell": (
                    list(certificate._epsilon_refined_worst_cell)
                    if certificate._epsilon_refined_worst_cell is not None
                    else None
                ),
                "epsilon_refined_worst_violation": float(
                    certificate._epsilon_refined_worst_violation
                ),
                "epsilon_unrefined_worst_cell": (
                    list(certificate._epsilon_unrefined_worst_cell)
                    if certificate._epsilon_unrefined_worst_cell is not None
                    else None
                ),
                "epsilon_unrefined_worst_violation": float(
                    certificate._epsilon_unrefined_worst_violation
                ),
                "epsilon_refined_cells": int(certificate._epsilon_refined_cells),
                "epsilon_refinement_budget": int(certificate._epsilon_refinement_budget),
                "epsilon_refinement_budget_exhausted": bool(
                    certificate._epsilon_refinement_budget_exhausted
                ),
                "epsilon_deep_refined_cells": int(
                    len(certificate._epsilon_deep_refined_cell_set)
                ),
                "epsilon_deep_refinement_budget": int(
                    certificate._epsilon_deep_refinement_budget
                ),
                "epsilon_deep_refinement_budget_exhausted": bool(
                    certificate._epsilon_deep_refinement_budget_exhausted
                ),
                "epsilon_deep_refined_worst_cell": (
                    list(certificate._epsilon_deep_refined_worst_cell)
                    if certificate._epsilon_deep_refined_worst_cell is not None
                    else None
                ),
                "epsilon_deep_refined_worst_violation": float(
                    certificate._epsilon_deep_refined_worst_violation
                ),
                "epsilon_worst_cell_deep_refined_in_certificate": (
                    tuple(report.epsilon_worst_cell)
                    in certificate._epsilon_deep_refined_cell_set
                    if report.epsilon_worst_cell is not None else None
                ),
                "epsilon_terminal": report.epsilon_terminal,
                "epsilon_worst_cell": (
                    list(report.epsilon_worst_cell)
                    if report.epsilon_worst_cell is not None
                    else None
                ),
                "epsilon_worst_cell_coarse_lower_bound": (
                    certificate._epsilon_coarse_cell_lowers.get(
                        tuple(report.epsilon_worst_cell)
                    )
                    if report.epsilon_worst_cell is not None else None
                ),
                "epsilon_worst_cell_certified_lower_bound": (
                    certificate._epsilon_final_cell_lowers.get(
                        tuple(report.epsilon_worst_cell)
                    )
                    if report.epsilon_worst_cell is not None else None
                ),
                "epsilon_worst_cell_refined_in_certificate": (
                    tuple(report.epsilon_worst_cell)
                    in certificate._epsilon_refined_cell_set
                    if report.epsilon_worst_cell is not None else None
                ),
                "epsilon_worst_cell_bounds": (
                    {
                        "lower": (
                            grid.lower
                            + np.asarray(report.epsilon_worst_cell, dtype=float)
                            * grid.spacing
                        ).tolist(),
                        "upper": (
                            grid.lower
                            + (
                                np.asarray(report.epsilon_worst_cell, dtype=float)
                                + 1.0
                            )
                            * grid.spacing
                        ).tolist(),
                    }
                    if report.epsilon_worst_cell is not None
                    else None
                ),
                # Diagnostic-only neighborhood: this does not change the
                # certificate gate or omit any admissible cell.
                "epsilon_worst_cell_neighborhood": worst_cell_neighborhood,
                "grid_nodes": report.grid_nodes,
                "admissible_nodes": report.admissible_nodes,
                "bellman_monotone": report.bellman_monotone,
                "discrete_subsolution": report.discrete_subsolution,
                "initial_certificate_value": certificate.value(solver.x0),
                "initial_corrected_lower_bound": certificate.corrected_lower_bound(
                    solver.x0, solver.horizon_steps * problem.dt
                ),
                "best_cost": best_cost,
                "incumbent_cost": solver.incumbent_cost,
                "pruned_rollouts": solver.pruned_rollouts_total,
                "infeasible_rollouts": solver.infeasible_rollouts_total,
            }
        )

    output = args.output / "certificate_igo_summary.json"
    output.write_text(
        json.dumps(rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(rows, indent=2, sort_keys=True))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
