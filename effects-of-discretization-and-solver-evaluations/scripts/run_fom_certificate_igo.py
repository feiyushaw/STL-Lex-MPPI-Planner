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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--certificate-iterations",
        nargs="+",
        type=int,
        default=[0, 50, 300],
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
        "--output",
        type=Path,
        default=ROOT / "outputs" / "fom_certificate_igo",
    )
    args = parser.parse_args()

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

    rows = []
    for capacity in args.certificate_iterations:
        certificate = SemiLagrangianFOMCertificate(
            problem,
            grid,
            iterations=capacity,
        )
        report = certificate.build()

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
        _, best_cost, reports = solver.solve()
        rows.append(
            {
                "certificate_iterations": capacity,
                "epsilon": report.epsilon,
                "epsilon_terminal": report.epsilon_terminal,
                "grid_nodes": report.grid_nodes,
                "admissible_nodes": report.admissible_nodes,
                "bellman_monotone": report.bellman_monotone,
                "discrete_subsolution": report.discrete_subsolution,
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
