#!/usr/bin/env python3
"""Compare separable and analytic heading HJB enclosures across grid shapes.

The existing certificate runner remains the single source of benchmark logic.
This harness runs it with identical seeds and settings for each grid/mode,
then checks that enabling the heading enclosure does not worsen certificate
violation metrics. It does not claim that sampled tests prove continuous-domain
conservativeness; that claim rests on the analytic enclosure and gradient bounds.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_fom_certificate_igo.py"
NO_REGRESSION_METRICS = (
    "epsilon",
    "epsilon_coarse",
    "epsilon_unrefined_worst_violation",
    "epsilon_refined_worst_violation",
)


def parse_grid_shape(value: str) -> tuple[int, int, int]:
    try:
        parts = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid grid shape {value!r}; expected NX,NY,NPHI"
        ) from exc
    if len(parts) != 3 or any(part < 2 for part in parts):
        raise argparse.ArgumentTypeError(
            f"invalid grid shape {value!r}; expected three integers >= 2"
        )
    return parts


def run_configuration(args, shape, mode: str) -> list[dict]:
    run_dir = args.output / (
        f"grid_{shape[0]}x{shape[1]}x{shape[2]}"
    ) / mode
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(RUNNER),
        "--certificate-iterations",
        *[str(value) for value in args.certificate_iterations],
        "--certificate-refinement-budget",
        str(args.refinement_budget),
        "--certificate-deep-refinement-budget",
        str(args.deep_refinement_budget),
        "--certificate-deep-refinement-subdivisions",
        *[str(value) for value in args.deep_refinement_subdivisions],
        "--grid-shape",
        *[str(value) for value in shape],
        "--seed",
        str(args.seed),
        "--n-samples",
        str(args.n_samples),
        "--n-iterations",
        str(args.n_iterations),
        "--query-benchmark-samples",
        "0",
        "--output",
        str(run_dir),
    ]
    if mode == "heading":
        command.append("--certificate-heading-extremum-enclosure")

    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"runner failed for grid={shape}, mode={mode}, "
            f"exit={completed.returncode}\n"
            f"stdout tail:\n{completed.stdout[-4000:]}\n"
            f"stderr tail:\n{completed.stderr[-4000:]}"
        )

    summary_path = run_dir / "certificate_igo_summary.json"
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    return [
        {
            "grid_shape": list(shape),
            "mode": mode,
            "run_directory": str(run_dir),
            **row,
        }
        for row in rows
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-shapes",
        nargs="+",
        type=parse_grid_shape,
        default=[
            (15, 15, 8),
            (21, 21, 12),
            (27, 27, 16),
        ],
        metavar="NX,NY,NPHI",
    )
    parser.add_argument(
        "--certificate-iterations",
        nargs="+",
        type=int,
        default=[0, 50, 300],
    )
    parser.add_argument("--refinement-budget", type=int, default=512)
    parser.add_argument("--deep-refinement-budget", type=int, default=8)
    parser.add_argument(
        "--deep-refinement-subdivisions",
        nargs=3,
        type=int,
        default=[4, 4, 2],
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=32,
        help="small IGO smoke budget; not used to compare optimizer quality",
    )
    parser.add_argument("--n-iterations", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "heading_enclosure_robustness",
    )
    args = parser.parse_args()

    if args.refinement_budget < 0 or args.deep_refinement_budget < 0:
        parser.error("refinement budgets must be non-negative")
    if any(value < 1 for value in args.deep_refinement_subdivisions):
        parser.error("deep-refinement subdivisions must be positive")
    if any(value < 0 for value in args.certificate_iterations):
        parser.error("certificate iteration counts must be non-negative")
    if args.n_samples < 1 or args.n_iterations < 0:
        parser.error("n-samples must be positive and n-iterations non-negative")

    args.output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for shape in args.grid_shapes:
        for mode in ("separable", "heading"):
            print(f"Running grid={shape}, mode={mode}", flush=True)
            all_rows.extend(run_configuration(args, shape, mode))

    lookup = {
        (
            tuple(row["grid_shape"]),
            int(row["certificate_iterations"]),
            row["mode"],
        ): row
        for row in all_rows
    }
    comparisons = []
    regressions = []
    for shape in args.grid_shapes:
        for capacity in args.certificate_iterations:
            baseline = lookup.get((tuple(shape), int(capacity), "separable"))
            heading = lookup.get((tuple(shape), int(capacity), "heading"))
            if baseline is None or heading is None:
                regressions.append(
                    {
                        "grid_shape": list(shape),
                        "certificate_iterations": int(capacity),
                        "reason": "missing baseline or heading result",
                    }
                )
                continue

            deltas = {
                metric: float(baseline[metric] - heading[metric])
                for metric in NO_REGRESSION_METRICS
            }
            no_regression = all(
                deltas[metric] >= -1e-8
                for metric in NO_REGRESSION_METRICS
            )
            comparison = {
                "grid_shape": list(shape),
                "certificate_iterations": int(capacity),
                "baseline_epsilon": float(baseline["epsilon"]),
                "heading_epsilon": float(heading["epsilon"]),
                "epsilon_reduction": deltas["epsilon"],
                "epsilon_reduction_percent": (
                    100.0 * deltas["epsilon"] / float(baseline["epsilon"])
                    if float(baseline["epsilon"]) > 0.0
                    else None
                ),
                "baseline_coarse_epsilon": float(baseline["epsilon_coarse"]),
                "heading_coarse_epsilon": float(heading["epsilon_coarse"]),
                "unrefined_violation_reduction": deltas[
                    "epsilon_unrefined_worst_violation"
                ],
                "refined_violation_reduction": deltas[
                    "epsilon_refined_worst_violation"
                ],
                "baseline_worst_cell": baseline["epsilon_worst_cell"],
                "heading_worst_cell": heading["epsilon_worst_cell"],
                "baseline_refined_cells": baseline["epsilon_refined_cells"],
                "heading_refined_cells": heading["epsilon_refined_cells"],
                "baseline_refinement_budget_exhausted": baseline[
                    "epsilon_refinement_budget_exhausted"
                ],
                "heading_refinement_budget_exhausted": heading[
                    "epsilon_refinement_budget_exhausted"
                ],
                "baseline_build_seconds": baseline["certificate_build_seconds"],
                "heading_build_seconds": heading["certificate_build_seconds"],
                "baseline_pruned_rollouts": baseline["pruned_rollouts"],
                "heading_pruned_rollouts": heading["pruned_rollouts"],
                "no_regression": no_regression,
            }
            comparisons.append(comparison)
            if not no_regression:
                regressions.append(comparison)

    report = {
        "schema_version": 1,
        "seed": args.seed,
        "grid_shapes": [list(shape) for shape in args.grid_shapes],
        "certificate_iterations": args.certificate_iterations,
        "refinement_budget": args.refinement_budget,
        "deep_refinement_budget": args.deep_refinement_budget,
        "deep_refinement_subdivisions": args.deep_refinement_subdivisions,
        "n_samples": args.n_samples,
        "n_iterations": args.n_iterations,
        "comparison_policy": {
            "baseline": "separable enclosure only",
            "heading": "analytic heading enclosure enabled; centered joint enclosure disabled",
            "no_regression_metrics": list(NO_REGRESSION_METRICS),
            "tolerance": 1e-8,
        },
        "comparisons": comparisons,
        "regressions": regressions,
        "passed": not regressions,
        "raw_runs": all_rows,
    }
    output_path = args.output / "heading_enclosure_robustness_summary.json"
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(
        {
            "passed": report["passed"],
            "comparison_count": len(comparisons),
            "regression_count": len(regressions),
            "summary": str(output_path),
        },
        indent=2,
    ))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
