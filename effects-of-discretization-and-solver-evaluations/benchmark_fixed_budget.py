"""Fixed-budget comparison of legacy MPPI, IGO, and IGO with pruning."""

import argparse
import os

import numpy as np
import pandas as pd
from interval import interval

from solver_evaluations import compute_discrete_lex_cost, generate_cost_functions, load_scenarios
from utils.fom_igo_solver import FOMIGOSolver
from utils.legacy_mppi_solver import LegacyMPPISolver
from utils.simulation_environment_1d import System, compute_reachable_intervals


def run_case(shifts, scenario_id, seed, args):
    costs = generate_cost_functions(shifts, args.n_rules)
    for cost in costs:
        cost.update_nof_intervals(args.interval_count)

    system = System(dt=1.0, u_min=-1.35, u_max=1.35)
    K = args.n_rules + 1
    discrete = compute_reachable_intervals(
        interval[0], costs, system, use_discrete_cost=True
    )
    optimum = compute_discrete_lex_cost(
        discrete, costs, interval[0], K, system
    )

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((args.iterations, args.samples, system.m, K - 1))

    mppi = LegacyMPPISolver(
        costs, 0.0, K, system, args.iterations, args.samples,
        args.initial_covariance, args.mppi_lambda, args.mppi_gamma,
        True, z
    )
    mppi_result = mppi.solve()

    rows = []
    for method, pruning in (("igo", False), ("igo_pruning", True)):
        solver = FOMIGOSolver(
            costs, 0.0, K, system,
            n_iterations=args.iterations,
            n_samples_constant=args.samples,
            n_samples_cosine=args.samples,
            n_samples_cosine_min=args.samples,
            cov=np.array([[args.initial_covariance]]),
            lamb=1.0, gamma=1.0,
            samples_shrinking_method="constant",
            input_clipping=True,
            random_seed=seed, verbose=False,
            pregenerated_eps=z,
            elite_quantile=args.elite_quantile,
            block_steps=args.block_steps,
            lr_mean=args.lr_mean,
            lr_cov=args.lr_cov,
            enable_certificate_pruning=pruning,
        )
        result = solver.solve()
        diag = solver.get_diagnostics()
        rows.append({
            "scenario_id": scenario_id, "seed": seed, "method": method,
            "discrete_optimum": int(optimum),
            "final_cost": int(result[2]), "best_cost": int(result[9]),
            "solve_time_sec": float(result[3]),
            "total_rollouts": diag["total_rollouts"],
            "pruned_rollouts": diag["pruned_rollouts"],
            "evaluated_transitions": diag["evaluated_transitions"],
            "rollout_effective_rate": diag["rollout_effective_rate"],
            "transition_effective_rate": diag["transition_effective_rate"],
            "optimal_match": int(result[9]) == int(optimum),
        })

    rows.append({
        "scenario_id": scenario_id, "seed": seed, "method": "legacy_mppi",
        "discrete_optimum": int(optimum),
        "final_cost": mppi_result["final_cost"],
        "best_cost": mppi_result["best_cost"],
        "solve_time_sec": mppi_result["solve_time_sec"],
        "total_rollouts": mppi_result["total_rollouts"],
        "pruned_rollouts": 0,
        "evaluated_transitions": mppi_result["evaluated_transitions"],
        "rollout_effective_rate": 1.0,
        "transition_effective_rate": 1.0,
        "optimal_match": mppi_result["best_cost"] == int(optimum),
    })
    return rows


def summarize(df):
    out = []
    for method, g in df.groupby("method", sort=False):
        opt = g["discrete_optimum"].to_numpy(float)
        best = g["best_cost"].to_numpy(float)
        gap = np.full(best.shape, np.nan)
        mask = opt > 0
        gap[mask] = 100.0 * (best[mask] - opt[mask]) / opt[mask]
        out.append({
            "method": method, "runs": len(g),
            "optimal_match_rate": g["optimal_match"].mean(),
            "zero_violation_rate": (g["best_cost"] == 0).mean(),
            "mean_relative_gap_pct": np.nanmean(gap),
            "median_relative_gap_pct": np.nanmedian(gap),
            "mean_solve_time_sec": g["solve_time_sec"].mean(),
            "mean_rollout_effective_rate": g["rollout_effective_rate"].mean(),
            "mean_transition_effective_rate": g["transition_effective_rate"].mean(),
        })
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario-file", default="scenarios/scenarios_n100000.txt")
    p.add_argument("--output-dir", default="outputs/fixed_budget_benchmark")
    p.add_argument("--scenarios", type=int, default=20)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--n-rules", type=int, default=8)
    p.add_argument("--interval-count", type=int, default=7)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--initial-covariance", type=float, default=0.5)
    p.add_argument("--mppi-lambda", type=float, default=1.0)
    p.add_argument("--mppi-gamma", type=float, default=0.6)
    p.add_argument("--elite-quantile", type=float, default=0.05)
    p.add_argument("--block-steps", type=int, default=5)
    p.add_argument("--lr-mean", type=float, default=1.0)
    p.add_argument("--lr-cov", type=float, default=0.25)
    args = p.parse_args()

    scenarios = load_scenarios(args.scenario_file)[:args.scenarios]
    rows = []
    for sid, shifts in enumerate(scenarios):
        for seed in range(args.seeds):
            rows.extend(run_case(shifts, sid, seed, args))

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    summary = summarize(df)
    df.to_csv(os.path.join(args.output_dir, "fixed_budget_results.csv"), index=False)
    summary.to_csv(os.path.join(args.output_dir, "fixed_budget_summary.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
