import os

import numpy as np
import pandas as pd
from interval import interval  # type: ignore
from joblib import Parallel, delayed  # type: ignore
from tabulate import tabulate  # type: ignore

from utils.fom_igo_solver import FOMIGOSolver
from utils.simulation_environment_1d import (
    CostFunction,
    System,
    calculate_rule_violations,
    compute_reachable_intervals,
)


def pregenerate_igo_latents(
    n_iterations,
    n_samples,
    K,
    u_dim,
    seed,
):
    """Pre-generate whitened IGO latent samples s ~ N(0,I)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_iterations, n_samples, u_dim, K))


def load_scenarios(filepath):
    """Load scenarios from a text file produced by save_scenarios."""
    exp_setups = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                exp_setups.append([float(x) for x in line.split()])
    print(f"Scenarios loaded from {filepath}")
    return exp_setups


def generate_cost_functions(shifts, n_rules):
    """Generate pointwise rule costs for the 1D IGO benchmark."""
    cost_functions = []
    for rule_idx in range(n_rules):
        shift = shifts[rule_idx]
        if rule_idx % 2 == 0:
            direction = "pos"
            lb, ub = shift, 10 + shift
        else:
            direction = "neg"
            lb, ub = -10 + shift, shift

        cost_functions.append(
            CostFunction(lb, ub, direction, time=rule_idx + 1)
        )
    return cost_functions


def check_violations(x_opt_disc, cost_functions, min_violations_threshold):
    """Check if the number of rule violations meets the minimum threshold."""
    violations = calculate_rule_violations(x_opt_disc, cost_functions)
    n_violated_rules = sum(1 for v in violations if v < 0)
    return n_violated_rules >= min_violations_threshold


def compute_discrete_lex_cost(x_opt_disc, cost_functions, x_0, K, system):
    """Evaluate the discrete reachable-set solution with the IGO Lex encoder."""
    solver = FOMIGOSolver(
        cost_functions=cost_functions,
        x0=x_0[0][0],
        K=K,
        system=system,
        n_iterations=0,
        verbose=False,
    )
    y_disc = np.array([[(iv[0][0] + iv[0][1]) / 2 for iv in x_opt_disc]])
    return int(solver.evaluate(y_disc))


def evaluate_scenario_igo(
    shifts,
    n_rules=8,
    combo=(5, 5, 5, 5, 5, 5, 5, 5),
    igo_num_iterations=30,
    igo_n_samples=500,
    igo_initial_covariance=0.5,
    igo_seed=42,
    igo_elite_quantile=0.05,
    igo_block_steps=5,
    igo_lr_mean=1.0,
    igo_lr_cov=0.25,
    min_violations_threshold=1,
    pregenerated_igo_samples=None,
):
    cost_functions = generate_cost_functions(shifts, n_rules)

    for j, c in enumerate(combo):
        cost_functions[j].update_nof_intervals(c)

    x_0 = interval[0]
    system = System(dt=1.0, u_min=-1.35, u_max=1.35)
    K = n_rules + 1

    x_opt_disc = compute_reachable_intervals(
        x_0, cost_functions, system, use_discrete_cost=True
    )

    if not check_violations(
        x_opt_disc, cost_functions, min_violations_threshold
    ):
        return None

    disc_lex_cost = compute_discrete_lex_cost(
        x_opt_disc, cost_functions, x_0, K, system
    )

    if pregenerated_igo_samples is not None:
        expected_shape = (
            igo_num_iterations,
            igo_n_samples,
            1,
            K,
        )
        if pregenerated_igo_samples.shape[0] < expected_shape[0] or \
           pregenerated_igo_samples.shape[1] < expected_shape[1]:
            raise ValueError("pregenerated IGO samples are too small")

    solver = FOMIGOSolver(
        cost_functions=cost_functions,
        x0=x_0[0][0],
        K=K,
        system=system,
        n_iterations=igo_num_iterations,
        n_samples_constant=igo_n_samples,
        n_samples_cosine=igo_n_samples,
        n_samples_cosine_min=igo_n_samples,
        cov=np.array([[igo_initial_covariance]]),
        lamb=1.0,  # compatibility only; ignored by IGO
        gamma=1.0,  # compatibility only; ignored by IGO
        beta_shrinking_method="exponential",  # compatibility only
        beta_min=1e-6,  # compatibility only
        samples_shrinking_method="constant",
        input_clipping=True,
        random_seed=igo_seed,
        verbose=False,
        pregenerated_eps=pregenerated_igo_samples,
        elite_quantile=igo_elite_quantile,
        block_steps=igo_block_steps,
        lr_mean=igo_lr_mean,
        lr_cov=igo_lr_cov,
        enable_certificate_pruning=True,
    )

    res = solver.solve()
    return {
        "disc_lex_cost": disc_lex_cost,
        "igo": int(res[2]),
        "igo_best": int(res[9]),
    }


def output_statistics(all_results, output_dir, max_results=None):
    valid_results = [res for res in all_results if res is not None]
    if max_results is not None:
        valid_results = valid_results[:max_results]

    if not valid_results:
        print("No valid results found to compute statistics.")
        return

    rows = []
    for key in ("igo", "igo_best"):
        gaps = []
        exact = 0
        worse = 0
        for res in valid_results:
            c = res.get(key)
            c_true = res.get("disc_lex_cost")
            if c is None or c_true is None or c_true <= 0:
                continue

            gap = ((float(c) - float(c_true)) / float(c_true)) * 100.0
            gaps.append(gap)
            if c == c_true:
                exact += 1
            elif c > c_true:
                worse += 1

        if not gaps:
            continue

        rows.append(
            [
                key,
                f"{100.0 * exact / len(gaps):.2f}%",
                f"{100.0 * worse / len(gaps):.2f}%",
                f"{np.mean(gaps):.3f}%",
                f"{np.median(gaps):.3f}%",
                f"{np.max(gaps):.3f}%",
            ]
        )

    headers = [
        "Method",
        "Exact reachable optimum (%)",
        "Above optimum (%)",
        "Mean relative gap (%)",
        "Median relative gap (%)",
        "Max relative gap (%)",
    ]
    table_str = tabulate(rows, headers=headers, stralign="right")

    print("\n--- IGO Solver Evaluation Statistics ---")
    print(table_str)

    os.makedirs(output_dir, exist_ok=True)
    table_path = os.path.join(output_dir, "igo_solver_statistics.txt")
    with open(table_path, "w") as f:
        f.write(
            f"Total Scenarios Evaluated: {len(valid_results)} "
            "(after threshold filtering)\n\n"
        )
        f.write("--- IGO Solver Evaluation Statistics ---\n")
        f.write(table_str)
        f.write("\n")
    print(f"\nSaved statistics table to {table_path}")


if __name__ == "__main__":
    run_new = True
    n_jobs = 10

    n_rules = 8
    combo = (7, 7, 7, 7, 7, 7, 7, 7)
    scenario_file = "scenarios/scenarios_n100000.txt"
    output_dir = "outputs/solver_evaluations"
    max_scenarios = 10000

    igo_n_samples = 400
    igo_num_iterations = 20
    igo_initial_covariance = 0.5
    igo_seed = 42
    igo_elite_quantile = 0.05
    igo_block_steps = 5
    igo_lr_mean = 1.0
    igo_lr_cov = 0.25

    min_violations_threshold = 3

    exp_setups = load_scenarios(scenario_file)
    if max_scenarios is not None:
        exp_setups = exp_setups[:max_scenarios]
    total_scenarios = len(exp_setups)
    print(f"Loaded {total_scenarios} scenarios for evaluation.")

    pregenerated_igo_samples = pregenerate_igo_latents(
        igo_num_iterations,
        igo_n_samples,
        n_rules + 1,
        1,
        igo_seed,
    )

    results_file = os.path.join(
        output_dir, f"igo_evaluation_results_n{total_scenarios}.csv"
    )

    if run_new:
        print("Starting parallel IGO experiments...")
        all_results = []

        for i, res in enumerate(
            Parallel(n_jobs=n_jobs)(
                delayed(evaluate_scenario_igo)(
                    shifts,
                    n_rules,
                    combo,
                    igo_num_iterations,
                    igo_n_samples,
                    igo_initial_covariance,
                    igo_seed,
                    igo_elite_quantile,
                    igo_block_steps,
                    igo_lr_mean,
                    igo_lr_cov,
                    min_violations_threshold=min_violations_threshold,
                    pregenerated_igo_samples=pregenerated_igo_samples,
                )
                for shifts in exp_setups
            ),
            1,
        ):
            all_results.append(res)
            if res is not None:
                print(f"Scenario {i}/{total_scenarios} evaluated successfully.")
            else:
                print(f"Scenario {i}/{total_scenarios} skipped.")

        valid_results = [r for r in all_results if r is not None]
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(valid_results).to_csv(results_file, index=False)
        print(f"Results saved to {results_file}")
    else:
        print(f"Loading results from {results_file}...")
        df = pd.read_csv(results_file)
        all_results = df.to_dict("records")

    output_statistics(all_results, output_dir, max_results=10000)
