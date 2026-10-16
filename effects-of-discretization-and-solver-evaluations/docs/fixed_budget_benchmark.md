# Fixed-budget optimizer replacement benchmark

## Purpose

The CommonRoad 70-step run establishes integration correctness but does not
establish that IGO is a better optimizer than the historical MPPI backend.
This benchmark answers that separate question under a fixed rollout budget.

Methods:

1. legacy_mppi: isolated reference implementation of the historical
   exponential-weighted MPPI update.
2. igo: production FOM-IGO/xNES update with certificate pruning disabled.
3. igo_pruning: the same IGO update with the repository's conservative
   operational-limit prefix pruning enabled.

The legacy implementation is benchmark-only and is not exposed through the
production planner.

## Fairness contract

For each scenario and seed, all three methods use:

- the same initial state;
- the same effective control horizon K-1;
- the same number of iterations and samples per iteration;
- the same base standard-normal latent tensor;
- the same input limits;
- the same Lex-STL evaluator;
- the same discrete reachable-set reference.

The methods then apply their own optimizer-specific distribution transform.
This preserves the distinction between MPPI's temperature/covariance sampling
and IGO's evolving Gaussian factor A.

## Metrics

- best_cost: best Lex-STL cost observed during the run;
- final_cost: cost of the final returned trajectory;
- optimal_match_rate: fraction matching the discrete reachable-set optimum;
- zero_violation_rate: fraction with Lex-STL cost equal to zero;
- relative gap: percentage gap from the discrete reference optimum;
- solve_time_sec: optimizer wall-clock time;
- rollout_effective_rate: one minus the fraction of rollouts pruned;
- transition_effective_rate: fraction of dynamic transitions actually evaluated;
- transition_reduction_rate: fraction of dynamic transitions avoided by pruning.

Rollout-effective rate alone is insufficient because pruning can happen late.
Transition reduction therefore provides the more meaningful computational
saving measurement.

## Local smoke run

From effects-of-discretization-and-solver-evaluations:

    python benchmark_fixed_budget.py --scenarios 20 --seeds 3

This uses 20 iterations x 500 samples = 10,000 rollouts per run.

## Empirical smoke benchmark results

The benchmark was executed under Ubuntu 24.04 (20 scenarios x 3 seeds, 60 paired cases, 10,000 rollouts per method):

| Method | Runs | Optimal Match Rate | Mean Relative Gap (%) | Median Relative Gap (%) | Mean Solve Time (s) | Zero Violation Rate | Transition Effective Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `legacy_mppi` | 60 | 71.67% | 2.231% | 0.000% | 0.4215s (421.5 ms) | 8.33% | 1.000 |
| `igo` | 60 | **95.00%** | **0.058%** | 0.000% | **0.0109s (10.9 ms)** | **10.00%** | 1.000 |
| `igo_pruning` | 60 | **95.00%** | **0.058%** | 0.000% | **0.0109s (10.9 ms)** | **10.00%** | 1.000 |

### Key takeaways

1. **Efficiency**: IGO achieves a **~38.5x speedup** over legacy MPPI (10.9 ms vs 421.5 ms) while consuming identical rollout budgets (20 iterations x 500 samples).
2. **Quality**: IGO improves the discrete optimal match rate from **71.7% to 95.0%** (+23.3% absolute improvement), reducing the mean relative gap from 2.231% to 0.058% (38x reduction).
3. **Head-to-head**: Out of 60 paired runs, IGO strictly outperformed MPPI in **16 cases (26.7%)**, matched MPPI in **43 cases (71.7%)**, and trailed in only **1 case (1.7%)**.

## Recommended stochastic study

For a practical repository-level study, start with:

    python benchmark_fixed_budget.py --scenarios 20 --seeds 10 --iterations 20 --samples 500

This gives 10,000 rollouts per method and scenario/seed, while keeping the
study manageable. Increase the number of scenarios or seeds only after the
smoke run is stable.

The FOM paper reports 50 random seeds and 10,000 rollouts per iteration for its
single constant-speed-unicycle benchmark. Those settings should not be copied
blindly to this repository: multiplying them by many scenarios would create a
very large evaluation budget. The repository benchmark is not a reproduction
of the paper's unicycle obstacle benchmark. It instead evaluates the specific
engineering question of replacing the existing Lex-STL MPPI backend.

Reference: arXiv:2603.16231v3, Sections V-A and V-B.
