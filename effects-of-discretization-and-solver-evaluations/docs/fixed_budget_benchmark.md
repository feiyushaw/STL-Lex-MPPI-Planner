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

## Publication-scale run

The FOM paper reports 50 random seeds and 10,000 rollouts per iteration.
For a comparable stochastic study, increase the benchmark budget explicitly,
for example:

    python benchmark_fixed_budget.py --scenarios 100 --seeds 50 --samples 10000

The repository benchmark is not a reproduction of the paper's unicycle
obstacle benchmark. The paper uses a constant-speed unicycle, a 900-step
horizon, independently constructed semi-Lagrangian certificates, and four FOM
variants plus AL-DDP. This repository study instead evaluates the specific
engineering question of replacing the existing Lex-STL MPPI backend.

Reference: arXiv:2603.16231v3, Sections V-A and V-B.
