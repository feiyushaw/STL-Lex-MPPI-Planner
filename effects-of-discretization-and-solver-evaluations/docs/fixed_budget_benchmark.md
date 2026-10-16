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

## Benchmark artifact status

A first Ubuntu smoke run was completed with 20 scenarios x 3 seeds and
10,000 rollouts per method. That run reported:

- legacy MPPI optimal-match rate: 71.67%;
- IGO optimal-match rate: 95.00%;
- legacy MPPI mean solve time: 421.5 ms;
- IGO mean solve time: 10.9 ms;
- IGO strictly better in 16/60 paired cases, tied in 43/60, worse in 1/60.

However, the first run is now **archived and superseded**. Code inspection found
that the feasible-incumbent state used by the pruning path was initialized but
was not updated when a zero-cost trajectory was found. Consequently the
reported igo_pruning row had zero pruning and was effectively identical to
primal-only IGO. The old CSVs are retained under
outputs/fixed_budget_benchmark/archive/ for provenance and must not be used
as evidence for pruning effectiveness.

The current implementation now records a feasible incumbent explicitly and
adds regression tests for both cases:

- a feasible incumbent exists -> certificate pruning may activate;
- no feasible incumbent exists -> pruning remains disabled.

The benchmark must therefore be rerun before treating igo_pruning results as
validated.

The MPPI-vs-IGO result itself remains useful as a preliminary fixed-budget
result. Its 38.5x wall-clock difference should be described as an
**implementation-level end-to-end speed difference**, not as a pure theoretical
speedup of IGO: the legacy reference uses a per-rollout Python loop whereas
the IGO path is substantially vectorized.

The 95% vs 71.67% optimal-match result is also a result on the repository's
synthetic 1D Lex-STL benchmark, not yet evidence of CommonRoad planning
superiority.

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

Reference: arXiv:2603.16231v3, Sections III-B and V-B.
