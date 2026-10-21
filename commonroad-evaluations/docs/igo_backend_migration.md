# IGO Backend Migration Contract

## Production objective

Replace the active CommonRoad planner's historical MPPI update with the FOM-IGO/xNES search path while retaining source/config compatibility for downstream users. The canonical runtime selector is `igo`; the old `mppi` selector remains a compatibility alias to the same active implementation. The independent `preemptive_mppi` planner is not changed by this migration.

## Canonical configuration

The active optimizer reads `planner.igo`:

- `num_iterations`, `n_samples_initial`, and `n_samples_final` configure the search budget.
- `initial_covariance` initializes each temporal block's covariance factor.
- `sampling_method: pureRandom` generates Gaussian latent samples, matching the xNES sampling assumption. LHS variants are opt-in experiment modes.
- `elite_quantile`, `block_steps`, `lr_mean`, and `lr_cov` configure block-xNES.
- `reporting_effort_weight` affects only the reported effort diagnostic, never Lex-STL ranking.
- `enable_operational_limit_pruning` enables the narrow rulebook-prefix guard for operational limits.

The old `planner.mppi` section is still parsed. If `planner.igo` is absent, its relevant values seed the new configuration, so existing YAML files continue to work. When `planner.igo` is present, its values take precedence; legacy MPPI temperature and sampling settings do not control the active IGO path.

## Certificate boundary

The production C++ guard only stops a rollout when an operational-limit violation is guaranteed to lose against an incumbent that already satisfies the highest-priority operational-limit rule. It is not the independent semi-Lagrangian HJB certificate.

The Python FOM/HJB benchmark constructs a certificate for a specific constant-speed unicycle Bolza objective. It must not prune arbitrary Lex-STL trajectories until the state dynamics, running/terminal objective, admissible set, interpolation error and corrected lower bound are shown to be compatible with the exact production objective. Keeping these mechanisms separately named is intentional and prevents an unsupported certificate claim.

## Validation gates

1. Build the CommonRoad pybind module and run all existing CTest/Python tests.
2. Run `fom_igo_config_test` to check canonical YAML parsing and isolation from the legacy sampling configuration.
3. Run the planner on the representative CommonRoad scenarios and verify that `planner_type = "igo"` selects the same C++ implementation as the compatibility alias.
4. Audit xNES sampling, rank selection, covariance updates and projection against the paper implementation.
5. Only after the first four gates pass, reconsider whether any additional HJB lower bound can be proven valid for a specific STL objective.

A passing config/unit test does not by itself establish the full paper implementation or CommonRoad planning superiority; scenario-level validation remains required.
