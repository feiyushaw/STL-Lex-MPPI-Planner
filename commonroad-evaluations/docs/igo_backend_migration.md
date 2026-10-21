# IGO Backend Migration Contract

## Production objective

Replace the active CommonRoad planner's historical MPPI update with the FOM-IGO/xNES search path while retaining source/config compatibility for downstream users. The canonical runtime selector is `igo`; the old `mppi` selector remains a compatibility alias to the same active implementation. The independent `preemptive_mppi` planner is not changed by this migration.

## Canonical configuration

The active optimizer reads `planner.igo`:

- `num_iterations`, `n_samples_initial`, and `n_samples_final` configure the search budget.
- `initial_covariance` initializes each temporal block's covariance factor.
- `sampling_method: pureRandom` generates Gaussian latent samples, matching the xNES sampling assumption. LHS variants are opt-in experiment modes.
- `elite_quantile`, `block_steps`, `lr_mean`, and `lr_cov` configure block-xNES.
- `effort_tiebreak_weight` affects only the deterministic control-effort tie-break after the primary Lex-STL costs are equal; it does not change the primary lexicographic cost.
- `enable_operational_limit_pruning` enables the narrow rulebook-prefix guard for operational limits.

The old `planner.mppi` section is still parsed. If `planner.igo` is absent, its relevant values seed the new configuration, so existing YAML files continue to work. When `planner.igo` is present, its values take precedence; legacy MPPI temperature and sampling settings do not control the active IGO path.

## xNES update contract

For each iteration, the production optimizer samples latent vectors $s_i\sim\mathcal{N}(0,I)$ and forms bounded controls from $u_i=m+A s_i$. It ranks completed rollouts by the Lex-STL integer cost, using the effort diagnostic only as a deterministic tie-breaker, then retains the top `elite_quantile` fraction with equal normalized weights.

For a temporal block $b$, the implementation computes

```text
g_m,b = sum_i w_i s_i,b
G_b   = sum_i w_i (s_i,b s_i,b^T - I_b)
m_b   <- m_b + lr_mean * A_b * g_m,b
A_b   <- A_b * exp(0.5 * lr_cov * sym(G_b))
```

The latent vectors are the original Gaussian samples, not reconstructed from the clipped controls. Control projection changes the evaluated rollout but does not rewrite the whitened statistics. Block slices use half-open `[start, end)` semantics. The C++ regression checks the exact mean and covariance-factor update on a deterministic two-elite example.

The paper's benchmark uses 10,000 rollouts per iteration and the top 5% elite set. The production defaults intentionally retain a smaller, configurable search budget for the CommonRoad planner; this is a deployment-budget adaptation, not a claim of reproducing the paper's benchmark settings.

## Certificate boundary

The production C++ guard only stops a rollout when an operational-limit violation is guaranteed to lose against an incumbent that already satisfies the highest-priority operational-limit rule. It is not the independent semi-Lagrangian HJB certificate.

The Python FOM/HJB benchmark constructs a certificate for a specific constant-speed unicycle Bolza objective. It must not prune arbitrary Lex-STL trajectories until the state dynamics, running/terminal objective, admissible set, interpolation error and corrected lower bound are shown to be compatible with the exact production objective. Keeping these mechanisms separately named is intentional and prevents an unsupported certificate claim.

## Reproduction commands

From the repository root, in the Ubuntu environment with the project's CMake dependencies installed:

```bash
cmake -S commonroad-evaluations -B commonroad-evaluations/build -DBUILD_FOM_IGO_TESTS=ON
cmake --build commonroad-evaluations/build --target lex_stl_planner_bindings fom_igo_block_updater_test fom_igo_config_test --parallel 2
ctest --test-dir commonroad-evaluations/build --output-on-failure
```

Then run the existing Python FOM/IGO suite from its experiment directory:

```bash
cd effects-of-discretization-and-solver-evaluations
python -m pytest -q tests
```

Run the representative CommonRoad replay from the CommonRoad directory:

```bash
cd commonroad-evaluations
python run_cr_scenario.py
```

These commands are a validation recipe, not a claim that the commands have already passed in this session.

## Validation gates

1. Build the CommonRoad pybind module and run all existing CTest/Python tests.
2. Run `fom_igo_config_test` to check canonical YAML parsing and isolation from the legacy sampling configuration.
3. Run the planner on the representative CommonRoad scenarios and verify that `planner_type = "igo"` selects the same C++ implementation as the compatibility alias.
4. Audit xNES sampling, rank selection, covariance updates and projection against the paper implementation.
5. Only after the first four gates pass, reconsider whether any additional HJB lower bound can be proven valid for a specific STL objective.

A passing config/unit test does not by itself establish the full paper implementation or CommonRoad planning superiority; scenario-level validation remains required.
