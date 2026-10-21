# STL-Lex-MPPI-Planner

The historical MPPI planner interface is retained for compatibility, but the
active optimization backend on the main planner path is IGO/xNES.

The active backend is configured under `planner.igo` in
`commonroad-evaluations/config/default_config.yaml`. It uses independent
Gaussian latent samples by default, as assumed by xNES. The old
`planner.mppi` section remains for compatibility and for the separate
preemptive-MPPI implementation; it no longer controls the active IGO optimizer.

Important certificate boundary: the CommonRoad C++ planner currently uses a
specialized operational-limit prefix guard. It is **not** the general
semi-Lagrangian HJB certificate implemented in the Python FOM benchmark.
That HJB certificate is not enabled for arbitrary Lex-STL costs because its
dynamics and Bolza objective must be proved compatible before its lower bound
can safely prune those rollouts.

See [the IGO backend migration contract](commonroad-evaluations/docs/igo_backend_migration.md) for the canonical config, compatibility fallback, and Ubuntu validation commands.

Code to reproduce the numerical experiments from:

Patrick Halder, Lothar Kiltz, Hannes Homburger, Johannes Reuter, Matthias Althoff,
"Lexicographic Minimum-Violation Motion Planning using Signal Temporal Logic,"
IEEE Open Journal of Intelligent Transportation Systems, 2026.
DOI: https://doi.org/10.1109/OJITS.2026.3726082

![Intersection scenario teaser](cr_scenario.gif)

## Repository Structure (Experiment Reproduction)

This repository has two main experiment directories:

1. [effects-of-discretization-and-solver-evaluations](effects-of-discretization-and-solver-evaluations)
	 - Contains the experiments for:
		 - effects of discretization,
		 - IGO/xNES solver replacement for the historical MPPI backend,
		 - comparison of robustness measures
	 - Installation and usage instructions are in its local README.

2. [commonroad-evaluations](commonroad-evaluations)
	 - Contains the CommonRoad experiments:
		 - intersection scenario evaluation,
		 - comparison with preemptive planning,
		 - comparison with alternative solvers on CommonRoad scenarios.
	 - Installation and usage instructions are in its local README.

## Contact

Patrick Halder: patrick.halder@tum.de

## Citation

```bibtex
@Article{Halder2026,
  author  = {Halder, Patrick and Kiltz, Lothar and Homburger, Hannes and Reuter, Johannes and Althoff, Matthias},
  journal = {IEEE Open Journal of Intelligent Transportation Systems},
  title   = {Lexicographic Minimum-Violation Motion Planning using Signal Temporal Logic},
  year    = {2026},
  pages   = {2307--2326},
  volume  = {7},
}
```
