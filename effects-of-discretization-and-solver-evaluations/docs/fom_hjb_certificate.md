# FOM HJB certificate implementation

## Scope

This change implements the certificate side of the FOM method in
arXiv:2603.16231v3 before any MPPI comparison or optimizer-specific tuning.

The implementation follows the paper's primal-dual separation:

1. define a Bolza problem with dynamics, running cost, terminal cost, and
   admissible state/control sets;
2. construct an independent finite-capacity certificate class;
3. initialize the certificate with a known HJB subsolution when available;
4. tighten it by monotone semi-Lagrangian Bellman sweeps;
5. validate the continuous HJB slack and terminal mismatch;
6. use the corrected certificate
   "v(x) - epsilon * (T-t) - epsilon_T" for prefix lower bounds;
7. compare those bounds with a feasible incumbent and discard dominated
   prefixes before completing their rollouts.

## Relation to the paper

For the benchmark in Section V-A/B:

- state is (p_x, p_y, phi);
- dynamics are the constant-speed unicycle;
- running cost is q_p ||p-p_g||^2 + r_u u^2;
- terminal cost is the radial warm start
  v_rad = q_p (d^3-rho^3) / (3 v_bar) for d > rho and zero inside;
- heading is periodic;
- obstacles are represented as inadmissible state regions;
- certificate tightening is semi-Lagrangian value iteration;
- the primal solver is implicit FOM with IGO/xNES;
- pruning uses the incumbent-based cumulative lower bound.

The implementation explicitly accounts for two numerical approximation layers
that are easy to get wrong:

### Control discretization

The semi-Lagrangian backup uses a finite control grid. A finite subset gives
an upper approximation to the continuous infimum, which is unsafe for a lower
bound if used directly. The implementation therefore subtracts a Lipschitz
error bound in the scalar control dimension before accepting the backup.

### Spatial interpolation

The certificate is multilinearly interpolated. The paper requires an
epsilon accounting for continuous HJB slack. For the unicycle benchmark,
cellwise derivative intervals, exact trigonometric intervals, the exact
position-cell minimum, and continuous minimization of the quadratic control
term are used to obtain a one-sided slack bound.

The terminal error is bounded over all grid cells intersecting the target,
rather than treating only target grid nodes as a global check. The radial
terminal-cell lower bound is zero because the terminal cost is zero on the
target set.

## Certificate invariants

A built certificate reports:

- epsilon: global running-slack feasibility tolerance;
- epsilon_terminal: terminal feasibility tolerance;
- bellman_monotone: every sweep improved or preserved the previous candidate;
- discrete_subsolution: the raw Bellman operator still dominates the selected
  certificate at all finite admissible nodes.

A certificate is not considered valid for pruning merely because the grid
backup succeeded. The corrected lower bound is the object used by the primal
search. The finite-control correction uses the Lipschitz constant of
dt*ell(x,u) + v(x + dt*f(x,u)); the running-cost derivative therefore carries
the same dt factor as the state-transition term.

For K = 0, the benchmark uses the analytic radial function directly, not its
grid interpolation. Its zero feasibility errors are asserted only because the
benchmark explicitly declares that function to be a continuous HJB
subsolution and the same function is the terminal cost.

For K > 0, an interpolation cell with a non-finite corner is never repaired by
renormalizing the remaining corners. Such cells use the declared global value
lower bound, and the admissible nodes adjacent to blocked cells and the outer
grid boundary are pinned to that same lower bound. Queries without a finite
certificate value disable pruning. If a problem does not provide a rigorous
cellwise HJB-slack bound or terminal-cell bound, its corresponding error is
infinite and pruning is disabled; sampled nodal finite differences are not
treated as a continuous certificate proof.

For custom Bolza problems, `global_value_lower_bound` must be a valid lower
bound on every admissible cost-to-go, and `running_cost_lower_bound` must be a
valid global lower bound on the running cost. Their zero defaults are valid
for this benchmark because both costs are nonnegative; incorrect declarations
would invalidate the certificate.

`discrete_subsolution` is only a finite-grid diagnostic. It does not replace
the continuous cellwise slack bound or terminal error accounting.

## IGO integration

CertificateGuidedIGOSolver is the implicit-FOM primal realization used for
the paper-style problem. It keeps the certificate fixed while optimizing the
Gaussian search distribution with the xNES update.

Pruned rollouts are not assigned their lower bound as if it were a completed
trajectory cost. They are discarded from the elite population.

## Important boundary

The existing CommonRoad planner has a five-state bicycle model and a
trajectory-level Lex-STL objective. The paper does not provide a direct,
drop-in HJB certificate for that hybrid objective. Therefore this PR does not
claim that the current 3D unicycle certificate is already a certified lower
bound for the CommonRoad Lex-STL score.

The correct next step is to define a Markov-compatible Bolza relaxation for
the CommonRoad objective (or an augmented state/automaton representation) and
prove the relation between that certificate objective and the Lex-STL score
before enabling certificate-based pruning there.

That separation is intentional: it prevents an attractive but mathematically
invalid certificate from silently pruning valid CommonRoad candidates.

## Validation sequence

Run:

~~~bash
python -m pytest -q \
  tests/test_fom_certificate.py \
  tests/test_fom_certificate_igo.py \
  tests/test_fom_igo_solver.py
~~~

Then run the C++ regression suite already associated with PR #1.

Only after these checks pass should the paper-style certificate capacity study
be run for K = 0, 50, 300.

The subsequent MPPI-vs-IGO comparison belongs to a separate experiment. It is
not used to establish certificate correctness.
