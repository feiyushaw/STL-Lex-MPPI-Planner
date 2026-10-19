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
- epsilon_worst_cell: grid-cell index attaining the largest finite slack
  violation, for diagnosing where epsilon originates;
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
renormalizing the remaining corners. In the generic path, a positive-weight
unavailable corner falls back to the declared global value lower bound, and
admissible nodes adjacent to blocked cells are pinned to that bound. In the
radial-extension path, blocked nodes use the global lower bound as interpolation
surrogates, while admissible halo nodes retain the tapered Bellman field rather
than being overwritten with zero. This avoids an artificial residual jump
between the radial boundary extension and the obstacle halo. The continuous
baseline-plus-residual interpolant is explicitly included in the cellwise
HJB-slack bound. Admissible nodes without a usable Bellman successor fall back
to the declared radial subsolution. Cells proven to lie wholly inside an
obstacle are excluded from the HJB domain; ambiguous non-finite cells make the
error gate infinite.

The paper benchmark additionally supplies the analytic radial function as a
global cost-to-go lower bound and as a continuous HJB subsolution outside the
finite grid. The stored nodal field is represented as that analytic function
plus a multilinearly interpolated residual. The residual is smoothly tapered
to zero at the outer boundary over four node layers, and out-of-grid queries
use the analytic radial function. This makes the boundary trace continuous
without forcing a large radial value to jump to zero. The HJB slack is still
bounded over every in-grid cell, including the tapered transition and cells
adjacent to obstacles; no cells are omitted merely to improve epsilon. This
extension is enabled only when the problem explicitly declares the outside
function to be a subsolution and uses it as the terminal-cost baseline. Queries without
a finite certificate value disable pruning. If a problem does not provide a
rigorous cellwise HJB-slack bound or terminal-cell bound, its corresponding
error is infinite and pruning is disabled; sampled nodal finite differences
are not treated as a continuous certificate proof.

## Local slack audit

\`cell_slack_neighborhood(center, radius=1)\` reports the selected cell and its
neighbors with their physical bounds, lower slack bound, violation, distance
to the outer grid boundary, and whether any corner touches the conservative
obstacle halo. Fully inadmissible cells are shown as excluded rather than
silently omitted from the diagnostic. This method is read-only and does not
change the cells or tolerances used by the certificate gate. The benchmark
runner stores this neighborhood beside each capacity result so the source of
the worst slack can be reviewed after a run.

## Batched certificate queries

\`value_batch\`, \`corrected_lower_bound_batch\`, and \`prefix_bound_batch\` are
vectorized counterparts of the scalar query methods. The IGO rollout loop
uses a single batch prefix-bound query for all currently active rollouts at
each horizon step. This avoids one Python-level interpolation call per
sample while preserving the same conservative rule: non-finite or
inadmissible certificate queries return a lower bound of negative infinity
and therefore cannot trigger pruning. Scalar-versus-batch equivalence and
pruned-rollout isolation are covered by regression tests.

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
