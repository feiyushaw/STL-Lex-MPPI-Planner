# HJB Gradient-Interval Coverage Audit

Status: diagnostic-only validation; no change to certificate values, bound
selection, or planner defaults.

## Motivation

The analytic heading enclosure computes a lower bound using gradient intervals
for the trilinear residual plus an interval enclosure for the radial baseline's
gradient. The continuous-domain argument depends on these intervals containing
the relevant gradients throughout every subcell. The PR #6 heading-extremum
oracle tests validate the scalar minimizer given a gradient box; they do not
independently validate construction of the gradient boxes.

This change exposes the already-computed per-subcell reports through the
diagnostic-only `cell_slack_diagnostics(...)` result and adds tests that
independently evaluate the gradient of the trilinear residual and the analytic
radial baseline at sampled subcell points.

## What is checked

- The residual gradient is reconstructed from the eight parent-cell nodal
  residuals, independently of the production interval-bound loop.
- For each tested subcell, the reported residual-gradient intervals are checked
  against independently reconstructed gradients at that same subcell's eight
  corners. Each trilinear derivative is bilinear in the other two coordinates
  and independent of the differentiated coordinate, so its extrema over the
  subcell occur at the endpoint combinations of the other two coordinates.
  The independent evaluator maps subcell corners back into parent-cell
  coordinates, including periodic-heading index wrapping.
- The radial-gradient interval is checked separately at every sampled point;
  total-gradient containment alone could otherwise hide an overly narrow
  component interval behind slack in the other component. For the implemented
  baseline V_rad = (q_position / (3 v_bar)) max(d^3 - rho^3, 0), where
  d = ||p - goal||, its gradient is zero for d < rho and
  (q_position / v_bar) d (p - goal) for d > rho. The derivative is
  discontinuous at d = rho, so exact kink samples are skipped.
- Samples include subcell interiors and faces for all 32 deep subcells in the
  configured `4 x 4 x 2` subdivision.
- The total gradient is reconstructed as residual gradient plus the radial
  baseline gradient, and each sampled component must lie within the reported
  total-gradient interval.
- Cases cover a free-space cell, cells intersecting the configured obstacle
  boundary, and cells adjacent to obstacle boundaries near the periodic
  heading seam.
- Non-finite nodal values are treated consistently with the existing
  fail-closed diagnostic path.

Samples exactly on the radial terminal-cost kink are skipped because the
classical gradient is not defined there. The test samples points on either side
of that kink where they lie inside a subcell. This is a coverage audit, not a
proof that an interval contains every point of the continuous domain.

## Reproduction

From `effects-of-discretization-and-solver-evaluations`:

```bash
python -m pytest -q \
  tests/test_fom_certificate.py \
  tests/test_fom_certificate_igo.py \
  tests/test_fom_igo_solver.py
```

The tests report the obstacle layout, parent cell, subcell, sampled state,
gradient, and interval if a containment check fails.

## Interpretation and next gates

Passing the subcell-corner residual-extrema checks and sampled radial/total
coverage is necessary but not sufficient for a formal continuous-domain
guarantee. The proof still needs to rely on the structure of
the trilinear derivative (bilinear in the other two coordinates, whose extrema
over a rectangle occur at corners) and on the radial-gradient interval
derivation. In particular, obstacle-admissibility boundaries and the radial
terminal-cost kink need explicit mathematical treatment rather than relying on
sampling alone.

This PR does not change any lower-bound formula or turn on either optional
enclosure by default. If coverage tests fail, fix the interval construction
before expanding the performance study or changing defaults.
