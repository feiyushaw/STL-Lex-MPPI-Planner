# Analytic Heading Enclosure: Robustness and Cross-Grid Validation

Status: validation infrastructure; no default behavior change.

## Purpose

PR #4 integrates the opt-in analytic heading-aware lower bound for the
constant-speed unicycle HJB certificate. The initial controlled experiment
showed lower global epsilon at grid shape \`(21, 21, 12)\`. This follow-up checks
two different questions:

1. Does the scalar analytic minimizer agree with an independent analytic
   reference across signed gradient boxes and difficult heading intervals?
2. Does enabling the heading enclosure preserve or improve certificate
   violation metrics across several grid resolutions?

These checks do not establish transfer to other dynamics or obstacle layouts.

## Independent analytic reference

For fixed heading, the transport expression is linear in \`g_x\` and \`g_y\`.
Therefore its minimum over a rectangular gradient box is attained at one of
the four gradient-box corners. For each corner \`(g_x, g_y)\`, the minimum of

\`g_x cos(phi) + g_y sin(phi)\`

over a closed heading interval is attained at an interval endpoint or a
stationary point \`atan2(g_y, g_x) + k*pi\`. The test oracle enumerates these
candidates without using the production helper's sign-change segmentation.

The deterministic randomized regression covers 256 gradient boxes, speeds,
heading intervals, intervals crossing trigonometric sign changes, singleton
heading intervals, and a full \`[-pi, pi]\` span. It checks agreement to a
scale-aware floating-point tolerance. Invalid bounds, non-finite values, and
non-positive speed are also rejected.

The oracle is a numerical implementation cross-check. The continuous-domain
lower-bound argument still depends on the analytic derivation and on the
certificate's gradient intervals enclosing the true interpolant gradients.

## Cross-grid A/B protocol

The harness compares:

- \`separable\`: existing separable enclosure, with both optional flags off;
- \`heading\`: \`--certificate-heading-extremum-enclosure\` enabled, with the
  centered joint enclosure still off.

Both modes use identical geometry, seeds, certificate iteration counts,
first-level refinement budget, deep-refinement budget, and subdivision shape.
Default grid shapes are \`(15, 15, 8)\`, \`(21, 21, 12)\`, and \`(27, 27, 16)\`.
The default certificate capacities are \`K = 0, 50, 300\`, first-level budget
512, deep budget 8, and deep subdivision shape \`4 x 4 x 2\`.

The harness uses a small IGO smoke budget (\`32\` samples and \`1\` iteration) to
keep this certificate-focused sweep practical. These runs must not be used as
optimizer-quality or production solve-time benchmarks.

For each grid and K, the JSON report records epsilon, coarse epsilon,
worst unrefined/refined violation, worst-cell identity, refined-cell count,
budget exhaustion, build time, and pruning counts. It fails if any of the
following metrics regresses by more than \`1e-8\`:

- \`epsilon\`
- \`epsilon_coarse\`
- \`epsilon_unrefined_worst_violation\`
- \`epsilon_refined_worst_violation\`

A non-regression result is necessary but not sufficient for enabling a default:
the method should also show repeatable epsilon reduction or another measurable
benefit, and its mathematical assumptions must remain valid.

## Reproduction on Ubuntu

From the \`effects-of-discretization-and-solver-evaluations\` directory:

\`\`\`bash
python -m pytest -q \\
  tests/test_fom_certificate.py \\
  tests/test_fom_certificate_igo.py \\
  tests/test_fom_igo_solver.py

python scripts/run_heading_enclosure_robustness.py \\
  --grid-shapes 15,15,8 21,21,12 27,27,16 \\
  --certificate-iterations 0 50 300 \\
  --refinement-budget 512 \\
  --deep-refinement-budget 8 \\
  --deep-refinement-subdivisions 4 4 2 \\
  --seed 7
\`\`\`

The harness writes per-mode \`certificate_igo_summary.json\` files and a
combined \`outputs/heading_enclosure_robustness/heading_enclosure_robustness_summary.json\`.
The combined report includes raw run rows, paired metric deltas, and a top-level
\`passed\` flag. Preserve this JSON as the experiment artifact.

## Interpretation and limitations

- The existing separable candidate remains part of the selected maximum, so a
  valid heading candidate should not weaken the per-cell lower bound.
- A changed worst-cell identity is expected when the prior bottleneck is
  tightened; it is not itself a failure.
- The default sweep varies resolution on the same benchmark geometry. A later
  phase should add alternative obstacle layouts and explicitly check gradient
  interval coverage near obstacle halos and grid boundaries.
- Dense sampling is not treated as proof of a continuous-domain inequality.
- The heading enclosure remains opt-in, and the centered joint enclosure
  remains an independent opt-in feature.
