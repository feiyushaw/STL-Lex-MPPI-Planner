# Analytic Heading-Aware HJB Transport Enclosure

## Motivation

The centered joint enclosure in PR #4 is safe to retain as an opt-in candidate, but the controlled Ubuntu experiment at commit `be4dff4` reported no global epsilon or pruning gain. Its heading-angle variation radius remains a major source of conservatism.

For the constant-speed unicycle, the transport contribution to the HJB slack is

```text
v * (g_x cos(phi) + g_y sin(phi))
```

where `g_x` and `g_y` are spatial derivatives of the value interpolant plus the radial baseline derivative. Independently bounding the two products loses the fact that sine and cosine share the same `phi`.

## Proposed bound

Given certified gradient intervals `g_x in [gx0,gx1]`, `g_y in [gy0,gy1]` and a subcell heading interval `phi in [phi_lo,phi_hi]`, the implementation computes

```text
min_phi [ min(gx0*cos(phi), gx1*cos(phi))
        + min(gy0*sin(phi), gy1*sin(phi)) ] * v
```

The heading interval is split at zeros of cosine and sine. Within each segment, the active gradient endpoint for each term is fixed, reducing the function to `A*cos(phi) + B*sin(phi)`. Its minimum occurs at a segment endpoint or a stationary point `atan2(B,A) + k*pi`; all such candidates are evaluated analytically.

This is a valid lower bound over the product of the gradient intervals and the shared heading interval. It can be combined with the independently conservative position-cost and control-term lower bounds. The final selected lower bound remains the maximum of valid candidates, so the established separable bound cannot be weakened.

## Scope and non-goals

- Applies only to the current constant-speed unicycle certificate, not to arbitrary dynamics.
- Does not change IGO rollout behavior, the value-iteration operator, or default certificate settings.
- Keeps the new method opt-in via `--certificate-heading-extremum-enclosure`.
- Retains the existing centered joint enclosure as a separate optional candidate.
- Uses a small downward floating-point margin; this is not a replacement for rigorous interval arithmetic or a formal rounding-error proof.

## Validation gates

1. Analytic minimization is no higher than dense numerical samples over representative signed gradient intervals and heading spans.
2. Heading-aware transport lower bound dominates the independent x/y product lower bounds within numerical tolerance.
3. Selected cell lower bounds and global epsilon do not regress against the separable baseline.
4. The option remains off by default and records method provenance in output diagnostics.
5. Ubuntu authoritative validation compares separable-only, centered-joint-only, and heading-aware modes on identical seeds/configurations. Report epsilon, worst-cell identity and bound, selected method, pruning counts, build time, and solve time.
6. Dense sampling is a diagnostic check, not a continuous-domain proof. The conservative guarantee follows from the analytic interval argument and correctness of the supplied gradient intervals.

## Ubuntu four-way comparison (2026-10-20)

The completed controlled comparison evaluated four configurations with matched inputs and budgets:

1. **Separable only** — reference.
2. **Centered joint only** — the centered joint transport/control candidate.
3. **Analytic heading only** — the analytic minimization of the combined heading transport term.
4. **Joint + analytic heading** — both optional candidates enabled.

Reported results for the tested grid and refinement configurations:

- Analytic-heading-only was the best configuration: coarse error decreased by about **35%**, the refinement budget retained about **25% headroom**, global epsilon reached approximately **10.14**, and the prior corner bottleneck at **[4, 11, 3]** was no longer the controlling cell.
- The joint-plus-heading configuration produced no additional measurable gain over analytic-heading-only; the result was governed by the analytic heading candidate.
- The centered joint candidate alone did not tighten any of the measured cells. Its control-term gain was cancelled by the control Lipschitz radius, while the transport difference was negative because the finite-cell transport radius was more conservative than the separable enclosure.
- Therefore, analytic heading is the **recommended reference configuration for subsequent experiments on this constant-speed unicycle benchmark**. The centered joint option should remain off unless a later derivation demonstrates a repeatable benefit.

These are empirical results for the tested geometry, grid, seeds and refinement budgets, not a universal ordering for all dynamics or discretizations. The analytic heading option remains opt-in in production configuration until the gradient-interval coverage audit and the corresponding Ubuntu regression run pass. A recommendation to use it as the experimental reference does not by itself justify enabling it for other dynamics or claiming a continuous-domain certificate beyond the assumptions of the derivation.

## Expected result

The method targets a specific source of dependency overestimation without adding spatial subdivisions. It may tighten the lower bound when the heading interval is broad and gradient signs make independent trigonometric products pessimistic. It is not expected to help every cell; acceptance depends on whether the new candidate becomes active at bottleneck cells and reduces global epsilon or improves pruning enough to justify its cost.
