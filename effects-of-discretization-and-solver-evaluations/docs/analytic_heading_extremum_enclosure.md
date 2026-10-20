# Analytic Heading-Interval Transport Enclosure

Status: experimental, opt-in. It does not change the default certificate bound.

## 1. Motivation

The Ubuntu validation of centered joint enclosure commit `be4dff4` passed 40 focused tests, but the matched experiment did not change global epsilon or pruning and added roughly 0.25 s of certificate-build time. Diagnostics showed that the selected lower bound at the reported worst cells remained the separable candidate.

The next targeted source of conservatism is the heading-angle dependency. The existing transport bound independently minimizes `g_x cos(phi)` and `g_y sin(phi)` over interval boxes. However, cosine and sine share the same heading angle, so their independent extrema need not be jointly attainable.

## 2. Bound being computed

On each subcell, existing interval analysis gives certified component intervals

`g_x in [gx0,gx1]`, `g_y in [gy0,gy1]`, and heading interval `phi in [phi_lo,phi_hi]`.

The new candidate computes the exact minimum of the interval-box relaxation

`L_heading = min_{phi in [phi_lo,phi_hi]} min_{gx in [gx0,gx1], gy in [gy0,gy1]} v_bar (gx cos(phi) + gy sin(phi))`.

This relaxation permits `g_x` and `g_y` to vary independently, so it remains conservative even though it does not retain their full spatial correlation. Unlike the existing separable product bound, it retains the common `phi` argument of cosine and sine.

For a fixed `phi`, the inner minimum chooses `gx0` when `cos(phi)>=0`, otherwise `gx1`; similarly it chooses `gy0` when `sin(phi)>=0`, otherwise `gy1`. Split the heading interval at every zero of cosine (`pi/2 + k*pi`) and sine (`k*pi`). On each resulting segment, the active endpoints are constant, leaving

`f(phi) = A cos(phi) + B sin(phi)`.

Its minimum on a closed segment occurs at an endpoint or a stationary point `phi = atan2(B,A) + k*pi`. Enumerating these finite candidates therefore computes the exact minimum of this interval-box relaxation, up to floating-point evaluation. The implementation applies a small scale-aware downward margin and `nextafter(..., -inf)` to protect the computed value; this is numerical protection, not a formal directed-rounding interval-arithmetic package.

The full candidate is

`L_heading_total = L_position + L_heading + L_control`,

where the position-cost lower bound and control lower bound remain the established conservative bounds. The selected subcell bound is

`L_selected = max(L_separable, L_centered_joint, L_heading_total)`,

with the centered-joint candidate participating only when its separate opt-in flag is enabled. The maximum of valid lower bounds remains a valid lower bound.

Because the common-angle feasible set is a subset of the independent cosine/sine interval product, the analytic heading transport lower bound should be no weaker than the existing sum of independent x/y transport lower bounds, up to numerical tolerance.

## 3. Implementation and diagnostics

- New opt-in factory argument: `heading_extremum_enclosure=False`.
- New opt-in CLI flag: `--certificate-heading-extremum-enclosure`.
- Default behavior remains unchanged when the flag is absent.
- Per-subcell diagnostics include `heading_transport_lower`, `heading_total_lower`, and `selected_bound_method`.
- Parent-cell diagnostics include `heading_cell_lower_bound` and `heading_cell_gain_over_separable`.
- The selected bound still passes through the existing global epsilon gate; this method does not bypass admissibility, refinement, or rollout-cost checks.

## 4. Ubuntu validation matrix

Run the focused suite:

```bash
python -m pytest -q \
  tests/test_fom_certificate.py \
  tests/test_fom_certificate_igo.py \
  tests/test_fom_igo_solver.py
```

Run the matched baseline and heading-aware experiment with the same grid, seed, iteration capacities, first-level refinement budget, deep budget, and subdivision shape:

```bash
python scripts/run_fom_certificate_igo.py \
  --certificate-iterations 0 50 300 \
  --certificate-refinement-budget 512 \
  --certificate-deep-refinement-budget 8 \
  --certificate-deep-refinement-subdivisions 4 4 2 \
  --output outputs/fom_cert_heading_off

python scripts/run_fom_certificate_igo.py \
  --certificate-iterations 0 50 300 \
  --certificate-refinement-budget 512 \
  --certificate-deep-refinement-budget 8 \
  --certificate-deep-refinement-subdivisions 4 4 2 \
  --certificate-heading-extremum-enclosure \
  --output outputs/fom_cert_heading_on
```

Optionally run a third ablation with both `--certificate-heading-extremum-enclosure` and `--certificate-joint-extremum-enclosure` to check complementarity; do not mix it into the primary one-factor comparison.

For K=50 and K=300 compare:

1. `epsilon`, the final per-cell maximum-violation consistency gate, and the active worst cell.
2. `heading_cell_lower_bound`, `heading_cell_gain_over_separable`, and the per-subcell selected method.
3. The invariant that the heading transport bound dominates the old x/y product lower bound within floating-point tolerance.
4. Terminal error, Bellman monotonicity, discrete-subsolution status, independent rollout-cost agreement, pruning counts, and completed rollouts.
5. Certificate-build time, diagnostic time, IGO solve time, and batch-query speedup.

The analytic candidate is useful only if it tightens the active cell/global epsilon or enables additional safe pruning without breaking any independent gate. If epsilon is unchanged, report the result as a successful conservative enclosure with no measured global planning benefit; do not enable it by default.

## 5. Scope

This is an interval-box relaxation for the 3D constant-speed unicycle benchmark. It does not certify the five-state CommonRoad bicycle model or the trajectory-level Lex-STL objective. It is also not a claim that the exact continuous Hamiltonian minimum has been computed: the exact minimum is taken over the conservative component-interval relaxation, not over the correlated true gradient field.
