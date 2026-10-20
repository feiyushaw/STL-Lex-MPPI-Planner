# Centered Joint HJB Extremum Enclosure

Status: experimental, opt-in; not the default certificate bound.

## 1. Motivation from the e69e991 sweep

The Ubuntu A/B/C/D sweep kept the first-level refinement budget at 512 and compared deep-refinement settings. The obstacle-corner cell `[4, 11, 3]` improved from a violation of 11.4134 (no deep refinement) to 9.5062 (8x8x4). However, the global epsilon remained controlled by other cells: `[10, 4, 6]` at K=50 and `[19, 19, 1]` at K=300. The reported control contribution in `[10, 4, 6]` was about -14.8.

This suggests an interval-dependency problem rather than a need to uniformly increase subdivision order. The existing enclosure computes lower bounds for x transport, y transport, and the minimized heading-control term independently. Their worst-case values need not occur at the same state, so summing them can be unnecessarily conservative.

The reported sweep validates the earlier implementation at commit `e69e991`; it does not validate this new experimental method.

## 2. HJB slack being bounded

For the constant-speed unicycle benchmark, the cellwise slack is

`H(x,y,phi) = q_p ||p-p_g||^2 + v_bar (g_x cos(phi) + g_y sin(phi)) + m(g_phi)`,

where `g = grad(V)`, and

`m(g_phi) = min_{u in [u_min,u_max]} (r_u u^2 + g_phi u)`.

The current separable enclosure lower-bounds the position cost, x transport, y transport, and control term separately, then sums those lower bounds.

## 3. Centered joint candidate

For each already-defined subcell, select one common anchor state `z_c=(x_c,y_c,phi_c)` at the subcell center. Evaluate the trilinear-residual gradient and analytic radial gradient at that same anchor, giving `g_xc`, `g_yc`, and `g_phic`.

The transport center is

`T_c = v_bar (g_xc cos(phi_c) + g_yc sin(phi_c))`.

Let `dg_x`, `dg_y`, and `dg_phi` be maximum deviations from the anchor gradient components to their existing conservative interval endpoints. Let `dcos` and `dsin` be maximum deviations of the cosine/sine intervals from their center values, and let `Cmax=max |cos(phi)|`, `Smax=max |sin(phi)|` on the subcell. Then

`R_T = v_bar [dg_x Cmax + |g_xc| dcos + dg_y Smax + |g_yc| dsin]`.

This follows from the exact identity

`g_x cos(phi) - g_xc cos(phi_c) = (g_x-g_xc) cos(phi) + g_xc (cos(phi)-cos(phi_c))`,

and the corresponding identity for y transport.

The control center is `C_c=m(g_phic)`. Since `m(g)=min_u(r_u u^2 + g u)` is `Umax`-Lipschitz for `Umax=max(|u_min|,|u_max|)`, the control deviation is bounded by

`R_C = Umax dg_phi`.

The candidate lower bound is

`L_joint = L_position + T_c + C_c - R_T - R_C`,

where `L_position` is the existing exact minimum of the nonnegative position cost over the subcell.

The production candidate is `max(L_separable, L_joint)`. Both are independently conservative lower bounds; taking their maximum remains a lower bound and cannot weaken the existing enclosure. The joint candidate does not clip, discard, or replace the physically negative control contribution. It only reduces avoidable dependency loss by combining the signed transport and control center values at one shared anchor and subtracting conservative variation radii.

## 4. Scope and caveats

- The feature is enabled only with `--certificate-joint-extremum-enclosure`; the default remains disabled until the Ubuntu validation and quality/runtime trade-off are reviewed.
- The existing interval enclosures for the gradient components and trigonometric functions remain the safety basis. This is a centered enclosure, not an exact global minimizer and not a replacement for interval verification.
- The radial baseline has a piecewise definition at the goal-radius boundary. The joint bound uses the existing gradient intervals for variation, including intervals that span the radial branch boundary; the center anchor is not used to narrow those intervals.
- The cellwise final epsilon remains the maximum violation over all admissible cells. Diagnostic fields do not bypass or relax that gate.
- The method remains specific to the 3D constant-speed unicycle HJB benchmark. It does not establish a certificate for the five-state CommonRoad bicycle model or the trajectory-level Lex-STL objective.

## 5. Ubuntu validation matrix

First run the focused suite:

```bash
python -m pytest -q \
  tests/test_fom_certificate.py \
  tests/test_fom_certificate_igo.py \
  tests/test_fom_igo_solver.py
```

Then compare the established B configuration with the opt-in feature. Use identical seeds, iteration capacities, first-level budget, deep budget, deep subdivision shape, and output directories:

```bash
python scripts/run_fom_certificate_igo.py \
  --certificate-iterations 0 50 300 \
  --certificate-refinement-budget 512 \
  --certificate-deep-refinement-budget 8 \
  --certificate-deep-refinement-subdivisions 4 4 2 \
  --output outputs/fom_cert_joint_off

python scripts/run_fom_certificate_igo.py \
  --certificate-iterations 0 50 300 \
  --certificate-refinement-budget 512 \
  --certificate-deep-refinement-budget 8 \
  --certificate-deep-refinement-subdivisions 4 4 2 \
  --certificate-joint-extremum-enclosure \
  --output outputs/fom_cert_joint_on
```

For K=50 and K=300, compare:

1. `epsilon` and the final per-cell maximum-violation consistency gate.
2. The worst refined, unrefined, and deep-refined cells and their violations.
3. In `slack_components.deep`, compare `separable_cell_lower_bound`,
   `joint_cell_lower_bound`, `selected_cell_lower_bound`, and
   `joint_cell_gain_over_separable`; inspect per-subcell
   `separable_total_lower`, `joint_total_lower`, and
   `selected_bound_method` to see where the joint candidate is active.
4. Certificate-build time, separate diagnostic time, IGO time, and pruning counts.
5. Terminal error, Bellman monotonicity, discrete-subsolution status, and independent rollout-cost agreement.

The expected structural invariant is that enabling the joint candidate cannot increase any cell's violation or global epsilon, because the selected lower bound is the maximum of two conservative lower bounds. A lower epsilon is useful only if all independent certificate and rollout-cost gates still pass. Dense sample checks are sanity tests, not a substitute for the enclosure argument above.
