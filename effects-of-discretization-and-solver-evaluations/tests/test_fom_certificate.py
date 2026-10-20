from dataclasses import replace

import numpy as np
import pytest

from utils.fom_certificate import (
    RegularGrid,
    SemiLagrangianFOMCertificate,
    make_paper_unicycle_problem,
    radial_terminal_cost,
)


def make_small_certificate(
    iterations=0,
    obstacles=(),
    refinement_budget=256,
    deep_refinement_budget=8,
    deep_refinement_subdivisions=(4, 4, 2),
):
    problem = make_paper_unicycle_problem(
        goal=(0.0, 0.0),
        goal_radius=0.5,
        forward_speed=1.0,
        q_position=1.0,
        r_control=0.1,
        control_min=-1.0,
        control_max=1.0,
        dt=0.1,
        obstacles=obstacles,
        n_controls=9,
        deep_refinement_subdivisions=deep_refinement_subdivisions,
    )
    grid = RegularGrid(
        lower=np.array([-2.0, -2.0, -np.pi]),
        upper=np.array([2.0, 2.0, np.pi]),
        shape=(9, 9, 8),
        periodic_axes=(2,),
    )
    cert = SemiLagrangianFOMCertificate(
        problem,
        grid,
        iterations=iterations,
        refinement_budget=refinement_budget,
        deep_refinement_budget=deep_refinement_budget,
    )
    return cert


def test_radial_warm_start_matches_closed_form():
    p = np.array([[3.0, 0.0, 0.0], [0.0, 0.0, 1.2]])
    value = radial_terminal_cost(p, (0.0, 0.0), 1.0, 2.0, 1.0)
    assert np.allclose(value, [52.0 / 3.0, 0.0])


def test_k0_certificate_has_zero_feasibility_error():
    cert = make_small_certificate(iterations=0)
    report = cert.build()
    assert report.iterations == 0
    assert report.epsilon_terminal == 0.0
    assert report.epsilon <= 1e-10
    assert report.discrete_subsolution
    assert report.bellman_monotone


def test_k0_queries_use_the_analytic_radial_function_off_grid():
    cert = make_small_certificate(iterations=0)
    cert.build()
    state = np.array([0.83, 0.17, 0.23])
    expected = radial_terminal_cost(
        state[None, :], (0.0, 0.0), 0.5, 1.0, 1.0
    )[0]
    assert np.isclose(cert.value(state), expected, rtol=0.0, atol=1e-12)


def test_certified_radial_extension_anchors_boundary_and_outside_queries():
    cert = make_small_certificate(iterations=2)
    cert.build()

    values = cert.values[-1].reshape(cert.grid.shape)
    radial = cert.problem.terminal_cost(cert.nodes).reshape(cert.grid.shape)

    # The outer boundary matches the analytic extension exactly. The
    # residual is smoothly tapered over four node layers instead of forcing
    # the first interior cell to jump from zero to a large radial value.
    assert np.allclose(values[0, :, :], radial[0, :, :])
    assert np.allclose(values[-1, :, :], radial[-1, :, :])
    assert np.all(values[1, :, :] >= radial[1, :, :] - 1e-12)
    assert not np.any(cert._obstacle_halo_mask[0, :, :])

    outside = np.array([2.5, 0.0, 0.0])
    expected = cert.problem.terminal_cost(outside[None, :])[0]
    assert np.isclose(cert.value(outside), expected, rtol=0.0, atol=1e-12)
    assert np.isfinite(cert.corrected_lower_bound(outside, 0.5))


def test_cell_slack_diagnostics_decompose_bound_without_changing_epsilon():
    cert = make_small_certificate(iterations=2)
    cert.build()
    epsilon_before = cert.epsilon

    rows = cert.cell_slack_neighborhood(center=(4, 4, 2), radius=0)
    assert len(rows) == 1
    components = rows[0]["slack_components"]
    assert components is not None

    coarse = components["coarse"]
    refined = components["refined"]
    assert coarse["subcell_count"] == 1
    assert refined["subcell_count"] == 8
    deep = components["deep"]
    assert deep["subcell_count"] == 32
    assert deep["subdivisions_per_axis"] == [4, 4, 2]
    assert refined["total_lower"] >= coarse["total_lower"] - 1e-9
    assert deep["total_lower"] >= refined["total_lower"] - 1e-9
    assert np.isclose(
        refined["position_cost_lower"]
        + refined["x_transport_lower"]
        + refined["y_transport_lower"]
        + refined["control_lower"],
        refined["total_lower"],
        rtol=0.0,
        atol=1e-10,
    )
    assert "worst_subcell_bounds" in refined
    assert "worst_subcell_bounds" in deep
    assert cert.epsilon == epsilon_before


def test_global_epsilon_matches_the_final_bounds_used_for_pruning():
    cert = make_small_certificate(iterations=2)
    report = cert.build()

    coarse_epsilon = max(
        (max(0.0, -value) for value in cert._epsilon_coarse_cell_lowers.values()),
        default=0.0,
    )
    final_epsilon = max(
        (max(0.0, -value) for value in cert._epsilon_final_cell_lowers.values()),
        default=0.0,
    )
    assert np.isclose(coarse_epsilon, cert._epsilon_coarse, atol=1e-12)
    assert np.isclose(final_epsilon, report.epsilon, atol=1e-12)
    assert report.epsilon_worst_cell is not None

    worst = tuple(report.epsilon_worst_cell)
    row = cert.cell_slack_neighborhood(center=worst, radius=0)[0]
    assert row["is_reported_worst_cell"]
    assert np.isclose(
        row["coarse_slack_lower_bound"],
        cert._epsilon_coarse_cell_lowers[worst],
        atol=1e-12,
    )
    assert np.isclose(
        row["slack_lower_bound"],
        cert._epsilon_final_cell_lowers[worst],
        atol=1e-12,
    )
    if row["refined_in_certificate"]:
        components = row["slack_components"]
        assert components is not None
        expected_bounds = [
            row["coarse_slack_lower_bound"],
            components["refined"]["total_lower"],
        ]
        if row["deep_refined_in_certificate"]:
            expected_bounds.append(components["deep"]["total_lower"])
        assert np.isclose(
            row["slack_lower_bound"],
            max(expected_bounds),
            atol=1e-12,
        )
    else:
        assert np.isclose(
            row["slack_lower_bound"],
            row["coarse_slack_lower_bound"],
            atol=1e-12,
        )
    assert np.isclose(
        max(
            cert._epsilon_refined_worst_violation,
            cert._epsilon_unrefined_worst_violation,
        ),
        report.epsilon,
        atol=1e-12,
    )
    if report.epsilon > 0.0 and not cert._epsilon_refinement_budget_exhausted:
        assert row["refined_in_certificate"]


def test_refinement_budget_must_be_non_negative():
    problem = make_paper_unicycle_problem(
        goal=(0.0, 0.0),
        goal_radius=0.5,
        forward_speed=1.0,
        q_position=1.0,
        r_control=0.1,
        control_min=-1.0,
        control_max=1.0,
        dt=0.1,
        n_controls=9,
    )
    grid = RegularGrid(
        lower=np.array([-2.0, -2.0, -np.pi]),
        upper=np.array([2.0, 2.0, np.pi]),
        shape=(9, 9, 8),
        periodic_axes=(2,),
    )
    with pytest.raises(ValueError, match="refinement_budget must be non-negative"):
        SemiLagrangianFOMCertificate(
            problem, grid, iterations=2, refinement_budget=-1
        )
    with pytest.raises(
        ValueError, match="deep_refinement_budget must be non-negative"
    ):
        SemiLagrangianFOMCertificate(
            problem, grid, iterations=2, deep_refinement_budget=-1
        )


def test_zero_refinement_budget_reports_unresolved_coarse_worst_cell():
    cert = make_small_certificate(iterations=2, refinement_budget=0)
    report = cert.build()

    assert cert._epsilon_refined_cells == 0
    assert cert._epsilon_refined_worst_cell is None
    assert cert._epsilon_unrefined_worst_cell is not None
    assert np.isclose(report.epsilon, cert._epsilon_coarse, atol=1e-12)
    assert np.isclose(
        cert._epsilon_unrefined_worst_violation,
        report.epsilon,
        atol=1e-12,
    )
    assert cert._epsilon_refinement_budget_exhausted


def test_refined_cell_slack_bound_is_no_weaker_than_coarse_bound():
    cert = make_small_certificate(iterations=2)
    cert.build()

    assert np.isfinite(cert._epsilon_coarse)
    assert cert.epsilon <= cert._epsilon_coarse + 1e-10
    assert cert.problem.cell_slack_lower_bound_refined is not None

    cell = cert._epsilon_worst_cell or (0, 4, 2)
    values = cert._effective_extension_values(cert.values[-1])
    coarse = cert.problem.cell_slack_lower_bound(values, cert.grid, cell)
    refined = cert.problem.cell_slack_lower_bound_refined(
        values, cert.grid, cell
    )
    deep = cert.problem.cell_slack_lower_bound_deep(
        values, cert.grid, cell
    )
    assert np.isfinite(coarse)
    assert np.isfinite(refined)
    assert np.isfinite(deep)
    # Each level bounds the same parent-cell infimum. More subdivision tightens
    # the interval enclosure without weakening the certified lower bound.
    assert refined >= coarse - 1e-9
    assert deep >= refined - 1e-9


def test_zero_deep_refinement_budget_disables_stage_without_exhaustion():
    cert = make_small_certificate(
        iterations=2,
        refinement_budget=32,
        deep_refinement_budget=0,
    )
    cert.build()

    assert cert._epsilon_deep_refined_cell_set == set()
    assert not cert._epsilon_deep_refinement_budget_exhausted


def test_deep_refinement_budget_is_separate_and_audited():
    cert = make_small_certificate(
        iterations=2,
        refinement_budget=64,
        deep_refinement_budget=2,
    )
    cert.build()

    assert len(cert._epsilon_deep_refined_cell_set) <= 2
    assert cert._epsilon_deep_refined_cell_set <= cert._epsilon_refined_cell_set
    for cell in cert._epsilon_deep_refined_cell_set:
        assert cert._epsilon_final_cell_lowers[cell] >= (
            cert._epsilon_coarse_cell_lowers[cell] - 1e-9
        )


def test_deep_refinement_shape_is_configurable_and_tightens_the_bound():
    coarse_shape = make_small_certificate(
        iterations=0, deep_refinement_subdivisions=(4, 4, 2)
    )
    deeper_shape = make_small_certificate(
        iterations=0, deep_refinement_subdivisions=(8, 8, 2)
    )
    cell = (4, 4, 2)
    values = coarse_shape.values[-1]
    coarse_report = coarse_shape.problem.cell_slack_diagnostics(
        values, coarse_shape.grid, cell
    )
    deeper_report = deeper_shape.problem.cell_slack_diagnostics(
        values, deeper_shape.grid, cell
    )

    assert coarse_report["deep"]["subdivisions_per_axis"] == [4, 4, 2]
    assert coarse_report["deep"]["subcell_count"] == 32
    assert deeper_report["deep"]["subdivisions_per_axis"] == [8, 8, 2]
    assert deeper_report["deep"]["subcell_count"] == 128
    assert (
        deeper_report["deep"]["total_lower"]
        >= coarse_report["deep"]["total_lower"] - 1e-9
    )
    assert (
        deeper_report["deep"]["total_lower"]
        >= deeper_report["refined"]["total_lower"] - 1e-9
    )


def test_deep_refinement_shape_rejects_invalid_values():
    base = dict(
        goal=(0.0, 0.0),
        goal_radius=0.5,
        forward_speed=1.0,
        q_position=1.0,
        r_control=0.1,
        control_min=-1.0,
        control_max=1.0,
        dt=0.1,
        n_controls=9,
    )
    for invalid in ((0, 4, 2), (4, 0, 2), (4, 4, 0), (4, 2), (4, 4, 2, 1), (True, 4, 2)):
        with pytest.raises(ValueError, match="deep_refinement_subdivisions"):
            make_paper_unicycle_problem(
                **base, deep_refinement_subdivisions=invalid
            )


def test_radial_extension_uses_nearest_residual_at_blocked_nodes():
    cert = make_small_certificate(
        iterations=2,
        obstacles=((-1.0, -0.5, 0.5, 1.0),),
    )
    cert.build()
    values = cert.values[-1]
    effective = cert._effective_extension_values(values)
    baseline = cert._domain_baseline_nodes.reshape(-1)
    blocked = ~cert._node_admissible.reshape(-1)

    assert np.any(blocked)
    assert np.all(np.isfinite(effective))
    # These obstacle nodes are away from the target, so the radial baseline
    # is strictly positive. A zero surrogate would recreate the artificial
    # baseline jump that dominates the global HJB slack.
    assert np.all(baseline[blocked] > 0.0)
    assert np.all(effective[blocked] > 0.0)
    assert np.allclose(
        effective[cert._node_admissible.reshape(-1)],
        values[cert._node_admissible.reshape(-1)],
    )

    # The exact same extended nodal field is passed to the cellwise slack
    # callback for a partially blocked cell.
    cell = (1, 5, 2)
    lo, hi = cert._cell_bounds(cell)
    assert not cert.problem.cell_is_fully_inadmissible(lo, hi)
    lower = cert.problem.cell_slack_lower_bound(
        effective, cert.grid, cell
    )
    assert np.isfinite(lower)


def test_certified_radial_extension_does_not_pin_obstacle_halo_to_zero():
    cert = make_small_certificate(
        iterations=2,
        obstacles=((-1.0, 1.0, -0.4, 0.4),),
    )
    report = cert.build()
    values = cert.values[-1].reshape(cert.grid.shape)
    radial = cert._domain_baseline_nodes.reshape(cert.grid.shape)
    halo = cert._obstacle_halo_mask & cert._node_admissible.reshape(cert.grid.shape)
    positive_baseline_halo = halo & (radial > 1e-8)

    assert np.any(positive_baseline_halo)
    # The continuous-extension path tapers Bellman residuals but must not
    # overwrite these nodes with zero after tapering. This avoids an artificial
    # zero-vs-radial jump at cells adjacent to the obstacle halo.
    assert np.all(
        values[positive_baseline_halo]
        >= radial[positive_baseline_halo] - 1e-10
    )
    assert np.isfinite(report.epsilon)
    assert np.isfinite(report.epsilon_terminal)


def test_control_lipschitz_bound_scales_running_cost_derivative_by_dt():
    cert = make_small_certificate(iterations=0)
    problem = cert.problem
    values = np.zeros(int(np.prod(cert.grid.shape)), dtype=float)
    bound = problem.control_lipschitz_bound(values, cert.grid)
    expected = problem.dt * 2.0 * 0.1 * 1.0
    assert np.isclose(bound, expected, rtol=0.0, atol=1e-12)


def test_radial_gradient_bound_respects_flat_target_interior():
    problem = make_paper_unicycle_problem(
        goal=(0.0, 0.0),
        goal_radius=0.5,
        forward_speed=1.0,
        q_position=1.0,
        r_control=0.1,
        control_min=-1.0,
        control_max=1.0,
        dt=0.1,
        n_controls=9,
    )
    grid = RegularGrid(
        lower=np.array([-1.0, -1.0, -np.pi]),
        upper=np.array([1.0, 1.0, np.pi]),
        shape=(9, 9, 8),
        periodic_axes=(2,),
    )
    values = problem.terminal_cost(grid.nodes)
    # This cell lies wholly inside the radius-0.5 target, where the clipped
    # radial function and its spatial gradient are exactly zero.
    lower_slack = problem.cell_slack_lower_bound(values, grid, (4, 4, 0))
    assert np.isclose(lower_slack, 0.0, rtol=0.0, atol=1e-12)


def test_interpolation_does_not_renormalize_around_infinite_corners():
    base = make_small_certificate(iterations=0)
    # Exercise the generic fail-closed path, which has no continuous
    # outside-domain extension and therefore falls back to the global bound
    # whenever a positive-weight corner is unavailable.
    problem = replace(
        base.problem,
        outside_domain_lower_bound=None,
        outside_domain_is_subsolution=False,
    )
    cert = SemiLagrangianFOMCertificate(problem, base.grid, iterations=0)
    cert.build()
    values = np.full(cert.grid.shape, 2.0, dtype=float)
    cell = (2, 2, 2)
    values[(3, 3, 3)] = np.inf

    exact_node = cert.nodes[np.ravel_multi_index(cell, cert.grid.shape)]
    assert np.isclose(
        cert._interpolate(values.reshape(-1), exact_node)[0], 2.0
    )

    interior = cert.grid.lower + (
        np.asarray(cell, dtype=float) + 0.5
    ) * cert.grid.spacing
    assert cert._interpolate(values.reshape(-1), interior)[0] == (
        cert.problem.global_value_lower_bound
    )


def test_radial_extension_is_continuous_across_grid_boundary():
    cert = make_small_certificate(iterations=2)
    cert.build()
    inside = np.array([2.0 - 1e-8, 0.31, 0.2])
    outside = np.array([2.0 + 1e-8, 0.31, 0.2])
    value_inside = cert._interpolate(cert.values[-1], inside)[0]
    value_outside = cert._interpolate(cert.values[-1], outside)[0]
    assert abs(value_inside - value_outside) < 1e-5


def test_residual_interpolation_is_continuous_at_obstacle_boundary():
    cert = make_small_certificate(
        iterations=2,
        obstacles=((-0.5, 0.5, -0.5, 0.5),),
    )
    cert.build()
    left = np.array([-0.5 - 1e-8, 0.25, 0.2])
    right = np.array([-0.5 + 1e-8, 0.25, 0.2])
    value_left = cert._interpolate(cert.values[-1], left)[0]
    value_right = cert._interpolate(cert.values[-1], right)[0]
    assert abs(value_left - value_right) < 1e-5


def test_nonfinite_certificate_query_disables_pruning():
    cert = make_small_certificate(
        iterations=1,
        obstacles=((-0.5, 0.5, -0.5, 0.5),),
    )
    report = cert.build()
    blocked = np.array([0.0, 0.0, 0.0])
    assert np.isinf(cert.value(blocked))
    assert cert.corrected_lower_bound(blocked, 1.0) == -np.inf
    # Fully blocked cells are excluded from the HJB domain; adjacent cells
    # use the conservative lower-bound extension and retain finite error gates.
    assert np.isfinite(report.epsilon)
    assert np.isfinite(report.epsilon_terminal)


def test_value_iteration_is_monotone_and_tightens_from_below():
    cert0 = make_small_certificate(iterations=0)
    r0 = cert0.build()
    cert2 = make_small_certificate(iterations=2)
    r2 = cert2.build()

    assert r2.bellman_monotone
    # The raw finite-grid Bellman diagnostic may fail after conservative
    # boundary/obstacle handling; pruning relies on the continuous slack
    # correction, which must remain finite for this fully specified problem.
    assert np.isfinite(r2.epsilon)
    assert np.isfinite(r2.epsilon_terminal)
    assert np.all(
        cert2.values[2][np.isfinite(cert0.values[0])]
        >= cert2.values[0][np.isfinite(cert0.values[0])] - 1e-10
    )
    assert cert2.value(np.array([1.0, 0.0, 0.0])) >= cert0.value(
        np.array([1.0, 0.0, 0.0])
    ) - 1e-10


def test_obstacle_cells_are_fail_closed():
    cert = make_small_certificate(
        iterations=1,
        obstacles=((-0.5, 0.5, -0.5, 0.5),),
    )
    cert.build()
    blocked = cert.problem.admissible(np.array([[0.0, 0.0, 0.0]]))
    assert not blocked[0]
    assert np.isinf(cert.value(np.array([0.0, 0.0, 0.0])))


def test_corrected_lower_bound_accounts_for_certificate_errors():
    cert = make_small_certificate(iterations=2)
    cert.build()
    state = np.array([1.0, 0.0, 0.0])
    raw = cert.value(state)
    corrected = cert.corrected_lower_bound(state, 0.7)
    assert corrected <= raw + 1e-12
    assert np.isfinite(corrected)


def test_terminal_error_covers_soft_terminal_cost_on_full_domain():
    cert = make_small_certificate(iterations=2)
    report = cert.build()

    values = cert.values[-1]
    terminal = cert.problem.terminal_cost(cert.nodes)
    admissible = cert._node_admissible.reshape(-1)
    observed_positive_slack = max(
        0.0,
        float(np.max(values[admissible] - terminal[admissible])),
    )

    # The current primal objective charges the radial terminal cost at every
    # endpoint and does not impose terminal_set as a hard constraint. Thus
    # epsilon_T must cover the full admissible domain, not only the goal set.
    assert report.epsilon_terminal >= observed_positive_slack - 1e-10


def test_batch_certificate_queries_match_scalar_queries():
    cert = make_small_certificate(
        iterations=2,
        obstacles=((-0.5, 0.5, -0.5, 0.5),),
    )
    cert.build()
    states = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],  # outside the grid, using the analytic extension
            [0.0, 0.0, 0.0],  # blocked state must disable pruning
            [1.0, 0.0, 3.13],  # close to the periodic heading seam
        ]
    )
    remaining = np.array([0.1, 0.5, 1.0, 0.25])
    prefix = np.array([1.0, 2.0, 3.0, 4.0])

    batch_values = cert.value_batch(states)
    scalar_values = np.array([cert.value(state) for state in states])
    assert np.allclose(batch_values, scalar_values, rtol=0.0, atol=1e-12)

    batch_lower = cert.corrected_lower_bound_batch(states, remaining)
    scalar_lower = np.array(
        [
            cert.corrected_lower_bound(state, float(time))
            for state, time in zip(states, remaining)
        ]
    )
    assert np.allclose(batch_lower, scalar_lower, rtol=0.0, atol=1e-12)

    batch_prefix = cert.prefix_bound_batch(prefix, states, remaining)
    scalar_prefix = np.array(
        [
            cert.prefix_bound(float(cost), state, float(time))
            for cost, state, time in zip(prefix, states, remaining)
        ]
    )
    assert np.allclose(batch_prefix, scalar_prefix, rtol=0.0, atol=1e-12)
    assert np.isneginf(batch_prefix[2])


def test_cell_slack_neighborhood_reports_boundary_and_halo_metadata():
    cert = make_small_certificate(
        iterations=2,
        obstacles=((-0.5, 0.5, -0.5, 0.5),),
    )
    cert.build()
    rows = cert.cell_slack_neighborhood(center=(3, 3, 3), radius=1)

    assert len(rows) == 27
    center = next(row for row in rows if row["is_center"])
    assert center["cell"] == [3, 3, 3]
    assert center["fully_inadmissible"]
    assert not center["included_in_slack_gate"]
    assert center["slack_lower_bound"] is None
    assert all("boundary_distance_cells" in row for row in rows)
    assert all("touches_obstacle_halo" in row for row in rows)
    assert any(row["included_in_slack_gate"] for row in rows)
    assert all(
        len(row["lower"]) == 3 and len(row["upper"]) == 3
        for row in rows
    )


def test_prefix_pruning_never_uses_an_uncorrected_certificate():
    cert = make_small_certificate(iterations=2)
    cert.build()
    state = np.array([1.0, 0.0, 0.0])
    raw = cert.value(state)
    corrected = cert.corrected_lower_bound(state, 1.0)
    assert corrected <= raw + 1e-12
    assert cert.prefix_bound(3.0, state, 1.0) >= 3.0 + corrected - 1e-12
