from dataclasses import replace

import numpy as np

from utils.fom_certificate import (
    RegularGrid,
    SemiLagrangianFOMCertificate,
    make_paper_unicycle_problem,
    radial_terminal_cost,
)


def make_small_certificate(iterations=0, obstacles=()):
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
