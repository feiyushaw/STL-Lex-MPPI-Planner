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


def test_interpolation_does_not_renormalize_around_infinite_corners():
    cert = make_small_certificate(iterations=0)
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


def test_nonfinite_certificate_query_disables_pruning():
    cert = make_small_certificate(
        iterations=1,
        obstacles=((-0.5, 0.5, -0.5, 0.5),),
    )
    cert.build()
    blocked = np.array([0.0, 0.0, 0.0])
    assert np.isinf(cert.value(blocked))
    assert cert.corrected_lower_bound(blocked, 1.0) == -np.inf


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


def test_prefix_pruning_never_uses_an_uncorrected_certificate():
    cert = make_small_certificate(iterations=2)
    cert.build()
    state = np.array([1.0, 0.0, 0.0])
    raw = cert.value(state)
    corrected = cert.corrected_lower_bound(state, 1.0)
    assert corrected <= raw + 1e-12
    assert cert.prefix_bound(3.0, state, 1.0) >= 3.0 + corrected - 1e-12
