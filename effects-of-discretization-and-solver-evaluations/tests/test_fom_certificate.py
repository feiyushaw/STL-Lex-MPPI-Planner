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
    assert np.allclose(value, [16.0 / 3.0, 0.0])


def test_k0_certificate_has_zero_feasibility_error():
    cert = make_small_certificate(iterations=0)
    report = cert.build()
    assert report.iterations == 0
    assert report.epsilon_terminal == 0.0
    assert report.epsilon <= 1e-10
    assert report.discrete_subsolution
    assert report.bellman_monotone


def test_value_iteration_is_monotone_and_tightens_from_below():
    cert0 = make_small_certificate(iterations=0)
    r0 = cert0.build()
    cert2 = make_small_certificate(iterations=2)
    r2 = cert2.build()

    assert r2.bellman_monotone
    assert r2.discrete_subsolution
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
