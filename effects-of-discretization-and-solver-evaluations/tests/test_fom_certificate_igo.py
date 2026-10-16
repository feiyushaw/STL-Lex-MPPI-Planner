import numpy as np

from utils.fom_certificate import (
    RegularGrid,
    SemiLagrangianFOMCertificate,
    make_paper_unicycle_problem,
)
from utils.fom_certificate_igo import CertificateGuidedIGOSolver


def make_certificate(iterations=2):
    problem = make_paper_unicycle_problem(
        goal=(0.0, 0.0),
        goal_radius=0.5,
        forward_speed=1.0,
        q_position=1.0,
        r_control=0.1,
        control_min=-1.0,
        control_max=1.0,
        dt=0.1,
        obstacles=((-0.2, 0.2, -1.0, 1.0),),
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
    cert.build()
    return cert


def test_certificate_guided_igo_has_finite_feasible_incumbent():
    cert = make_certificate(iterations=1)
    solver = CertificateGuidedIGOSolver(
        cert,
        x0=np.array([1.5, -1.0, 0.0]),
        horizon_steps=8,
        n_samples=32,
        n_iterations=3,
        block_steps=4,
        elite_quantile=0.25,
        initial_std=0.25,
        random_seed=3,
    )
    best_control, best_cost, reports = solver.solve()

    assert best_control is not None
    assert np.isfinite(best_cost)
    assert len(reports) == 3
    assert all(np.isfinite(r.incumbent_cost) for r in reports)
    assert solver.pruned_rollouts_total >= 0


def test_certificate_prefix_bound_is_never_above_known_rollout_cost():
    cert = make_certificate(iterations=2)
    problem = cert.problem
    x = np.array([1.0, 0.8, 0.0], dtype=float)
    total = 0.0
    horizon = 8

    for k in range(horizon):
        u = np.array([0.0], dtype=float)
        total += problem.dt * float(
            problem.running_cost(x[None, :], u[None, :])[0]
        )
        x = x + problem.dt * problem.dynamics(
            x[None, :], u[None, :]
        )[0]
        bound = cert.prefix_bound(
            total,
            x,
            problem.dt * (horizon - k - 1),
        )
        terminal = float(problem.terminal_cost(x[None, :])[0])
        # The certificate is a lower bound on every feasible completion.
        assert bound <= total + terminal + 1e-7
