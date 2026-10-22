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

    # Recompute the returned trajectory independently. A pruned prefix must
    # never be reported as a completed Bolza trajectory.
    x = solver.x0.copy()
    independent_cost = 0.0
    for k in range(solver.horizon_steps):
        u = best_control[:, k]
        independent_cost += solver.problem.dt * float(
            solver.problem.running_cost(x[None, :], u[None, :])[0]
        )
        x = x + solver.problem.dt * solver.problem.dynamics(
            x[None, :], u[None, :]
        )[0]
        assert solver.problem.admissible(x[None, :])[0]
    independent_cost += float(
        solver.problem.terminal_cost(x[None, :])[0]
    )
    assert np.isclose(best_cost, independent_cost, rtol=1e-10, atol=1e-10)

    # Reverse triangle inequality gives a model-independent lower bound for
    # this constant-speed unicycle, including the nonnegative terminal cost.
    initial_distance = float(np.linalg.norm(solver.x0[:2]))
    dt = solver.problem.dt
    speed = 1.0
    running_lb = sum(
        dt * max(initial_distance - speed * dt * k, 0.0) ** 2
        for k in range(solver.horizon_steps)
    )
    final_distance_lb = max(
        initial_distance - speed * dt * solver.horizon_steps, 0.0
    )
    terminal_lb = float(
        solver.problem.terminal_cost(
            np.array([[final_distance_lb, 0.0, 0.0]])
        )[0]
    )
    assert best_cost >= running_lb + terminal_lb - 1e-8


def test_certificate_prefix_bound_is_never_above_known_rollout_cost():
    cert = make_certificate(iterations=2)
    problem = cert.problem
    x0 = np.array([1.0, 0.8, 0.0], dtype=float)
    horizon = 8
    states = [x0.copy()]
    stage_costs = []
    x = x0.copy()

    for _ in range(horizon):
        u = np.array([0.0], dtype=float)
        stage_costs.append(
            problem.dt * float(
                problem.running_cost(x[None, :], u[None, :])[0]
            )
        )
        x = x + problem.dt * problem.dynamics(
            x[None, :], u[None, :]
        )[0]
        assert problem.admissible(x[None, :])[0]
        states.append(x.copy())

    full_cost = sum(stage_costs) + float(
        problem.terminal_cost(states[-1][None, :])[0]
    )
    prefix = 0.0
    for k, stage in enumerate(stage_costs):
        prefix += stage
        bound = cert.prefix_bound(
            prefix,
            states[k + 1],
            problem.dt * (horizon - k - 1),
        )
        assert bound <= full_cost + 1e-7


def test_corrected_prefix_bound_stays_below_random_feasible_rollout_costs():
    cert = make_certificate(iterations=2)
    problem = cert.problem
    rng = np.random.default_rng(20261019)
    x0 = np.array([1.0, 0.8, 0.0])
    horizon = 8
    checked_paths = 0

    for _ in range(64):
        controls = rng.uniform(-1.0, 1.0, size=horizon)
        states = [x0.copy()]
        stages = []
        x = x0.copy()
        feasible = True
        for control in controls:
            stages.append(
                problem.dt
                * float(
                    problem.running_cost(
                        x[None, :], np.array([[control]])
                    )[0]
                )
            )
            x = x + problem.dt * problem.dynamics(
                x[None, :], np.array([[control]])
            )[0]
            if not problem.admissible(x[None, :])[0]:
                feasible = False
                break
            states.append(x.copy())

        if not feasible:
            continue

        total_cost = sum(stages) + float(
            problem.terminal_cost(states[-1][None, :])[0]
        )
        prefix = 0.0
        for k, stage in enumerate(stages):
            prefix += stage
            bound = cert.prefix_bound(
                prefix,
                states[k + 1],
                problem.dt * (horizon - k - 1),
            )
            assert bound <= total_cost + 1e-7
        checked_paths += 1

    assert checked_paths >= 8


def test_pruned_prefixes_are_not_scored_or_used_as_elites():
    cert = make_certificate(iterations=1)
    solver = CertificateGuidedIGOSolver(
        cert,
        x0=np.array([1.5, -1.0, 0.0]),
        horizon_steps=8,
        n_samples=4,
        n_iterations=1,
        block_steps=4,
        elite_quantile=0.5,
        initial_std=0.25,
        random_seed=11,
    )
    # Force the pruning branch to isolate the rollout bookkeeping invariant.
    solver.incumbent_cost = -1.0e9
    solver.certificate.prefix_bound_batch = (
        lambda prefix, states, remaining_time: np.ones(states.shape[0])
    )
    controls = np.zeros((4, 1, 8), dtype=float)

    _, costs, pruned, infeasible_count, completed_count = solver._rollout_batch(
        controls
    )
    assert np.all(pruned)
    assert infeasible_count == 0
    assert completed_count == 0
    assert np.all(np.isinf(costs))

    mean_before = solver.mean.copy()
    factors_before = [factor.copy() for factor in solver.factors]
    selected = solver._update_distribution(
        np.zeros((4, 1, 8), dtype=float), costs
    )
    assert selected == -1
    assert np.array_equal(solver.mean, mean_before)
    assert all(
        np.array_equal(before, after)
        for before, after in zip(factors_before, solver.factors)
    )
