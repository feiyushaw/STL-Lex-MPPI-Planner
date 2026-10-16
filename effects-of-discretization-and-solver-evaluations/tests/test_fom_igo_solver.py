import numpy as np

from utils.fom_igo_solver import FOMIGOSolver


class DummySystem:
    m = 1
    n = 1
    p = 1
    dt = 1.0
    u_min = -2.0
    u_max = 2.0

    def f(self, x, u):
        return np.asarray(x) + np.asarray(u)

    def g(self, x, u):
        return np.asarray(x)


class DummyCost:
    def __init__(self, time):
        self.time = time
        self.direction = "pos"
        self.lb = 0.0
        self.ub = 0.0
        self.intervals = [
            ((-10.0, 0.0),),
            ((0.0, 10.0),),
        ]
        self.nof_intervals = 2

    def evaluate_trajectory(self, y):
        return int(float(y[0, self.time]) > 0.0)


def make_solver(**kwargs):
    return FOMIGOSolver(
        [DummyCost(time=2)],
        x0=0.0,
        K=3,
        system=DummySystem(),
        n_iterations=1,
        n_samples_constant=4,
        cov=np.array([[1.0]]),
        random_seed=7,
        verbose=False,
        enable_certificate_pruning=False,
        **kwargs,
    )


def test_pregenerated_samples_are_whitened_latents():
    samples = np.zeros((1, 4, 1, 3), dtype=float)
    samples[0, :, 0, :] = np.array(
        [
            [-2.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    solver = make_solver(pregenerated_eps=samples)
    assert np.array_equal(solver._latent_samples(0, 4), samples[0])


def test_elite_rank_update_moves_mean_toward_better_samples():
    samples = np.zeros((1, 4, 1, 3), dtype=float)
    samples[0, :, 0, 0] = np.array([-2.0, -1.0, 1.0, 2.0])

    solver = make_solver(
        pregenerated_eps=samples,
        elite_quantile=0.5,
        lr_mean=1.0,
        lr_cov=0.0,
    )
    u, _, _, _, _, _ = solver.fom_igo_step(
        0, np.zeros((1, 3)), solver._init_block_factors(), 4
    )

    assert u[0, 0] < 0.0


def test_out_of_range_interval_never_becomes_zero_violation():
    solver = make_solver()
    value = solver._vectorized_interval_index(
        np.array([100.0]),
        np.array([-10.0, 0.0]),
        np.array([0.0, 10.0]),
    )
    assert value[0] == 1.0


def test_legacy_mppi_name_points_to_igo_backend():
    from utils.mppi_solver import MPPISolver

    assert MPPISolver is FOMIGOSolver


def test_partial_block_update_uses_endpoint_width():
    samples = np.zeros((1, 4, 1, 7), dtype=float)
    samples[0, 0, 0, :] = -1.0
    samples[0, 1, 0, :] = -0.5

    solver = FOMIGOSolver(
        [DummyCost(time=2)],
        x0=0.0,
        K=7,
        system=DummySystem(),
        n_iterations=1,
        n_samples_constant=4,
        cov=np.array([[1.0]]),
        random_seed=7,
        verbose=False,
        enable_certificate_pruning=False,
        pregenerated_eps=samples,
        elite_quantile=0.5,
        block_steps=3,
        lr_mean=1.0,
        lr_cov=0.0,
    )

    u, A_blocks, _, _, _, _ = solver.fom_igo_step(
        0, np.zeros((1, 7)), solver._init_block_factors(), 4
    )

    assert [a.shape for a in A_blocks] == [(3, 3), (3, 3), (1, 1)]
    assert np.all(np.isfinite(u))
