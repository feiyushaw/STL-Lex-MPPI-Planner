import numpy as np

from utils.fom_igo_solver import FOMIGOSolver
from utils.legacy_mppi_solver import LegacyMPPISolver
from utils.simulation_environment_1d import System


class Cost:
    time = 1
    nof_intervals = 2
    direction = "pos"
    lb = 0.0
    ub = 0.0
    intervals = [((-10.0, 0.0),), ((0.0, 10.0),)]

    def evaluate_trajectory(self, y):
        return int(float(y[0, self.time]) > 0.0)


def test_legacy_mppi_and_igo_share_budget_shape():
    system = System(dt=1.0, u_min=-2.0, u_max=2.0)
    z = np.zeros((2, 4, 1, 2))
    solver = LegacyMPPISolver(
        [Cost()], 0.0, 3, system, 2, 4, 1.0, 1.0, 0.6, True, z
    )
    result = solver.solve()
    assert result["total_rollouts"] == 8
    assert result["evaluated_transitions"] == 16


def test_igo_diagnostics_without_pruning():
    system = System(dt=1.0, u_min=-2.0, u_max=2.0)
    solver = FOMIGOSolver(
        [Cost()], x0=0.0, K=3, system=system,
        n_iterations=1, n_samples_constant=4,
        cov=np.array([[1.0]]), random_seed=1,
        verbose=False, enable_certificate_pruning=False,
    )
    solver.solve()
    diagnostics = solver.get_diagnostics()
    assert diagnostics["total_rollouts"] == 4
    assert diagnostics["pruned_rollouts"] == 0
    assert diagnostics["evaluated_transitions"] == 8
    assert diagnostics["transition_effective_rate"] == 1.0
