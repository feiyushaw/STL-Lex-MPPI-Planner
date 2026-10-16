"""
Backward-compatible import surface for the planner optimizer.

MPPISolver is retained as an import name only. Its implementation is
FOMIGOSolver (IGO/xNES); MPPI weighting is no longer present.
"""

from .fom_igo_solver import FOMIGOSolver

MPPISolver = FOMIGOSolver

__all__ = ["FOMIGOSolver", "MPPISolver"]
