"""
FOM Certificate-Guided Block-Diagonal xNES (IGO) Solver for STL-Lex-MPPI-Planner.
Replaces the Shrinking-beta MPPI solver in `utils/mppi_solver.py` based on:
Wei et al., "Featurized Occupation Measures for Structured Global Search in Numerical Optimal Control" (arXiv:2603.16231v3).
"""

from .fom_igo_solver import FOMIGOSolver, MPPISolver

__all__ = ["FOMIGOSolver", "MPPISolver"]
