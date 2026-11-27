"""Mean-field family: the classical dimer reference line and the lattice engine."""

from .dimer import *
from .lattice import *

__all__ = [
    "MeanFieldLattice",
    "dimer_fixed_points",
    "dimer_jacobian",
    "dimer_lambda",
    "mean_field_flow",
]
