"""The mean-field family: classical dimer reference line and lattice engine.

This package gathers the classical mean-field machinery of the
Bose-Hubbard model: the closed-form dimer (:mod:`bohnasa.mean_field.dimer`,
``L = 2``) and the general lattice engine
(:mod:`bohnasa.mean_field.lattice`, arbitrary ``L``), sharing the adaptive
integration plumbing of :mod:`bohnasa.mean_field.solver`. The closed-form
dimer stays an independent implementation: it is the analytic cross-check
of the lattice engine's ``L = 2`` reduction, not a wrapper around it.

Conventions are stated once here and pinned by the tests. They follow the
doctoral thesis cited in ``CITATION.cff`` (Sec. 10.1 and App. A.1).

Phase space.
    A state is the real vector ``z = (q_0, ..., q_{L-1}, p_0, ...,
    p_{L-1})`` of length ``2L``; the mean-field amplitudes are
    ``phi_j = (q_j + i p_j) / sqrt(2)`` with occupations
    ``n_j = |phi_j|^2 = (q_j^2 + p_j^2) / 2``.

Quarter-circle coupling.
    With ``J = cos(theta)`` and ``U = sin(theta) * L / N_phys`` in the
    standard-form quantum model
    ``H = -J sum_<jk> (c^dag_j c_k + h.c.) + (U/2) sum_j n_j (n_j - 1)``
    (``Hamiltonian`` translation: ``onsite_u = U / 2``), the flow written
    for density-scaled amplitudes ``phi_j = sqrt(L / N_phys) * alpha_j`` is
    independent of the total particle number ``N_phys``. The engine fixes
    the gauge ``N_phys = L`` (unit filling), where ``U = sin(theta)``. The
    dimer formula ``U = 2 sin(theta) / N`` is the ``L = 2`` special case of
    this rule only.

Shell normalization.
    The gauge is a fixed occupation shell ``sum_j n_j = L``.

Canonical brackets.
    ``{q_j, p_k} = delta_jk`` with the symplectic matrix
    ``Omega = [[0, I], [-I, 0]]`` on the ``(q, p)`` layout, so the flow is
    ``dz/dt = Omega grad H`` (equivalently
    ``i dphi_j/dt = dH / dconjugate(phi_j)``, Gross-Pitaevskii / DNLS).
    In the dimer's reduced variables ``s = (n_0 - n_1) / N`` and
    ``phi = arg(phi_1) - arg(phi_0)`` the bracket is ``{phi, s} = 2 / N``
    (the ``L = 2`` special case).

Sample times.
    Both engines integrate the autonomous flow from
    ``start = max(0, min(times))`` with the state at ``start`` equal to the
    initial state. ``times`` need not be sorted and may repeat (rows are
    restored to the caller's order), and times at or below ``start`` report
    the initial state. The Lyapunov renormalization grid is a different
    contract and stays strictly increasing.
"""

from .dimer import *
from .lattice import *

__all__ = [
    "MeanFieldLattice",
    "dimer_fixed_points",
    "dimer_jacobian",
    "dimer_lambda",
    "mean_field_flow",
]
