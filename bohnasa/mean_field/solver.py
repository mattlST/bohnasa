"""Shared adaptive-integration plumbing for the mean-field family.

The closed-form dimer (:mod:`bohnasa.mean_field.dimer`) and the general
lattice engine (:mod:`bohnasa.mean_field.lattice`) drive the same
``scipy.integrate.solve_ivp`` machinery with the same allowlist and message
prefixes. This module is the single home of that plumbing; it carries no
physics - the right-hand sides, Jacobians and state gauges stay with their
engines.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

__all__ = [
    "INTEGRATION_METHODS",
    "check_sample_times",
    "check_solver_knobs",
    "solve_samples",
]

# Allowlisted solve_ivp schemes (issue #24 review rounds). Explicit
# Runge-Kutta schemes ignore (and warn about) ``jac``, so the Jacobian is
# attached only to the implicit ones.
INTEGRATION_METHODS = ("DOP853", "RK23", "RK45", "Radau", "BDF", "LSODA")
_EXPLICIT_METHODS = ("RK23", "RK45", "DOP853")

# Default solve_ivp tolerances of the dimer flow: the accuracy point the
# issue #24 anchors were measured at. Pinned references: cross-method
# agreement <= 6.2e-8 and the monotone rtol-ladder deviations 4.22e-9 ...
# 4.78e-3 (tests/unit/test_mean_field_flow.py), and the
# ODE-perturbation-slope floor 3.34e-4, bit-identical across the CI scipy
# matrix (tests/physics/test_otoc_physics.py). Loosen both only with a
# re-measured anchor; they are caller-adjustable keyword arguments for
# speed/accuracy trades. The lattice engine pins its own defaults.
_DEFAULT_RTOL = 1e-10
_DEFAULT_ATOL = 1e-12


def check_sample_times(times: Any, *, prefix: str) -> np.ndarray:
    """Validate sample times and return them as a float array.

    A non-empty finite 1D array is required; sortedness and duplicates are
    the caller's semantics, not this check's.
    """
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError(f"{prefix}: times must be a non-empty 1D array")
    if not np.all(np.isfinite(times)):
        raise ValueError(f"{prefix}: times must be finite")
    return times


def check_solver_knobs(method: str, rtol: float, atol: float, *, prefix: str) -> None:
    """Validate the solver allowlist and tolerances with a unified prefix."""
    if method not in INTEGRATION_METHODS:
        raise ValueError(f"{prefix}: method must be one of {INTEGRATION_METHODS}")
    if not (np.isfinite(rtol) and rtol > 0.0 and np.isfinite(atol) and atol > 0.0):
        raise ValueError(f"{prefix}: rtol and atol must be positive and finite")


def solve_samples(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    y0: np.ndarray,
    t_span: tuple[float, float],
    targets: np.ndarray,
    *,
    method: str,
    rtol: float,
    atol: float,
    jac: Callable[[float, np.ndarray], np.ndarray] | None = None,
    prefix: str,
) -> np.ndarray:
    """Integrate ``rhs`` and return the states at ``targets`` as rows.

    ``targets`` must already be sorted and duplicate-free (``solve_ivp``
    rejects duplicate ``t_eval`` entries). The Jacobian is attached only for
    implicit schemes; explicit Runge-Kutta methods would ignore it.

    Raises
    ------
    RuntimeError
        If ``solve_ivp`` reports an unsuccessful integration.
    """
    # scipy-stubs types ``method`` as an exhaustive Literal; the family
    # deliberately accepts any allowlisted solve_ivp scheme name, so the
    # options are passed through an Any-typed mapping.
    solver_options: dict[str, Any] = {"method": method}
    if jac is not None and method not in _EXPLICIT_METHODS:
        solver_options["jac"] = jac
    sol = solve_ivp(
        rhs,
        t_span,
        y0,
        t_eval=targets,
        rtol=rtol,
        atol=atol,
        **solver_options,
    )
    if not sol.success:
        raise RuntimeError(f"{prefix}: integration failed: {sol.message}")
    return sol.y.T
