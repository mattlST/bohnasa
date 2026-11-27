"""Classical mean-field limit of the interacting boson model.

The mean-field counterpart of the lattice Bose-Hubbard model is a discrete
nonlinear Schrodinger flow on coherent-state amplitudes. This module holds
the closed-form dimer (L = 2) results - fixed points, stability exponent,
flow - which serve as the exact reference line for the quantum growth
rates of :class:`bohnasa.SquaredCommutator`; the general L-site engine,
of which the dimer is the L = 2 special case, is
:class:`bohnasa.MeanFieldLattice` in :mod:`bohnasa.mean_field.lattice`.

The family conventions (phase space, quarter-circle coupling, shell
normalization, canonical brackets, sample times) are stated once in
:mod:`bohnasa.mean_field`. Dimer-scoped, ``L = 2`` only: the two-mode
canonical pair ``{phi, s} = 2/N`` (both modes contribute) sets the factor 2
in the flow equations, and the quarter-circle coupling is the ``L = 2``
case ``U = 2 sin(theta)/N`` of the general ``U = sin(theta) * L / N_phys``
rule, which makes the classical flow ``N``-independent.

Closed forms and the pitchfork structure follow the doctoral thesis cited in
``CITATION.cff`` (quarter-circle system, off-phase saddle).
"""

from typing import Literal

import numpy as np

from .solver import (
    _DEFAULT_ATOL,
    _DEFAULT_RTOL,
    check_sample_times,
    check_solver_knobs,
    solve_samples,
)
from .solver import INTEGRATION_METHODS as _INTEGRATION_METHODS  # noqa: F401  (re-export)

__all__ = [
    "dimer_fixed_points",
    "dimer_jacobian",
    "dimer_lambda",
    "mean_field_flow",
]

# Numerical settings policy (issue #25): every knob is a named constant with
# its measured regime; see CONTRIBUTING.md "Numerical settings policy".
#
# sqrt guard for the flow and the Jacobian. In float64 the clip is active
# only at |s| == 1 exactly (the nearest interior double gives 1 - s*s =
# 2.2e-16, far above the floor); it turns the wall singularity from inf into
# a finite but astronomically stiff value (measured dphi = 7.7e149 at
# s = 1, phi = 0.5 vs inf without it). Real bounded orbits stay away from the
# wall by >= 5.5e-2 (measured s0 = 0.9 orbit over t in [0, 20]), so the floor
# never perturbs the physics anchors.
_ROOT_FLOOR = 1e-300
# Realness tolerance for the initial pair (s, phi): accepts complex input
# carrying only round-off imaginary parts (measured <= 2.8e-17 for FFT
# round-trips) and rejects physical-scale ones; a 1e-3j initial is pinned
# invalid by tests/unit/test_otoc.py::test_mean_field_flow_rejects_complex_initial.
_REAL_INPUT_TOL = 1e-12


def _mf_fields(s: float, phi: float, theta: float) -> tuple[float, float]:
    """Right-hand side of the dimer flow ds = -2 dh/dphi, dphi = +2 dh/ds
    for h(s, phi) = -2 cos(theta) sqrt(1-s^2) cos(phi)
    + (sin(theta)/2) (1+s^2)."""
    c, u = np.cos(theta), np.sin(theta)
    root = np.sqrt(np.clip(1.0 - s * s, _ROOT_FLOOR, None))
    ds = -4.0 * c * root * np.sin(phi)
    dphi = 4.0 * c * s * np.cos(phi) / root + 2.0 * u * s
    return ds, dphi


def dimer_lambda(theta: float) -> float:
    """Stability exponent of the (0, pi) mean-field saddle, 0 if stable.

    Examples
    --------
    >>> from bohnasa import dimer_lambda
    >>> round(dimer_lambda(1.35), 4)
    0.9706
    """
    # mu^2 = 8 cos^2 (tan - 2): instability window is literally tan > 2,
    # and cos^2 = 1/(1 + tan^2)
    t = np.tan(theta)
    mu2 = 8.0 * (t - 2.0) / (1.0 + t * t)
    return float(np.sqrt(mu2)) if mu2 > 0.0 else 0.0


def dimer_fixed_points(theta: float) -> np.ndarray:
    """Mean-field fixed points as rows ``(s, phi, lambda)``; ``lambda > 0``
    marks saddles, ``lambda = 0`` centers.

    For ``tan(theta) > 2`` the point (0, pi) becomes a saddle and a
    symmetry-broken pair ``+-sqrt(1 - 4 cos^2/sin^2)`` is born
    (pitchfork at ``theta = arctan(2)``, see :func:`dimer_lambda`).

    Examples
    --------
    >>> from bohnasa import dimer_fixed_points
    >>> len(dimer_fixed_points(1.0))   # below the pitchfork: two points
    2
    >>> len(dimer_fixed_points(1.35))  # above: saddle, pair, and the axis
    4
    """
    c, u = np.cos(theta), np.sin(theta)
    points = [(0.0, 0.0, 0.0)]
    points.append((0.0, np.pi, dimer_lambda(theta)))
    ratio = 4.0 * c * c / (u * u) if u != 0.0 else np.inf
    if ratio < 1.0:
        s_star = float(np.sqrt(1.0 - ratio))
        points.append((s_star, np.pi, 0.0))
        points.append((-s_star, np.pi, 0.0))
    return np.array(points)


def dimer_jacobian(theta: float, s: float, phi: float) -> np.ndarray:
    """2x2 Jacobian of the mean-field flow at (s, phi).

    Examples
    --------
    >>> from bohnasa import dimer_jacobian
    >>> float(dimer_jacobian(1.35, 0.0, 3.141592653589793).trace())
    0.0
    """
    c, u = np.cos(theta), np.sin(theta)
    root = np.sqrt(max(1.0 - s * s, _ROOT_FLOOR))
    return np.array(
        [
            [4.0 * c * s * np.sin(phi) / root, -4.0 * c * root * np.cos(phi)],
            [
                4.0 * c * np.cos(phi) * (1.0 / root + s * s / root**3) + 2.0 * u,
                -4.0 * c * s * np.sin(phi) / root,
            ],
        ]
    )


def mean_field_flow(
    theta: float,
    initial: tuple[float, float],
    times: np.ndarray,
    *,
    method: Literal["DOP853", "RK23", "RK45", "Radau", "BDF", "LSODA"] = "DOP853",
    rtol: float = _DEFAULT_RTOL,
    atol: float = _DEFAULT_ATOL,
) -> np.ndarray:
    """Integrate the classical dimer flow with an adaptive solver.

    Parameters
    ----------
    theta : float
        Quarter-circle coupling angle.
    initial : tuple of float
        Starting ``(s, phi)``.
    times : numpy.ndarray
        Sample times (need not be sorted; duplicates allowed). The flow
        starts at ``max(0, min(times))`` with the initial pair at that time;
        times at or below it report the initial state.
    method : str, keyword-only, default "DOP853"
        Integration method, one of ``"DOP853"``, ``"RK23"``, ``"RK45"``,
        ``"Radau"``, ``"BDF"``, ``"LSODA"`` (handed to ``solve_ivp``).
        DOP853 is the anchor-pinned reference of issue #24; at the default
        tolerances all listed methods agree with it to better than 5e-8 on
        the bounded orbit and saddle probes (measured).
    rtol : float, keyword-only, default 1e-10
        Relative integration tolerance handed to ``solve_ivp``. The default
        is the accuracy point pinned by the physics anchors of issue #24;
        callers trading accuracy for speed may loosen it.
    atol : float, keyword-only, default 1e-12
        Absolute integration tolerance (same rationale).

    Returns
    -------
    numpy.ndarray
        Trajectory at ``times``, shape ``(len(times), 2)``. The independent
        numerical reference for :func:`dimer_fixed_points` and
        :func:`dimer_jacobian`.

    Raises
    ------
    ValueError
        If ``times`` is empty, not 1D, or non-finite; if ``initial`` is not
        a real pair; if ``method`` is not one of the listed integrators; or
        if ``rtol``/``atol`` are not positive and finite.

    Examples
    --------
    >>> import numpy as np
    >>> from bohnasa import mean_field_flow
    >>> traj = mean_field_flow(1.35, (0.0, np.pi), np.array([0.0, 5.0]))
    >>> float(np.abs(traj - np.array([0.0, np.pi])).max()) < 1e-12
    True
    >>> other = mean_field_flow(1.35, (0.0, np.pi), np.array([0.0, 5.0]),
    ...                         method="RK45")
    >>> float(np.abs(other - traj).max()) < 1e-6
    True
    """
    times = check_sample_times(times, prefix="mean_field_flow")
    check_solver_knobs(method, rtol, atol, prefix="mean_field_flow")
    y0 = np.asarray(initial, dtype=complex).ravel()
    if y0.size != 2 or np.abs(y0.imag).max() > _REAL_INPUT_TOL:
        raise ValueError("mean_field_flow: initial must be a real pair (s, phi)")
    s0, phi0 = float(y0[0].real), float(y0[1].real)
    start = max(0.0, float(times.min()))
    order = np.argsort(times)
    targets = times[order]
    out = np.empty((len(times), 2))
    future = targets > start
    out[order[~future]] = (s0, phi0)
    if np.any(future):
        # DOP853 (issue #24): adaptive high-order integration of the smooth
        # 2D flow. The default tolerances are the anchor-pinned accuracy
        # point; rtol/atol are caller-adjustable for speed/accuracy trades.
        # solve_ivp rejects duplicate t_eval entries, so the sorted targets
        # are deduplicated and mapped back with np.unique's inverse index.
        unique, inverse = np.unique(targets[future], return_inverse=True)

        def rhs(_t: float, y: np.ndarray) -> np.ndarray:
            """``solve_ivp`` right-hand side ``f(t, y)`` (autonomous flow;
            explicit signature for the strict scipy-stubs on py3.10/3.11)."""
            ds, dphi = _mf_fields(float(y[0]), float(y[1]), theta)
            return np.array([ds, dphi])

        samples = solve_samples(
            rhs,
            np.array([s0, phi0]),
            (start, float(targets[-1])),
            unique,
            method=method,
            rtol=rtol,
            atol=atol,
            prefix="mean_field_flow",
        )
        out[order[future]] = samples[inverse]
    return out
