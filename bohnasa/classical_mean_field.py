"""Classical mean-field limit of the interacting boson model.

The mean-field counterpart of the lattice Bose-Hubbard model is a discrete
nonlinear Schrodinger flow on coherent-state amplitudes. This module holds
the closed-form dimer (L = 2) results - fixed points, stability exponent,
flow - which serve as the exact reference line for the quantum growth
rates of :class:`bohnasa.SquaredCommutator`; the general L-site engine,
of which the dimer is the L = 2 special case, is
:class:`bohnasa.MeanFieldLattice` in :mod:`bohnasa.mean_field_lattice`.

Dimer convention: quarter-circle coupling ``J = cos(theta)``,
``U = 2 sin(theta)/N``, so the classical flow is ``N``-independent. The
two-mode canonical pair ``{phi, s} = 2/N`` (both modes contribute) sets the
factor 2 in the flow equations.

Convention, closed forms and the pitchfork structure follow the doctoral
thesis cited in ``CITATION.cff`` (quarter-circle system, off-phase saddle).
"""

import numpy as np
from scipy.integrate import solve_ivp

__all__ = [
    "dimer_fixed_points",
    "dimer_jacobian",
    "dimer_lambda",
    "mean_field_flow",
]


def _mf_fields(s: float, phi: float, theta: float) -> tuple[float, float]:
    """Right-hand side of the dimer flow ds = -2 dh/dphi, dphi = +2 dh/ds
    for h(s, phi) = -2 cos(theta) sqrt(1-s^2) cos(phi)
    + (sin(theta)/2) (1+s^2)."""
    c, u = np.cos(theta), np.sin(theta)
    root = np.sqrt(np.clip(1.0 - s * s, 1e-300, None))
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
    root = np.sqrt(max(1.0 - s * s, 1e-300))
    return np.array(
        [
            [4.0 * c * s * np.sin(phi) / root, -4.0 * c * root * np.cos(phi)],
            [
                4.0 * c * np.cos(phi) * (1.0 / root + s * s / root**3) + 2.0 * u,
                -4.0 * c * s * np.sin(phi) / root,
            ],
        ]
    )


def mean_field_flow(theta: float, initial: tuple[float, float], times: np.ndarray) -> np.ndarray:
    """Integrate the classical dimer flow with an adaptive DOP853.

    Parameters
    ----------
    theta : float
        Quarter-circle coupling angle.
    initial : tuple of float
        Starting ``(s, phi)``.
    times : numpy.ndarray
        Sample times (need not be sorted; duplicates allowed). Times at or
        below zero report the initial state: the flow starts at ``t = 0``.

    Returns
    -------
    numpy.ndarray
        Trajectory at ``times``, shape ``(len(times), 2)``. The independent
        numerical reference for :func:`dimer_fixed_points` and
        :func:`dimer_jacobian`.

    Examples
    --------
    >>> import numpy as np
    >>> from bohnasa import mean_field_flow
    >>> traj = mean_field_flow(1.35, (0.0, np.pi), np.array([0.0, 5.0]))
    >>> float(np.abs(traj - np.array([0.0, np.pi])).max()) < 1e-12
    True
    """
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("mean_field_flow: times must be a non-empty 1D array")
    if not np.all(np.isfinite(times)):
        raise ValueError("mean_field_flow: times must be finite")
    y0 = np.asarray(initial, dtype=complex).ravel()
    if y0.size != 2 or np.abs(y0.imag).max() > 1e-12:
        raise ValueError("mean_field_flow: initial must be a real pair (s, phi)")
    s0, phi0 = float(y0[0].real), float(y0[1].real)
    order = np.argsort(times)
    targets = times[order]
    out = np.empty((len(times), 2))
    future = targets > 0.0
    out[order[~future]] = (s0, phi0)
    if np.any(future):
        # DOP853 (issue #24): adaptive high-order integration of the smooth
        # 2D flow, with tolerances fixed by the issue. solve_ivp rejects
        # duplicate t_eval entries, so the sorted targets are deduplicated
        # and mapped back with np.unique's inverse index.
        unique, inverse = np.unique(targets[future], return_inverse=True)

        def rhs(_t: float, y: np.ndarray) -> np.ndarray:
            """``solve_ivp`` right-hand side ``f(t, y)`` (autonomous flow;
            explicit signature for the strict scipy-stubs on py3.10/3.11)."""
            ds, dphi = _mf_fields(float(y[0]), float(y[1]), theta)
            return np.array([ds, dphi])

        sol = solve_ivp(
            rhs,
            (0.0, float(targets[-1])),
            np.array([s0, phi0]),
            t_eval=unique,
            method="DOP853",
            rtol=1e-10,
            atol=1e-12,
        )
        if not sol.success:
            raise RuntimeError(f"mean_field_flow: integration failed: {sol.message}")
        out[order[future]] = sol.y.T[inverse]
    return out
