"""General L-site mean-field lattice engine (discrete nonlinear Schrodinger).

This module provides the classical mean-field flow of the lattice
Bose-Hubbard model for arbitrary ``L``; the closed-form dimer in
:mod:`bohnasa.classical_mean_field` is the ``L = 2`` special case, pinned by
reduction tests. Conventions below are stated once here and follow the
doctoral thesis cited in ``CITATION.cff`` (Sec. 10.1 and App. A.1).

Phase space. A state is the real vector ``z = (q_0, ..., q_{L-1}, p_0, ...,
p_{L-1})`` of length ``2L``; the mean-field amplitudes are
``phi_j = (q_j + i p_j) / sqrt(2)`` with occupations
``n_j = |phi_j|^2 = (q_j^2 + p_j^2) / 2``.

Coupling. Quarter-circle parameterization ``J = cos(theta)``,
``U = 2 sin(theta) / N``, so the flow written for density-scaled amplitudes
``phi_j * sqrt(L / N)`` is independent of the total particle number ``N``;
the norm is then a gauge choice and this engine fixes unit filling ``N = L``
(occupation shell ``sum_j n_j = L``). With ``c = cos(theta)`` and
``u = sin(theta)`` the Hamiltonian is

    H = -c * sum_j (q_j q_{j+1} + p_j p_{j+1})
        + (u / 2) * sum_j n_j^2
        - (mu / 2) * sum_j (q_j^2 + p_j^2),

where ``mu`` acts as a chemical-potential-like shift. The neighbour sums are
roll sums modulo ``L`` for ``connections="pbc"`` (default) and clipped at the
chain ends for ``connections="obc"``. A two-site PBC ring visits its single
bond from both directions, so the bond counts twice; that is precisely the
dimer convention of :mod:`bohnasa.classical_mean_field`, whose reduced
variables ``s = (n_0 - n_1) / N``, ``phi = arg(phi_1) - arg(phi_0)``,
canonical bracket ``{phi, s} = 2 / N`` and per-particle Hamiltonian
``h(s, phi) = H / N`` are reproduced exactly (see the reduction tests).

Flow. Hamilton's equations ``dz/dt = Omega grad H`` with the standard
symplectic matrix ``Omega = [[0, I], [-I, 0]]`` on the ``(q, p)`` layout,
equivalently ``i dphi_j/dt = dH / dconjugate(phi_j)`` (Gross-Pitaevskii /
DNLS). Both ``H`` and the norm are conserved along exact solutions; the
analytic Jacobian ``D(Omega grad H) = Omega Hess H`` is supplied to
``scipy.integrate.solve_ivp`` (default ``method="Radau"``).

Fixed points. Because the flow carries a global ``U(1)`` phase, the
physically meaningful stationary objects are *relative* fixed points:
states that are constant in a frame rotating with frequency ``gamma``,
i.e. states with ``dz/dt = gamma * (-p, q)``, fixed points of the
phase-reduced flow - the same objects the dimer module lists as ``(s, phi)``
points. Stability is read from the rotating-frame matrix
``D(gamma)(z) = Omega Hess H - gamma * [[0, -I], [I, 0]]``
(:meth:`MeanFieldLattice.stability_matrix`): the lab-frame Jacobian at a
rotating fixed point has a ``gamma``-dependent spectrum and does *not*
display the reduced stability exponents, so :func:`bohnasa.dimer_lambda`
reduces to the nonzero spectrum of ``stability_matrix``, not of
``jacobian`` (see tests).

Lyapunov. Tangent vectors obey ``dv/dt = D(Omega grad H)(z(t)) v`` along
the base trajectory; the maximal exponent is the time average
``(1/T) * sum log-stretch`` with renormalization on the ``times`` grid
(Benettin), the log of the end-to-end separation (two-trajectory), or the
leading diagonal of the QR flow (QR of the propagated tangent basis every
step). ``gamma``-rotation of the base orbit is physical and included.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.optimize as sopt
from scipy.integrate import solve_ivp

__all__ = ["MeanFieldLattice"]

# Dense (2L x 2L) Jacobians are used throughout: the engine targets small
# systems (L <= ~20, phase-space dim <= 40) where Radau factorizes a dense
# O(40^3) matrix per step and a scipy.sparse pipeline would add machinery
# without a measured speedup (CODING_RULES: benchmark before optimizing).
_MAX_EXACT_SEED_PATTERNS = 256  # exhaustive +-sign seeds only for L <= 8


class MeanFieldLattice:
    r"""Analytic DNLS mean-field engine on an ``L``-site ring or chain.

    Parameters
    ----------
    L : int
        Number of lattice sites, ``L >= 2``.
    theta : float
        Quarter-circle coupling angle, ``J = cos(theta)``,
        ``U = 2 sin(theta) / N`` with ``N = L``.
    mu : float, default 0.0
        Chemical-potential-like shift; for uniform ``mu`` it only rotates
        the global phase, leaving relative dynamics unchanged.
    connections : {"pbc", "obc"}, default "pbc"
        Neighbour convention; see the module docstring. The dimer reduction
        to :mod:`bohnasa.classical_mean_field` holds for ``"pbc"``.

    Examples
    --------
    >>> import numpy as np
    >>> lat = MeanFieldLattice(2, 1.35)
    >>> lat.dim, lat.filling
    (4, 2.0)
    >>> round(lat.hamiltonian(np.array([np.sqrt(2), np.sqrt(2), 0.0, 0.0])), 10)
    0.0996966095
    >>> float(lat.particle_number(np.array([2.0, 0.0, 0.0, 0.0])))
    2.0
    """

    def __init__(
        self,
        L: int,
        theta: float,
        mu: float = 0.0,
        connections: str = "pbc",
    ) -> None:
        if isinstance(L, bool) or not isinstance(L, (int, np.integer)):
            raise TypeError(f"MeanFieldLattice: L must be an int, got {type(L).__name__}")
        if L < 2:
            raise ValueError(f"MeanFieldLattice: L must be >= 2, got {L}")
        for name, val in (("theta", theta), ("mu", mu)):
            if not np.isscalar(val) or not np.isfinite(val):
                raise ValueError(f"MeanFieldLattice: {name} must be a finite scalar")
        if connections not in ("pbc", "obc"):
            raise ValueError('MeanFieldLattice: connections must be "pbc" or "obc"')
        self._L = int(L)
        self._theta = float(theta)
        self._mu = float(mu)
        self._connections = connections
        self._c = float(np.cos(self._theta))
        self._u = float(np.sin(self._theta))
        adj = np.zeros((self._L, self._L))
        for j in range(self._L):
            for delta in (1, -1):
                k = j + delta
                if connections == "pbc":
                    k %= self._L
                elif not 0 <= k < self._L:
                    continue
                if k != j:
                    # multiplicity counting: the L = 2 PBC ring adds the
                    # neighbour twice, matching the dimer roll-sum convention
                    adj[j, k] += 1.0
        self._adj = adj

    @property
    def L(self) -> int:
        """Number of lattice sites."""
        return self._L

    @property
    def theta(self) -> float:
        """Quarter-circle coupling angle."""
        return self._theta

    @property
    def mu(self) -> float:
        """Chemical-potential-like shift."""
        return self._mu

    @property
    def connections(self) -> str:
        """Neighbour convention, ``"pbc"`` or ``"obc"``."""
        return self._connections

    @property
    def dim(self) -> int:
        """Phase-space dimension ``2L``."""
        return 2 * self._L

    @property
    def filling(self) -> float:
        """Conserved total occupation of the unit-filling shell, ``N = L``."""
        return float(self._L)

    def _state(self, z: Sequence[float] | np.ndarray) -> np.ndarray:
        """Validate a real state vector (internal helper).

        Complex input with negligible imaginary part is accepted (its real
        part is used); anything else raises ValueError.
        """
        arr = np.asarray(z, dtype=complex)
        if arr.ndim != 1 or arr.size != self.dim:
            raise ValueError(f"expected a state vector of shape ({self.dim},), got {np.shape(z)}")
        if np.max(np.abs(arr.imag)) > 1e-12:
            raise ValueError("state must be real (q, p) amplitudes")
        out = arr.real.astype(float)
        if not np.all(np.isfinite(out)):
            raise ValueError("state contains non-finite entries")
        return out

    def _blocks(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q, p = z[: self._L], z[self._L :]
        return q, p, 0.5 * (q * q + p * p)

    def hamiltonian(self, z: Sequence[float] | np.ndarray) -> float:
        """Mean-field energy ``H(z)``; conserved along trajectories.

        Examples
        --------
        >>> lat = MeanFieldLattice(4, 0.9)
        >>> state = [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5]
        >>> round(lat.hamiltonian(state), 10)
        -2.4960756932
        """
        z = self._state(z)
        q, p, n = self._blocks(z)
        hop = 0.5 * (q @ self._adj @ q + p @ self._adj @ p)
        pot = 0.5 * self._u * float(np.sum(n * n))
        shift = 0.5 * self._mu * float(q @ q + p @ p)
        return float(-self._c * hop + pot - shift)

    def particle_number(self, z: Sequence[float] | np.ndarray) -> float:
        """Total occupation ``sum_j (q_j^2 + p_j^2) / 2``; conserved.

        Examples
        --------
        >>> lat = MeanFieldLattice(3, 1.0)
        >>> float(lat.particle_number([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
        3.0
        """
        z = self._state(z)
        return float(0.5 * (z @ z))

    def flow(self, z: Sequence[float] | np.ndarray) -> np.ndarray:
        r"""Right-hand side ``Omega grad H`` of the DNLS flow.

        ``dq_j/dt = -cos(theta) * (roll p)_j + sin(theta) * n_j p_j - mu p_j``
        and the partner equation for ``p``, with
        ``(roll x)_j = x_{j+1} + x_{j-1}`` in the chosen connection
        convention. Fixed points rotate: ``flow`` is along the phase orbit,
        not zero (see :meth:`fixed_points`).

        Examples
        --------
        >>> import numpy as np
        >>> lat = MeanFieldLattice(2, 1.35)
        >>> ground = np.array([np.sqrt(2.0), np.sqrt(2.0), 0.0, 0.0])
        >>> np.round(lat.flow(ground), 8)
        array([ 0.        ,  0.        , -0.76043675, -0.76043675])
        """
        z = self._state(z)
        q, p, n = self._blocks(z)
        dq = -self._c * (self._adj @ p) + self._u * n * p - self._mu * p
        dp = self._c * (self._adj @ q) - self._u * n * q + self._mu * q
        return np.concatenate((dq, dp))

    def jacobian(self, z: Sequence[float] | np.ndarray) -> np.ndarray:
        r"""Analytic Jacobian ``Omega Hess H`` of the flow, shape ``(2L, 2L)``.

        Traceless and symplectic (``M.T @ Omega + Omega @ M = 0``), the
        general-L analogue of :func:`bohnasa.dimer_jacobian`; consumed by
        ``solve_ivp``.

        Examples
        --------
        >>> import numpy as np
        >>> lat = MeanFieldLattice(3, 1.0, mu=0.3)
        >>> jac = lat.jacobian([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
        >>> bool(np.abs(np.trace(jac)) < 1e-15)
        True
        >>> omega = np.block([[np.zeros((3, 3)), np.eye(3)], [-np.eye(3), np.zeros((3, 3))]])
        >>> bool(np.allclose(jac.T @ omega + omega @ jac, 0.0, atol=1e-12))
        True
        """
        z = self._state(z)
        q, p, _ = self._blocks(z)
        hqq = -self._c * self._adj + np.diag(self._u * (3.0 * q * q + p * p) / 2.0 - self._mu)
        hpp = -self._c * self._adj + np.diag(self._u * (q * q + 3.0 * p * p) / 2.0 - self._mu)
        hqp = np.diag(self._u * q * p)
        return np.block([[hqp, hpp], [-hqq, -hqp]])

    def rotation_frequency(self, z: Sequence[float] | np.ndarray) -> float:
        """Phase-orbit frequency ``gamma`` of a relative fixed point.

        Least-squares slope of the global phase rotation,
        ``gamma = <flow(z), g(z)> / <g(z), g(z)>`` with the generator
        ``g = (-p, q)``; exact on the phase orbit of a fixed point.

        Examples
        --------
        >>> import numpy as np
        >>> lat = MeanFieldLattice(2, 1.35)
        >>> saddle = np.array([np.sqrt(2.0), -np.sqrt(2.0), 0.0, 0.0])
        >>> round(lat.rotation_frequency(saddle), 10)
        -1.413736732
        """
        z = self._state(z)
        g = np.concatenate((-z[self._L :], z[: self._L]))
        gg = float(g @ g)
        if gg == 0.0:
            raise ValueError("rotation_frequency: vanishing phase generator")
        return float(self.flow(z) @ g / gg)

    def stability_matrix(self, z: Sequence[float] | np.ndarray) -> np.ndarray:
        r"""Jacobian in the frame rotating with ``rotation_frequency(z)``.

        ``M_rot = jacobian(z) - gamma * [[0, -I], [I, 0]]``; at a relative
        fixed point its spectrum is ``{0, 0}`` (the phase orbit) plus the
        stability pair(s) ``+-lambda`` / ``+-i omega`` - the general-L
        counterpart of :func:`bohnasa.dimer_jacobian`, whose nonzero
        spectrum it reproduces for ``L = 2`` (see tests). The *lab-frame*
        :meth:`jacobian` does not have this property: its spectrum depends
        on where the phase orbit happens to sit.

        Examples
        --------
        >>> import numpy as np
        >>> from bohnasa import dimer_lambda
        >>> lat = MeanFieldLattice(2, 1.35)
        >>> saddle = np.array([np.sqrt(2.0), -np.sqrt(2.0), 0.0, 0.0])
        >>> eigs = np.linalg.eigvals(lat.stability_matrix(saddle))
        >>> round(float(max(eigs.real)), 10) == round(dimer_lambda(1.35), 10)
        True
        >>> float(np.sort(np.abs(eigs))[1]) < 1e-8   # the phase-orbit zero pair
        True
        """
        z = self._state(z)
        gamma = self.rotation_frequency(z)
        rot = np.zeros((self.dim, self.dim))
        rot[: self._L, self._L :] = -np.eye(self._L)
        rot[self._L :, : self._L] = np.eye(self._L)
        return self.jacobian(z) - gamma * rot

    def trajectory(
        self,
        z0: Sequence[float] | np.ndarray,
        times: Sequence[float] | np.ndarray,
        method: str = "Radau",
        rtol: float = 1e-9,
        atol: float = 1e-12,
    ) -> np.ndarray:
        """Integrate the flow with ``solve_ivp`` (analytic-Jacobian Radau).

        Parameters
        ----------
        z0 : array_like
            Initial state, length ``2L``.
        times : array_like
            Strictly increasing sample times; ``times[0]`` is the start time.
        method : str, default "Radau"
            Implicit ``solve_ivp`` scheme (matches the reference engine the
            dimer numbers were checked against); ``"DOP853"`` also works.
        rtol, atol : float
            Solver tolerances.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(len(times), 2L)``, row ``k`` is the state at
            ``times[k]``.

        Examples
        --------
        >>> import numpy as np
        >>> lat = MeanFieldLattice(2, 1.35)
        >>> start = np.array([1.3, -0.6, 0.2, 0.4])
        >>> traj = lat.trajectory(start, np.linspace(0.0, 4.0, 33))
        >>> traj.shape
        (33, 4)
        >>> float(np.ptp([lat.hamiltonian(z) for z in traj])) < 1e-9
        True
        """
        z0 = self._state(z0)
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or times.size == 0:
            raise ValueError("times must be a non-empty 1D array")
        if not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0.0):
            raise ValueError("times must be finite and strictly increasing")
        if times.size == 1:
            return np.array([z0])
        # scipy-stubs types ``method`` as an exhaustive Literal; the engine
        # deliberately accepts any solve_ivp scheme name, so the options are
        # passed through an Any-typed mapping. Explicit Runge-Kutta schemes
        # ignore (and warn about) ``jac``, so only implicit ones get it.
        solver_options: dict[str, Any] = {"method": method}
        if method not in ("RK23", "RK45", "DOP853"):
            solver_options["jac"] = lambda _t, y: self.jacobian(y)
        sol = solve_ivp(
            lambda _t, y: self.flow(y),
            (float(times[0]), float(times[-1])),
            z0,
            t_eval=times,
            rtol=rtol,
            atol=atol,
            **solver_options,
        )
        if not sol.success:
            raise RuntimeError(f"trajectory integration failed: {sol.message}")
        return sol.y.T

    def fixed_points(self, n_random: int = 60, tol: float = 1e-9) -> np.ndarray:
        r"""Relative fixed points as rows ``(n_fixed, 2L)``, sorted by energy.

        Solves ``flow(z*) = gamma * (-p, q)`` - stationary up to the global
        phase rotation, the general-L counterpart of the dimer points in
        :func:`bohnasa.dimer_fixed_points` - with gauge ``p_0 = 0``,
        ``q_0 > 0`` on the shell ``particle_number = L`` (imposed exactly by
        eliminating ``q_0`` through the norm). ``scipy.optimize.root`` (Hybr)
        runs from deterministic seeds: all ``+-`` amplitude sign patterns of
        homogeneous states, plane-wave winding states and ``n_random``
        uniform sphere points. Duplicates are removed; the search is
        deterministic but *not* certified complete for ``L >= 3``
        (branch-search caveat, thesis Sec. 10.1). For ``L = 2`` the result
        matches the closed dimer set exactly (reduction tests).

        Parameters
        ----------
        n_random : int, default 60
            Number of random sphere seeds in addition to the structured ones.
        tol : float, default 1e-9
            Accepted residual norm of the fixed-point equations.

        Examples
        --------
        >>> lat = MeanFieldLattice(2, 1.35)
        >>> pts = lat.fixed_points()
        >>> pts.shape
        (4, 4)
        >>> float(np.abs(pts[:, 2]).max())   # gauge p_0 = 0
        0.0
        """
        if not isinstance(n_random, int) or n_random < 0:
            raise ValueError("n_random must be a non-negative integer")
        found: list[tuple[np.ndarray, float]] = []
        for seed in self._fp_seeds(n_random):
            y, ok = self._solve_fp(seed, tol)
            if not ok:
                continue
            z, gamma = self._unpack_fp(y)
            if any(np.max(np.abs(z - z_old)) < 1e-6 for z_old, _ in found):
                continue
            found.append((z, gamma))
        if not found:
            return np.empty((0, self.dim))
        found.sort(key=lambda item: self.hamiltonian(item[0]))
        return np.array([z for z, _ in found])

    def _fp_seeds(self, n_random: int) -> list[np.ndarray]:
        """Deterministic seed list for the fixed-point search (internal)."""
        sites = self._L
        seeds: list[np.ndarray] = []
        n_pattern = 2**sites
        grid = np.arange(min(n_pattern, _MAX_EXACT_SEED_PATTERNS))
        signs = np.where((grid[:, None] >> np.arange(sites)) & 1, -1.0, 1.0)
        if n_pattern > _MAX_EXACT_SEED_PATTERNS:  # sparse coverage for large L
            rng = np.random.default_rng(26)
            signs = np.vstack((signs, rng.choice((-1.0, 1.0), size=(n_random, sites))))
        for row in signs:  # homogeneous |phi_j| = 1 with +- phase patterns
            seeds.append(self._pack_fp(row * np.sqrt(2.0), np.zeros(sites)))
        for m in range(sites):  # winding plane waves phi_j = exp(2 pi i m j / L)
            k = 2.0 * np.pi * m / sites
            js = np.arange(sites)
            seeds.append(
                self._pack_fp(np.sqrt(2.0) * np.cos(k * js), np.sqrt(2.0) * np.sin(k * js))
            )
        rng = np.random.default_rng(1234)
        for _ in range(n_random):  # uniform shell points
            v, w = rng.normal(size=sites), rng.normal(size=sites)
            scale = np.sqrt(2.0 * sites / np.sum(v * v + w * w))
            seeds.append(self._pack_fp(v * scale, w * scale))
        return seeds

    def _pack_fp(self, q: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Full (q, p) seed -> unknowns (q_1.., p_1.., gamma); p_0 == 0 gauge."""
        return np.concatenate((q[1:], p[1:], [0.0]))

    def _unpack_fp(self, y: np.ndarray) -> tuple[np.ndarray, float]:
        """Unknowns -> (state, rotation frequency); q_0 eliminated via the norm."""
        q_rest, p_rest = y[: self._L - 1], y[self._L - 1 : -1]
        q0 = np.sqrt(max(2.0 * self._L - float(np.sum(q_rest * q_rest + p_rest * p_rest)), 0.0))
        z = np.concatenate(([q0], q_rest, [0.0], p_rest))
        return z, float(y[-1])

    def _fp_residual(self, y: np.ndarray) -> np.ndarray:
        """Gauge-fixed fixed-point equations (2L - 1 residuals, internal).

        The convention is ``flow(z*) = gamma * g(z*)`` with phase-generator
        ``g = (-p, q)``. The section components are ``dq_j/dt + gamma p_j``
        for ``j >= 1`` and ``dp_j/dt - gamma q_j`` for all ``j``; the
        omitted ``j = 0`` momentum-section component is implied: norm
        conservation gives ``sum_j (q_j G_q_j + p_j G_p_j) = 0``, so all
        but one residual vanishing forces it closed for ``q_0 > 0``.
        """
        z, gamma = self._unpack_fp(y)
        f = self.flow(z)
        res = np.empty(self.dim - 1)
        res[: self._L - 1] = f[: self._L][1:] + gamma * z[self._L :][1:]
        res[self._L - 1 :] = f[self._L :] - gamma * z[: self._L]
        return res

    def _solve_fp(self, seed: np.ndarray, tol: float) -> tuple[np.ndarray, bool]:
        sol = sopt.root(
            self._fp_residual, seed, method="hybr", options={"xtol": 1e-13, "maxfev": 4000}
        )
        if not sol.success or float(np.linalg.norm(self._fp_residual(sol.x))) >= tol:
            return seed, False
        z, _ = self._unpack_fp(sol.x)
        # the eliminated q0 = sqrt(max(2L - shell, 0)) has a spurious clamped
        # branch (q0 = 0, off the number shell); real fixed points must sit
        # on it exactly, independent of where the Newton step landed
        if abs(self.particle_number(z) - self._L) > 1e-8:
            return seed, False
        return sol.x, True

    def lyapunov(
        self,
        z0: Sequence[float] | np.ndarray,
        times: Sequence[float] | np.ndarray,
        method: str = "benettin",
        v0: Sequence[float] | np.ndarray | None = None,
        delta: float = 1e-8,
        rtol: float = 1e-9,
        atol: float = 1e-12,
    ) -> float:
        r"""Maximal Lyapunov exponent of the flow over the ``times`` horizon.

        The tangent space is driven by ``dv/dt = M(z(t)) v`` with the
        analytic ``M = jacobian(z)``; the base trajectory is re-integrated
        segment by segment between the entries of ``times``, which double as
        the renormalization grid (choose it fine enough for convergence).

        Parameters
        ----------
        z0 : array_like
            Base state of the trajectory.
        times : array_like
            Strictly increasing time grid; its spacing is the
            renormalization interval of the rescaling methods.
        method : {"benettin", "two_trajectory", "qr"}, default "benettin"
            ``"benettin"``: single tangent vector, renormalized to norm
            ``delta`` every step; ``"two_trajectory"``: finite-difference
            separation of two trajectories started ``delta`` apart (saturates
            at long times, use moderate horizons); ``"qr"``: QR
            decomposition of the propagated tangent basis every step - the
            independent reference implementation (agreement with Benettin is
            tested). ``qr`` starts from the identity basis and ignores
            ``v0`` and ``delta``.
        v0 : array_like, optional
            Initial perturbation direction, normalized internally. Default:
            deterministic pseudo-random (fixed seed).
        delta : float, default 1e-8
            Perturbation magnitude.
        rtol, atol : float
            Solver tolerances per segment.

        Returns
        -------
        float
            Time-averaged maximal exponent, ``(1 / T) * sum log-stretches``
            over the horizon ``T = times[-1] - times[0]``.

        Examples
        --------
        >>> import numpy as np
        >>> lat = MeanFieldLattice(4, 1.35)
        >>> start = np.array([1.0, 1.5, 0.7, 1.3, 0.4, -0.2, 0.9, 0.1])
        >>> start *= np.sqrt(8.0) / np.linalg.norm(start)
        >>> times = np.linspace(0.0, 40.0, 401)
        >>> round(lat.lyapunov(start, times), 4)
        0.0828
        >>> round(lat.lyapunov(start, times, method="qr"), 4)
        0.084
        """
        if method not in ("benettin", "two_trajectory", "qr"):
            raise ValueError('lyapunov: method must be "benettin", "two_trajectory" or "qr"')
        z0 = self._state(z0)
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or times.size < 2 or not np.all(np.diff(times) > 0.0):
            raise ValueError("times must be a strictly increasing 1D grid")
        if not np.isfinite(delta) or delta <= 0.0:
            raise ValueError("delta must be positive")
        if v0 is None:
            v0_arr = np.random.default_rng(26).normal(size=self.dim)
        else:
            v0_arr = self._state(v0)
        v0_arr = v0_arr / float(np.linalg.norm(v0_arr))
        if method == "two_trajectory":
            return self._lyapunov_two(z0, times, delta * v0_arr, rtol, atol)
        if method == "qr":
            return self._lyapunov_qr(z0, times, rtol, atol)
        return self._lyapunov_benettin(z0, times, delta * v0_arr, rtol, atol)

    def _integrate_extended(
        self,
        t_a: float,
        t_b: float,
        y0: np.ndarray,
        rtol: float,
        atol: float,
        matrix: bool = False,
    ) -> np.ndarray | None:
        """One segment of the joint base + tangent system (internal).

        Radau with finite-differenced system Jacobian: the analytic tangent
        block is ``jacobian(z)`` but the mixed third-derivative coupling
        would need a tensor contraction; phase-space dims here are small
        (<= 40), so the FD Jacobian costs nothing measurable and keeps the
        engine closed-form. Correctness is cross-checked by method agreement
        in the tests, not by this choice.
        """
        dim = self.dim

        def rhs(_t: float, w: np.ndarray) -> np.ndarray:
            m = self.jacobian(w[:dim])
            tangent = (m @ w[dim:].reshape(dim, dim)).ravel() if matrix else m @ w[dim:]
            return np.concatenate((self.flow(w[:dim]), tangent))

        sol = solve_ivp(rhs, (t_a, t_b), y0, method="Radau", rtol=rtol, atol=atol)
        if not sol.success:
            return None
        return sol.y[:, -1]

    def _lyapunov_benettin(
        self, z0: np.ndarray, times: np.ndarray, v0: np.ndarray, rtol: float, atol: float
    ) -> float:
        dim = self.dim
        norm0 = float(np.linalg.norm(v0))
        stretch, y = 0.0, np.concatenate((z0, v0))
        for i in range(len(times) - 1):
            y_next = self._integrate_extended(times[i], times[i + 1], y, rtol, atol)
            if y_next is None:
                raise RuntimeError(f"lyapunov segment [{times[i]}, {times[i + 1]}] failed")
            norm_new = float(np.linalg.norm(y_next[dim:]))
            stretch += np.log(norm_new / norm0)
            y_next[dim:] *= norm0 / norm_new
            y = y_next
        return float(stretch / (times[-1] - times[0]))

    def _lyapunov_qr(self, z0: np.ndarray, times: np.ndarray, rtol: float, atol: float) -> float:
        dim = self.dim
        y = np.concatenate((z0, np.eye(dim).ravel()))
        log_max: list[float] = []
        for i in range(len(times) - 1):
            y_next = self._integrate_extended(times[i], times[i + 1], y, rtol, atol, matrix=True)
            if y_next is None:
                raise RuntimeError(f"lyapunov segment [{times[i]}, {times[i + 1]}] failed")
            mat = y_next[dim:].reshape(dim, dim)
            q_mat, r_mat = np.linalg.qr(mat)
            diag = np.diag(r_mat).copy()
            diag[diag == 0.0] = 1.0
            q_mat *= np.sign(diag)  # make the QR gauge deterministic
            log_max.append(float(np.log(np.abs(diag[0]))))
            y = np.concatenate((y_next[:dim], q_mat.ravel()))
        return float(np.sum(log_max) / (times[-1] - times[0]))

    def _lyapunov_two(
        self, z0: np.ndarray, times: np.ndarray, dz0: np.ndarray, rtol: float, atol: float
    ) -> float:
        base = self.trajectory(z0, times, rtol=rtol, atol=atol)
        pert = self.trajectory(z0 + dz0, times, rtol=rtol, atol=atol)
        d0 = float(np.linalg.norm(pert[0] - base[0]))
        d_end = float(np.linalg.norm(pert[-1] - base[-1]))
        if d0 == 0.0 or d_end == 0.0:
            raise RuntimeError("two-trajectory separation collapsed below precision")
        return float(np.log(d_end / d0) / (times[-1] - times[0]))
