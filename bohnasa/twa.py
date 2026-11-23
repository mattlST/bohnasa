"""Truncated-Wigner (TWA) semiclassical sampling of mean-field engine orbits.

This module turns a classical mean-field engine into a semiclassical
ensemble (issue #30): sample the Wigner representation of an initial
many-body state around a Gross-Pitaevskii (GP) point, propagate every
sample with the engine's own integrator, and average observables. The
route complements exact diagonalization
(:class:`bohnasa.SquaredCommutator`) and single-trajectory mean field
(:class:`bohnasa.MeanFieldLattice`): it is the finite-hbar bridge between
quantum scrambling and classical Lyapunov growth (Ehrenfest window plus
saturation).

Engine protocol. Any engine object exposing

- ``dim`` - phase-space dimension ``2L`` (real states ``z = (q, p)``),
- ``filling`` - conserved total occupation ``N = sum_j n_j`` of its shell,
- ``trajectory(z0, times)`` - states of one sample at the requested times,
  shape ``(len(times), dim)`` (for a map engine: ``times`` are step counts),

can be sampled; no integrator is implemented here.
:class:`bohnasa.MeanFieldLattice` is the primary engine and satisfies the
protocol directly. The kicked-ring map of issue #29 (planned module
``bohnasa.kicked``, branch ``feat/kicked-bh-29``) matches the same
protocol through its ``(s, phi)``/``alpha`` gauge - the gauge lives inside
the engine's state and this sampler only displaces and propagates engine
states - but nothing here imports it yet: the merged
:class:`bohnasa.KickedMeanFieldMap` (issue #29) exposes ``step`` /
``iterate`` but not the protocol's ``dim`` / ``filling`` / ``trajectory``,
so adoption needs only a three-method shim; until then passing it to
:class:`TruncatedWigner` raises the protocol ``TypeError``, and the
protocol path itself is exercised by the closed-form ``ShiftEngine``
test engine in ``tests/unit/test_twa.py``.

Phase space and hbar bookkeeping. Following
:mod:`bohnasa.mean_field_lattice`, ``phi_j = (q_j + i p_j) / sqrt(2)``
with occupations ``n_j = |phi_j|^2`` on the engine shell
``sum_j n_j = N = L`` (unit filling) and the canonical bracket
``{q_j, p_k} = delta_jk`` (symplectic matrix ``Omega = [[0, I], [-I, 0]]``,
``flow = Omega grad H``). The quantum model the engine approximates has
``N_phys`` bosons with ``[b_j, b_k^dag] = delta_jk``; density-scaled
amplitudes ``psi_j = b_j / sqrt(N_phys)`` carry the Moyal expansion
parameter

    hbar_eff = 1 / N_phys        ({psi_j, psi_k^dag} = -i / N_phys),

the same parameter the reduced dimer bracket ``{phi, s} = 2 / N`` of
:mod:`bohnasa.classical_mean_field` encodes. The engine fixes the gauge
``N_phys = L`` (its quarter-circle coupling ``J = cos(theta)``,
``U = 2 sin(theta) / N`` makes the flow identical for every ``N_phys``
after the density rescaling ``phi_j = sqrt(L / N_phys) alpha_j`` of the
physical Wigner amplitudes ``alpha_j`` of ``b_j``), so the engine-variable
Moyal unit is

    hbar_M = L * hbar_eff = L / N_phys,     (engine shell: hbar_M = 1).

Wigner sampling rule (a): each quadrature of a physical mode carries the
half-quantum vacuum fluctuation (variance ``1/2``, the exact Weyl zero
point of ``b^dag b <-> |alpha|^2 - 1/2``), so in engine coordinates

    z_sample = z_gp + delta,    delta_q, delta_p ~ N(0, hbar_M / 2).

Symbol dictionary (b): the Weyl symbol of the physical occupation operator
is ``n_j <-> (N_phys / L) |phi_j|^2 - 1/2``; on the engine shell it is
``n_j <-> |phi_j|^2 - 1/2`` (the zero point is ``_WIGNER_ZERO_POINT``).
Truncation rule (c): the leading Moyal term of every commutator,
``[A, B] <-> i hbar_M {A, B}_Omega`` with the Poisson bracket
``{F, G} = grad F^T Omega grad G`` evaluated on engine states, applied to
the Heisenberg-evolved symbol ``O(t)(z) = O(Phi_t z)``:

    C_TWA(t) = hbar_M^2 * mean_samples( {O(t), O}_Omega(z0)^2 ),

the classical limit of the squared commutator
``C(t) = || [O(t), O] |psi> ||^2`` of :class:`bohnasa.SquaredCommutator`.
The bracket is evaluated per sample by central finite differences of two
shadow engine trajectories started at ``z0 +- eps X_O``, where
``X_O = Omega grad O`` is the Hamiltonian vector field of the symbol at
the initial point - the engine's integrator is reused, never reimplemented.

Measured validation limits (``tests/physics/test_twa_physics.py``, L = 2
dimer): rules (a)-(c) reproduce the exact quantum short-time coefficient
up to the leading truncation error, ``C_TWA / C_quantum = 1 + hbar_eff +
O(t hbar_eff)`` - measured deviation coefficient ``0.94 - 1.00`` times
``hbar_eff`` at ``t -> 0`` for ``N_phys = 8..32``, rising to
``1.5 - 2.5 hbar_eff`` across the window up to ``t = 1`` - while the
unsampled bracket point value matches the quantum short-time coefficient
itself to ``1.3e-7`` relative (the +hbar_eff term is the finite
wave-packet width, i.e. genuinely semiclassical). The sample mean of any
symbol converges to the deterministic ensemble average at the
Monte-Carlo rate ``1 / sqrt(samples)``. Validation (1) - "TWA -> mean
field for many samples" - is checked on a REGULAR (below-pitchfork)
orbit only: a single chaotic trajectory is not the mean-field orbit and
sample-number growth cannot fix it.

Relation to issue #34 (documented physics, clearly separate from the
measured claims above). The thesis result this feature generalizes is the
universal early-time crossover of the OTOC growth exponent of ``sqrt(C)``
from ``2 lambda_s`` to ``lambda_s``, sharp as ``hbar_eff -> 0`` and
governed by the wave-packet shape around the hyperbolic fixed point (see
issue #34 and the quantum anchors
``test_duration_of_superclassical_growth_scales_logarithmically_with_N``
and ``test_effective_rate_converges_to_twice_the_classical_exponent`` in
``tests/physics/test_otoc_physics.py``, where ``d log(C)/dt`` runs at
``2 * lambda_s`` in the window). Because TWA propagates the wave packet
itself as a phase-space ensemble, in this module the crossover is a
*classical* structure: ``log C_TWA`` runs at ``4 lambda_s`` before and
``2 lambda_s`` after the crossover, and the late-time departure of the
quantum curve (saturation on the Ehrenfest time ``log(N) / lambda_s``)
is precisely what TWA cannot capture. The physics test measures the two
sides of the crossover for TWA, quantum and the classical point value at
tiny L, N and pins them within measured bands. The engine's *global*
maximal Lyapunov exponent (``MeanFieldLattice.lyapunov`` at L >= 4) is
deliberately NOT pinned against a TWA slope here: the exponent is
IC-specific and horizon-converging (measured for the L = 4 seed-0 orbit
0.2197 over t <= 20 and 0.2342 over t <= 40, while the module doctest's
own initial condition gives 0.0828), and the sampled wave-packet growth
of ``log C_TWA`` on the short CI horizons is not the asymptotic 2*lambda
of a single tangent vector (measured slope 0.96 over t in [1, 5] for the
seed-0 orbit) - that side is documented physics tied to issues #26/#39,
while the pinned L = 2 comparison uses the exact saddle exponent
``lambda_s`` (``dimer_lambda`` = the ``stability_matrix`` eigenvalue the
engine reduces to).
"""

from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np

__all__ = ["TruncatedWigner", "WignerSampler"]

# Numerical settings policy (issue #25): every numeric setting below is a
# named constant with its measured regime; see CONTRIBUTING.md "Numerical
# settings policy".
#
# Default determinism seed of TruncatedWigner. Tested invariant: the
# doctests on this module and the byte-reproducibility tests in
# tests/unit/test_twa.py track this exact stream; changing it changes every
# sample set they quote. Threading (issue #30 acceptance): the RNG stream
# itself is bit-stable across OMP_NUM_THREADS = 1 vs default (measured
# 2026-11-22, 16-core host) - what is NOT bit-stable is the Radau
# propagation downstream (<= 2e-9 relative drift), which is why only
# sample arrays, not curves, are pinned byte-wise in the tests.
_DEFAULT_SEED = 2026
# Central-difference step of the Poisson-bracket shadow trajectories, as a
# multiple of the Hamiltonian vector field X_O = Omega grad O (the bracket
# displacement is eps * X_O). Measured on the L = 2 saddle OTOC anchor at
# N_phys = 16, K = 600, over t in [0.5, 2] (scan behind
# tests/physics/test_twa_physics.py::test_bracket_fd_step_lands_in_measured_gap):
# C_TWA agrees to >= 4 digits for every eps from 3e-4 down to 1e-7 (no
# round-off floor reached), the curvature truncation error becomes visible
# at 1e-2 (-2.6 % at t = 2 against the plateau) and grows with eps (t = 2:
# -20 % at 3e-2, -79 % at 1e-1). 1e-4 sits >= 30x below the knee and >=
# 300x above the tested lower edge, inside the plateau on both sides.
_BRACKET_FD_EPS = 1e-4
# Central-difference step of the symbol gradient building X_O. The
# difference quotient of a quadratic symbol is its derivative exactly (to
# round-off), and the quadratic Weyl symbols are this module's design
# target; measured on the shell symbols: gradient constant across
# h = 1e-3..1e-7 to <= 1e-7 relative (tests/unit/test_twa.py). Truncation
# for general smooth symbols is O(h^2), round-off floor O(eps_mach / h).
_GRAD_FD_STEP = 1e-5
# Half-quantum vacuum variance per quadrature of sampling rule (a) - the
# exact Weyl zero point of b^dag b, not a tunable: the short-time
# validation constant C_TWA / C_quantum = 1 + hbar_eff + ... fails over
# when this factor moves (tests/physics/test_twa_physics.py;
# tested-invariant, class (a) of the settings policy).
_WIGNER_ZERO_POINT = 0.5


class MeanFieldEngine(Protocol):
    """Duck type of the engines :class:`TruncatedWigner` samples.

    The protocol is documented in the module docstring; every entry point
    checks it through :func:`_validate_engine` and raises
    ``TypeError``/``ValueError`` with protocol language, so any object
    with the three members works - no subclassing required.

    Examples
    --------
    >>> from bohnasa import MeanFieldLattice
    >>> engine = MeanFieldLattice(2, 1.35)
    >>> engine.dim, engine.filling
    (4, 2.0)
    """

    dim: int
    filling: float
    trajectory: Callable[..., np.ndarray]


def _is_positive_finite_scalar(x: object) -> bool:
    """True when x is a real scalar convertible to a positive finite float."""
    if isinstance(x, (bool, str, bytes)):
        return False
    try:
        arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        return False
    return arr.ndim == 0 and bool(np.isfinite(arr)) and float(arr) > 0.0


def _validate_engine(engine: MeanFieldEngine) -> None:
    """Check the documented engine protocol (internal; runtime duck check)."""
    for member in ("dim", "filling", "trajectory"):
        if not hasattr(engine, member):
            raise TypeError(
                f"TruncatedWigner: engine does not satisfy the mean-field protocol, "
                f"missing attribute {member!r} (see the module docstring)"
            )
    dim = engine.dim
    if isinstance(dim, bool) or not isinstance(dim, (int, np.integer)) or dim < 4 or dim % 2:
        raise ValueError(f"TruncatedWigner: engine.dim must be an even integer >= 4, got {dim!r}")
    if not _is_positive_finite_scalar(engine.filling):
        raise ValueError("TruncatedWigner: engine.filling must be a positive finite scalar")
    if not callable(engine.trajectory):
        raise TypeError("TruncatedWigner: engine.trajectory must be callable")


def _validate_symbol(func: object) -> Callable[[np.ndarray], float]:
    """Wrap a scalar field symbol with per-state validation (internal)."""
    if not callable(func):
        raise TypeError(f"TruncatedWigner: func must be callable, got {type(func).__name__}")

    def symbol(z: np.ndarray) -> float:
        val = func(z)
        if (
            isinstance(val, (str, bytes))
            or np.ndim(val) != 0
            or np.iscomplexobj(val)
            or not np.isfinite(val)
        ):
            raise ValueError(
                "TruncatedWigner: func must return a finite real scalar per state, "
                f"got {np.asarray(val)!r}"
            )
        return float(val)

    return symbol


def _as_times(times: Sequence[float] | np.ndarray) -> np.ndarray:
    """Validate the sample-time grid (internal); samples live at t = 0."""
    arr = np.asarray(times, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("TruncatedWigner: times must be a non-empty 1D array")
    if not np.all(np.isfinite(arr)):
        raise ValueError("TruncatedWigner: times must be finite")
    if np.any(arr < 0.0):
        raise ValueError(
            "TruncatedWigner: times must be nonnegative (samples are drawn at t = 0 "
            "and propagated forward)"
        )
    if arr.size > 1 and not np.all(np.diff(arr) > 0.0):
        raise ValueError("TruncatedWigner: times must be strictly increasing")
    return arr


def _as_samples(samples: int, minimum: int) -> int:
    """Validate a sample count (internal)."""
    if isinstance(samples, bool) or not isinstance(samples, (int, np.integer)):
        raise TypeError(f"TruncatedWigner: samples must be an int, got {type(samples).__name__}")
    if samples < minimum:
        raise ValueError(
            f"TruncatedWigner: samples must be >= {minimum}, got {samples}"
            + (" (a standard error needs at least two samples)" if minimum == 2 else "")
        )
    return int(samples)


def _omega_apply(v: np.ndarray) -> np.ndarray:
    """Symplectic map ``Omega @ v``, ``Omega = [[0, I], [-I, 0]]`` on ``(q, p)``."""
    half = v.shape[0] // 2
    return np.concatenate((v[half:], -v[:half]))


def _grad_symbol(
    func: Callable[[np.ndarray], float], z: np.ndarray, step: float | None = None
) -> np.ndarray:
    """Central-difference gradient of a scalar symbol (internal, hot loop).

    Exact to round-off for the quadratic Weyl symbols this module is
    designed around; the default step is the measured constant
    ``_GRAD_FD_STEP`` (the ``step`` argument exists for the regime probe
    in ``tests/unit/test_twa.py::test_gradient_step_truncation_gap``).
    """
    h = _GRAD_FD_STEP if step is None else step
    out = np.empty_like(z)
    for k in range(z.shape[0]):
        zp, zm = z.copy(), z.copy()
        zp[k] += h
        zm[k] -= h
        out[k] = (func(zp) - func(zm)) / (2.0 * h)
    return out


class WignerSampler:
    r"""Wigner-representation initial samples around a Gross-Pitaevskii point.

    Implements sampling rule (a) of the module docstring: the GP point
    ``z_gp`` (engine coordinates) receives independent Gaussian noise of
    variance ``hbar_M / 2 = L * hbar_eff / 2`` per canonical quadrature -
    the half-quantum vacuum fluctuation of the physical modes expressed in
    engine variables. On the engine's own unit-filling shell
    (``hbar_eff = 1 / L``, i.e. ``N_phys = L``) this is exactly half a
    quantum per quadrature.

    Parameters
    ----------
    engine : MeanFieldEngine
        Any engine satisfying the module docstring protocol.
    hbar_eff : float, optional
        Moyal expansion parameter ``1 / N_phys`` of the quantum model being
        approximated; the default ``None`` uses the engine shell value
        ``1 / engine.filling``.

    Examples
    --------
    >>> import numpy as np
    >>> from bohnasa import MeanFieldLattice
    >>> lat = MeanFieldLattice(2, 1.35)
    >>> sampler = WignerSampler(lat)
    >>> sampler.noise_std           # sqrt(hbar_M / 2), hbar_M = 1 on the shell
    0.7071067811865476
    >>> saddle = np.array([np.sqrt(2.0), -np.sqrt(2.0), 0.0, 0.0])
    >>> np.round(sampler.initial(saddle, 2, seed=7), 6)
    array([[ 1.415083, -1.202969, -0.193845, -0.629744],
           [ 1.092713, -2.115414,  0.042528,  0.947675]])
    """

    def __init__(self, engine: MeanFieldEngine, *, hbar_eff: float | None = None) -> None:
        _validate_engine(engine)
        self.engine = engine
        if hbar_eff is None:
            hbar_eff = 1.0 / float(engine.filling)
        elif not _is_positive_finite_scalar(hbar_eff):
            raise ValueError(
                f"WignerSampler: hbar_eff must be a positive finite scalar or None, got "
                f"{hbar_eff!r}"
            )
        self.hbar_eff = float(hbar_eff)
        # engine-variable Moyal unit hbar_M = L * hbar_eff (module docstring, rule (a))
        self.hbar_moyal = engine.dim / 2.0 * self.hbar_eff
        self.noise_std = float(np.sqrt(_WIGNER_ZERO_POINT * self.hbar_moyal))

    def initial(
        self,
        state: Sequence[float] | np.ndarray,
        samples: int,
        seed: int | None = _DEFAULT_SEED,
    ) -> np.ndarray:
        """Draw ``samples`` Wigner initial conditions around ``state``.

        Parameters
        ----------
        state : array_like
            GP expansion point ``z_gp`` in engine coordinates, shape ``(dim,)``.
        samples : int
            Number of samples (positive).
        seed : int or None, default ``_DEFAULT_SEED``
            Seed of ``numpy.random.default_rng``. The same seed reproduces
            the array byte-for-byte, and the first ``K' <= K`` rows of a
            draw of ``K`` rows are exactly the draw of ``K'`` rows
            (nested ensembles; tested invariant).

        Returns
        -------
        numpy.ndarray
            Float array of shape ``(samples, dim)``.

        Examples
        --------
        >>> import numpy as np
        >>> from bohnasa import MeanFieldLattice
        >>> lat = MeanFieldLattice(4, 1.2)
        >>> gp = np.concatenate((np.ones(4), np.zeros(4))) * np.sqrt(2.0)
        >>> z = WignerSampler(lat).initial(gp, 16, seed=3)
        >>> z.shape
        (16, 8)
        >>> float(z.std()) > 0.0
        True
        """
        z_gp = self._validated_state(state)
        samples = _as_samples(samples, 1)
        rng = np.random.default_rng(seed)
        return z_gp + rng.normal(scale=self.noise_std, size=(samples, self.engine.dim))

    def _validated_state(self, state: Sequence[float] | np.ndarray) -> np.ndarray:
        """Validate a GP expansion point (internal; layout per module docstring)."""
        arr = np.asarray(state, dtype=float)
        if arr.ndim != 1 or arr.size != self.engine.dim:
            raise ValueError(
                f"WignerSampler: state must have shape ({self.engine.dim},), got {np.shape(state)}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("WignerSampler: state contains non-finite entries")
        return arr.copy()


class TruncatedWigner:
    r"""Truncated-Wigner ensemble of mean-field engine trajectories.

    Samples the Wigner representation of an initial many-body state around
    the GP point ``state`` (rule (a) of the module docstring), propagates
    each sample with ``engine.trajectory`` (the engine's own integrator)
    and returns sample means with standard errors:

    - :meth:`observable` - the sample mean of a field symbol ``func``,
    - :meth:`otoc` - the TWA squared commutator ``C_TWA(t)`` (rule (c)),
      directly comparable to :class:`bohnasa.SquaredCommutator` through
      the symbol dictionary (b) of the module docstring.

    Determinism: every draw re-derives its ensemble from ``seed`` alone
    (``numpy.random.default_rng(seed)``), so results are byte-reproducible
    across reruns, independent of the method-call order, and nested:
    increasing ``samples`` extends the previous ensemble instead of
    redrawing it, which makes convergence studies consistent.

    Parameters
    ----------
    engine : MeanFieldEngine
        Engine satisfying the module docstring protocol;
        :class:`bohnasa.MeanFieldLattice` is the primary engine.
    state : array_like, optional
        GP expansion point ``z_gp`` in engine coordinates. The default
        ``None`` is the uniform unit-filling point ``phi_j = 1`` (i.e.
        ``q_j = sqrt(2)``, ``p_j = 0``, occupation ``n_j = 1`` per site on
        the engine shell) - a relative fixed point of the DNLS flow.
    seed : int or None, default ``_DEFAULT_SEED``
        Base seed of all sample draws; ``None`` requests fresh system
        entropy (not reproducible).
    hbar_eff : float, optional
        Moyal parameter ``1 / N_phys`` of the quantum model; the default
        ``None`` is the engine shell value ``1 / engine.filling``.

    See Also
    --------
    bohnasa.SquaredCommutator : the exact quantum reference curve.
    bohnasa.MeanFieldLattice : the primary classical engine.

    Examples
    --------
    The uniform unit-filling GP fixed point of a below-pitchfork dimer,
    sampled around with vacuum noise (engine shell, ``hbar_eff = 1/2``):

    >>> import numpy as np
    >>> from bohnasa import MeanFieldLattice
    >>> lat = MeanFieldLattice(2, 1.0)
    >>> sim = TruncatedWigner(lat, seed=7)
    >>> times = np.linspace(0.0, 4.0, 9)
    >>> func = lambda z: 0.5 * (z[0] ** 2 + z[2] ** 2) - 0.5
    >>> mean, se = sim.observable(16, times, func)
    >>> np.round(mean, 4)
    array([0.7192, 0.6971, 0.7424, 0.7063, 0.6597, 0.7449, 0.8507, 0.7619,
           0.5732])
    >>> np.round(se, 4)
    array([0.1839, 0.2013, 0.2602, 0.1788, 0.1511, 0.1708, 0.2653, 0.2169,
           0.162 ])
    >>> sim.hbar_eff, sim.hbar_moyal
    (0.5, 1.0)
    """

    def __init__(
        self,
        engine: MeanFieldEngine,
        state: Sequence[float] | np.ndarray | None = None,
        seed: int | None = _DEFAULT_SEED,
        *,
        hbar_eff: float | None = None,
    ) -> None:
        self.sampler = WignerSampler(engine, hbar_eff=hbar_eff)
        self.engine = engine
        if state is None:
            half = engine.dim // 2
            state = np.concatenate((np.full(half, np.sqrt(2.0)), np.zeros(half)))
        self.state = self.sampler._validated_state(state)
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, (int, np.integer))):
            raise TypeError(f"TruncatedWigner: seed must be an int or None, got {seed!r}")
        self.seed = seed
        self.hbar_eff = self.sampler.hbar_eff
        self.hbar_moyal = self.sampler.hbar_moyal

    def sample(self, samples: int) -> np.ndarray:
        """Draw the ensemble's Wigner initial conditions, shape ``(samples, dim)``.

        Examples
        --------
        >>> from bohnasa import MeanFieldLattice
        >>> sim = TruncatedWigner(MeanFieldLattice(2, 1.35), seed=5)
        >>> sim.sample(4).shape
        (4, 4)
        """
        return self.sampler.initial(self.state, _as_samples(samples, 1), seed=self.seed)

    def propagate(self, samples: int, times: Sequence[float] | np.ndarray) -> np.ndarray:
        """Propagate the whole ensemble; shape ``(samples, len(times), dim)``.

        ``engine.trajectory`` is called once per sample on ``times`` when
        ``times[0] == 0`` and on ``[0] + times`` otherwise (the samples are
        defined at ``t = 0``).

        Examples
        --------
        >>> import numpy as np
        >>> from bohnasa import MeanFieldLattice
        >>> sim = TruncatedWigner(MeanFieldLattice(2, 1.0), seed=5)
        >>> trajs = sim.propagate(6, np.linspace(0.0, 2.0, 5))
        >>> trajs.shape
        (6, 5, 4)
        """
        times = _as_times(times)
        z0 = self.sample(samples)
        grid = self._grid(times)
        dim = self.engine.dim
        out = np.empty((z0.shape[0], times.size, dim))
        for k, z in enumerate(z0):
            out[k] = self._propagate_one(z, grid)[grid.size - times.size :]
        return out

    def observable(
        self,
        samples: int,
        times: Sequence[float] | np.ndarray,
        func: Callable[[np.ndarray], Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample mean and standard error of a field symbol ``func(z)``.

        Parameters
        ----------
        samples : int
            Number of ensemble members (``>= 2``: a standard error needs
            two samples).
        times : array_like
            Strictly increasing nonnegative sample times.
        func : callable
            Symbol ``z -> value`` (scalar or array, finite) evaluated on
            every propagated engine state. For physical occupations use
            the symbol dictionary (b) of the module docstring; on the
            engine shell ``n_j <-> 0.5 * (q_j**2 + p_j**2) - 0.5``.

        Returns
        -------
        tuple of numpy.ndarray
            ``(mean, stderr)``: ``mean[..., k]`` is the ensemble average
            at ``times[k]``, ``stderr = std(samples, ddof=1) /
            sqrt(samples)``.

        Examples
        --------
        >>> import numpy as np
        >>> from bohnasa import MeanFieldLattice
        >>> sim = TruncatedWigner(MeanFieldLattice(2, 1.0), seed=7)
        >>> func = lambda z: 0.5 * (z[0] ** 2 + z[2] ** 2) - 0.5
        >>> mean, se = sim.observable(8, np.array([0.0, 1.0]), func)
        >>> round(float(mean[0]), 4), round(float(se[0]), 4)
        (0.6322, 0.2626)
        """
        trajs = self.propagate(_as_samples(samples, 2), times)
        values = np.array([[func(z) for z in traj_k] for traj_k in trajs])
        mean = values.mean(axis=0)
        stderr = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
        return mean, stderr

    def otoc(
        self,
        samples: int,
        times: Sequence[float] | np.ndarray,
        func: Callable[[np.ndarray], Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""TWA squared commutator ``C_TWA(t)`` and its standard error.

        Evaluates rule (c) of the module docstring sample by sample: the
        scalar symbol ``func`` (the dictionary-(b) partner of the quantum
        probe) drives the Hamiltonian vector field ``X_O = Omega grad O``
        at the initial point, two shadow engine trajectories from
        ``z0 +- _BRACKET_FD_EPS * X_O`` give the Poisson bracket
        ``{O(t), O}`` by central difference, and
        ``C_TWA = hbar_M^2 * mean_samples(bracket^2)``. Times equal to
        ``0`` report exactly ``0`` - the symbol of ``[O, O]`` vanishes -
        mirroring :meth:`bohnasa.SquaredCommutator.curve`.

        Parameters
        ----------
        samples : int
            Number of ensemble members (``>= 2``).
        times : array_like
            Strictly increasing nonnegative sample times.
        func : callable
            Scalar symbol ``z -> float`` of the probe operator.

        Returns
        -------
        tuple of numpy.ndarray
            ``(c, stderr)``, both shape ``(len(times),)``.

        Raises
        ------
        ValueError
            If ``func`` returns a non-scalar or non-finite value on any
            sample, in addition to the shared input validation.

        Examples
        --------
        >>> import numpy as np
        >>> from bohnasa import MeanFieldLattice
        >>> sim = TruncatedWigner(MeanFieldLattice(2, 1.35), seed=9)
        >>> imbalance = lambda z: 0.25 * (z[0] ** 2 + z[2] ** 2 - z[1] ** 2 - z[3] ** 2)
        >>> c, se = sim.otoc(8, np.array([0.0, 0.5, 1.0]), imbalance)
        >>> float(c[0])
        0.0
        >>> np.round(c[1:], 4), np.round(se[1:], 4)
        (array([0.1075, 0.2774]), array([0.0241, 0.0686]))
        """
        times = _as_times(times)
        symbol = _validate_symbol(func)
        z0 = self.sample(_as_samples(samples, 2))
        grid = self._grid(times)
        br = np.empty((z0.shape[0], times.size))
        skip = grid.size - times.size
        for i, z in enumerate(z0):
            x_o = _omega_apply(_grad_symbol(symbol, z))
            traj_p = self._propagate_one(z + _BRACKET_FD_EPS * x_o, grid)
            traj_m = self._propagate_one(z - _BRACKET_FD_EPS * x_o, grid)
            f_p = np.array([symbol(v) for v in traj_p[skip:]])
            f_m = np.array([symbol(v) for v in traj_m[skip:]])
            br[i] = (f_p - f_m) / (2.0 * _BRACKET_FD_EPS)
        quad = (self.hbar_moyal * br) ** 2
        c = quad.mean(axis=0)
        stderr = quad.std(axis=0, ddof=1) / np.sqrt(quad.shape[0])
        zero = times == 0.0
        c[zero] = 0.0
        stderr[zero] = 0.0
        return c, stderr

    def _grid(self, times: np.ndarray) -> np.ndarray:
        """Integration grid: ``times`` when it starts at 0, else ``[0] + times``."""
        if times[0] == 0.0:
            return times
        return np.concatenate(([0.0], times))

    def _propagate_one(self, z: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """One engine trajectory on the full integration grid (internal)."""
        traj = np.asarray(self.engine.trajectory(z, grid), dtype=float)
        if traj.shape != (grid.size, self.engine.dim):
            raise ValueError(
                f"TruncatedWigner: engine.trajectory returned shape {traj.shape}, "
                f"expected ({grid.size}, {self.engine.dim})"
            )
        return traj
