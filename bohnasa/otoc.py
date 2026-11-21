"""Out-of-time-order squared-commutator growth (OTOC).

The squared commutator ``C(t) = || [O(t), O] |psi> ||^2`` with
``O(t) = exp(iHt) O exp(-iHt)`` is the scrambling diagnostic of quantum
chaos. Evaluated by the VECTOR TRICK,

    [O(t), O]|psi> = exp(iHt) O exp(-iHt) (O|psi>)
                     - O exp(iHt) O exp(-iHt) |psi>,

which needs only state propagations and never assembles the dim^2 x dim^2
Liouville machinery.

Scope of the operator form: the *same-operator* (auto-)commutator
``[O(t), O]``; the general asymmetric ``[O_1(t), O_2]`` is not
implemented (follow-up issue). Thermal states enter through the Gibbs
expectation of the squared commutator,
``C_beta(t) = Tr([O(t), O]^dag [O(t), O] rho_beta)``
(:meth:`SquaredCommutator.thermal_curve`), evaluated as a
Boltzmann-weighted contraction over the energy eigenbasis - still no
Liouville-space assembly. ``curve`` and ``commuted`` remain pure-state
probes; generic (non-Gibbs) density-matrix evolution is out of scope.
The classical reference line for the growth rates is
:mod:`bohnasa.classical_mean_field`.
"""

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .operators import Operator

__all__ = ["SquaredCommutator"]

# Auto-dispatch threshold for the dense eigensolve fallback. It bounds the
# one-shot O(dim^3) LAPACK cost, measured on the non-tridiagonal 2-particle
# chain (16-core host, fresh process per point; benchmarks/propagation_scaling.py
# reproduces the shape, a warm single process runs somewhat higher):
# eigensolve 0.18 s at dim 861, 2.5 s at 1770, 5.0 s at 2080, 16.6 s at
# 2926; peak RSS ~340-780 MB where the krylov path stays under 150 MB.
# The STEMR path is exempt at any size - for width-1 Hamiltonians the full
# eigendecomposition is O(dim^2). Below the limit the eigensolve amortizes
# over the curve: a 2001-point curve costs 0.3 s, vs 13.4 s for per-point
# expm_multiply at dim 861; above it, short curves take the O(dim)-memory
# krylov path and long-curve callers pass method="eig" explicitly. The
# dispatch boundary is pinned by
# tests/unit/test_otoc.py::test_method_auto_uses_stemr_for_tridiagonal_above_the_dense_limit;
# retune with a fresh run of the benchmark above on the target host.
_DENSE_EIG_LIMIT = 2000
# columns per spectral batch: caps the dim x block complex temporaries at
# ~130 MB for dim ~8200 while staying in the BLAS-3 regime - measured flat:
# a 4096-time curve at dim 2925 takes 8.0-8.6 s for any block from 128 to
# 4096 (16-core host), so the block size buys memory headroom, not
# throughput; at 1024 columns each temporary is 16 KB per row.
_SPECTRAL_BLOCK = 1024


def _is_tridiagonal(h: sp.csr_matrix) -> bool:
    """Hermitian and band width 1: the STEMR eigensolve path is O(dim^2),
    so it beats per-time-point krylov recursion at any size."""
    d = np.asarray(h.diagonal(), dtype=float)
    if d.size < 2:
        return False
    e = np.asarray(h.diagonal(1))
    lower = np.asarray(h.diagonal(-1))
    if e.size != d.size - 1 or not np.allclose(lower, e.conj()):
        return False
    diff = h - sp.diags([e, d, e.conj()], [-1, 0, 1], shape=h.shape, format="csr")
    diff.eliminate_zeros()
    return diff.nnz == 0


def _hermitian_eigh(h: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Eigenpairs of a Hermitian sparse matrix: O(dim^2) STEMR for
    symmetric-tridiagonal input, dense LAPACK otherwise."""
    if _is_tridiagonal(h):
        d = np.asarray(h.diagonal(), dtype=float)
        e = np.asarray(h.diagonal(1))
        return sla.eigh_tridiagonal(d, e, check_finite=False)
    return np.linalg.eigh(np.asarray(h.toarray(), dtype=complex))


class SquaredCommutator:
    r"""Squared commutator ``[O(t), O]`` applied by state propagation only.

    Binds a Hamiltonian and an operator once (mirroring
    :class:`~bohnasa.krylov.Liouvillian`), then evaluates ``C(t)`` for any
    probe state - or the Gibbs-state curve ``C_beta(t)`` via
    :meth:`thermal_curve`. Propagation paths:

    - ``method="eig"`` (default for ``dim <= 2000`` or tridiagonal ``H``):
      one eigendecomposition (STEMR for tridiagonal), curve evaluated as
      blocked spectral matrix-products, ``O(dim^2)`` memory - the only
      path that also serves :meth:`thermal_curve` (it needs the full
      energy eigenbasis for the Boltzmann weights);
    - ``method="krylov"`` (default above the threshold): ``expm_multiply``
      per propagation, ``O(dim)`` memory - the path that reaches the large
      systems;
    - :meth:`from_propagator`: Floquet driving, ``times`` are kick counts.

    Parameters
    ----------
    hamiltonian : Operator
        Time-independent Hamiltonian with a matrix representation.
    operator : Operator
        Probe operator ``O`` (e.g. a site ``Number`` or the dimer
        imbalance); any ``Operator``/matrix-like with ``matrix``.
    method : {"auto", "eig", "krylov"}, default "auto"

    Examples
    --------
    >>> import numpy as np
    >>> from bohnasa import FixedNFockSpace, Hamiltonian, Number
    >>> fock = FixedNFockSpace(4, 2)
    >>> J, U = 1.0, 0.5
    >>> t = np.array([[0.0, -2 * J], [-2 * J, 0.0]])
    >>> H = Hamiltonian(fock, one_body=t, onsite_u=U)
    >>> O = (Number(fock, 0) - Number(fock, 1)) * 0.5
    >>> sc = SquaredCommutator(H, O)
    >>> times = np.linspace(0.0, 2.0, 11)
    >>> state = np.zeros(fock.dim, dtype=complex)
    >>> state[fock.dict[(0, 4)]] = 1.0
    >>> float(sc.curve(state, times)[0])
    0.0
    """

    def __init__(self, hamiltonian: Operator, operator: Operator, method: str = "auto") -> None:
        if hamiltonian.matrix is None:
            raise ValueError("SquaredCommutator: hamiltonian has no matrix representation")
        if operator.matrix is None:
            raise ValueError("SquaredCommutator: operator has no matrix representation")
        h = sp.csr_matrix(hamiltonian.matrix)
        if h.shape[0] != h.shape[1]:
            raise ValueError(f"SquaredCommutator: hamiltonian must be square, got {h.shape}")
        self._o = sp.csr_matrix(operator.matrix)
        if self._o.shape != h.shape:
            raise ValueError(
                f"SquaredCommutator: operator shape {self._o.shape} != hamiltonian {h.shape}"
            )
        self.dim = int(h.shape[0])
        self._h: sp.csr_matrix | None = h
        self._propagator: sp.csr_matrix | None = None
        self._eps: np.ndarray | None = None
        self._vecs: np.ndarray | None = None
        self._htrace: complex | None = None
        self._thermal_w: np.ndarray | None = None
        if method == "auto":
            # a tridiagonal Hamiltonian (dimer: always; ring fixed-N: sparse
            # but wider band) is cheap to fully diagonalize at any size, so
            # the size threshold only guards the dense-eig fallback
            tridiag = _is_tridiagonal(h)
            method = "eig" if (self.dim <= _DENSE_EIG_LIMIT or tridiag) else "krylov"
        if method not in ("eig", "krylov"):
            raise ValueError(f"SquaredCommutator: unknown method {method!r}")
        self.method = method
        if method == "eig":
            self._eps, self._vecs = _hermitian_eigh(h)
        else:
            self._htrace = complex(np.sum(h.diagonal()))

    @classmethod
    def from_propagator(cls, propagator: np.ndarray, operator: Operator) -> "SquaredCommutator":
        """Bind a Floquet step unitary instead of a Hamiltonian.

        ``curve(state, times)`` then takes integer (or integer-valued float)
        kick counts; ``C(n)`` uses ``O(n) = U^dagger^n O U^n``.
        """
        u = sp.csr_matrix(np.asarray(propagator, dtype=complex))
        obj = cls.__new__(cls)
        obj.dim = int(u.shape[0])
        obj._o = sp.csr_matrix(operator.matrix)
        if obj._o.shape != u.shape:
            raise ValueError(
                f"SquaredCommutator: operator shape {obj._o.shape} != propagator {u.shape}"
            )
        obj._h = None
        obj._eps = None
        obj._vecs = None
        obj._thermal_w = None
        obj._propagator = u
        obj.method = "floquet"
        return obj

    def _backward(self, vec: np.ndarray, t: float) -> np.ndarray:
        """Apply exp(-iHt)."""
        if self._propagator is not None:
            out = vec
            for _ in range(int(t)):
                out = self._propagator @ out
            return out
        if self.method == "eig":
            assert self._vecs is not None and self._eps is not None  # set by __init__
            return self._vecs @ (np.exp(-1j * self._eps * t) * (self._vecs.conj().T @ vec))
        assert self._h is not None and self._htrace is not None  # set by __init__
        return spla.expm_multiply(
            spla.aslinearoperator(-1j * t * self._h), vec, traceA=(-1j * t) * self._htrace
        )

    def _forward(self, vec: np.ndarray, t: float) -> np.ndarray:
        """Apply exp(+iHt) (adjoint propagation)."""
        if self._propagator is not None:
            out = vec
            for _ in range(int(t)):
                out = self._propagator.conj().T @ out
            return out
        if self.method == "eig":
            assert self._vecs is not None and self._eps is not None  # set by __init__
            return self._vecs @ (np.exp(1j * self._eps * t) * (self._vecs.conj().T @ vec))
        assert self._h is not None and self._htrace is not None  # set by __init__
        return spla.expm_multiply(
            spla.aslinearoperator(1j * t * self._h), vec, traceA=(1j * t) * self._htrace
        )

    def _validated(self, state: np.ndarray) -> np.ndarray:
        psi = np.asarray(state, dtype=complex)
        if psi.ndim == 2:
            raise ValueError(
                "SquaredCommutator: mixed states are not supported; pass a state vector "
                f"of shape ({self.dim},)"
            )
        if psi.shape != (self.dim,):
            raise ValueError(
                f"SquaredCommutator: state must have shape ({self.dim},), got {psi.shape}"
            )
        return psi

    def commuted(self, state: np.ndarray, t: float) -> np.ndarray:
        """The vector ``[O(t), O] |psi>`` at time ``t`` (vector trick)."""
        psi = self._validated(state)
        o_psi = self._o @ psi
        if t == 0:
            # O(0) = O exactly; return an exact zero, not eig round-trip noise
            return np.zeros(self.dim, dtype=complex)
        term1 = self._forward(self._o @ self._backward(o_psi, t), t)
        term2 = self._o @ self._forward(self._o @ self._backward(psi, t), t)
        return term1 - term2

    def curve(self, state: np.ndarray, times: np.ndarray, normalize: bool = False) -> np.ndarray:
        """``C(t) = || [O(t), O] |psi> ||^2`` at the requested times.

        Parameters
        ----------
        state : numpy.ndarray
            Probe state vector of shape ``(dim,)`` (mixed states / density
            matrices are not supported; not assumed normalized - see
            ``normalize``).
        times : numpy.ndarray
            Times (Floquet mode: integer kick counts).
        normalize : bool, default False
            If True, divide by the pure-state weight ``<psi|O^dag O|psi>``
            so the curve measures the growth relative to the operator's
            initial weight.

        Returns
        -------
        numpy.ndarray of float
            Nonnegative curve, ``C(t) = 0`` exactly at ``t = 0`` and
            identically zero when ``O`` commutes with ``H``.
        """
        if self.method == "floquet":
            if not np.allclose(times, np.round(times)) or np.any(np.asarray(times) < 0):
                raise ValueError("SquaredCommutator: floquet times must be nonnegative integers")
        psi = self._validated(state)
        times = np.asarray(times, dtype=float)
        weight = float(np.vdot(self._o @ psi, self._o @ psi).real)
        if self.method == "eig":
            c = self._curve_spectral(psi, times)
        else:
            c = np.array([np.vdot(v, v).real for v in (self.commuted(psi, t) for t in times)])
        if normalize:
            if weight == 0.0:
                raise ValueError(
                    "SquaredCommutator: state annihilated by operator, cannot normalize"
                )
            return c / weight
        return c

    def _curve_spectral(self, psi: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Vector trick evaluated as blocked matrix-matrix products.

        All time points propagate simultaneously: ``exp(-/+iHt)v`` for
        every t is ``V @ (V^dag v * exp(-/+i eps t))`` - two dense gemms
        per stage instead of a gemv per time point, so the curve scales
        with multithreaded BLAS throughput and reaches system sizes where
        the per-point krylov recursion crawls.
        """
        assert self._vecs is not None and self._eps is not None  # eig path only
        v, eps = self._vecs, self._eps
        o_psi = self._o @ psi
        seed0 = v.conj().T @ psi
        seed1 = v.conj().T @ o_psi
        out = np.empty(len(times), dtype=float)
        for s in range(0, len(times), _SPECTRAL_BLOCK):
            tt = times[s : s + _SPECTRAL_BLOCK]
            ph = np.exp(-1j * np.outer(eps, tt))
            # term1 = exp(+iHt) O exp(-iHt) (O psi)
            s1 = v @ (seed1[:, None] * ph)
            u1 = self._o @ s1
            term1 = v @ ((v.conj().T @ u1) * ph.conj())
            # term2 = O exp(+iHt) O exp(-iHt) psi
            s0 = v @ (seed0[:, None] * ph)
            u0 = self._o @ s0
            f0 = v @ ((v.conj().T @ u0) * ph.conj())
            term2 = self._o @ f0
            d = term1 - term2
            out[s : s + len(tt)] = (np.abs(d) ** 2).sum(axis=0)
        out[times == 0] = 0.0  # exact, like commuted() short-circuits
        return out

    def _operator_in_eigenbasis(self) -> np.ndarray:
        """``O`` in the energy eigenbasis, computed once and cached.

        Follows the ``H.eigval``/``H.eigvec`` lazy-caching pattern of
        :func:`bohnasa.time_evolution.time_evolution_exact_ind`; the cache
        lives on this instance because the matrix needs the Hamiltonian
        eigenvectors and the probe operator together. Dense ``dim x dim``
        conversion is justified: the spectral thermal contraction is an
        ``O(dim^2)``-memory / ``O(dim^3)``-per-time-point path of the same
        memory class as :meth:`_curve_spectral` (never the ``dim^2 x dim^2``
        Liouville operator).
        """
        if self._thermal_w is None:
            assert self._vecs is not None  # eig path, checked by caller
            self._thermal_w = self._vecs.conj().T @ (self._o @ self._vecs)
        return self._thermal_w

    def thermal_curve(
        self, times: np.ndarray, beta: float, regularization: str = "none"
    ) -> np.ndarray:
        r"""Gibbs-state squared commutator ``C_beta(t)`` at inverse temperature ``beta``.

        For ``rho_beta = exp(-beta H) / Z(beta)`` with ``Z(beta) = Tr exp(-beta H)``,
        the thermal squared commutator is the Gibbs expectation

        .. math::
            C_\beta(t) = \mathrm{Tr}\!\left( [O(t),O]^\dagger [O(t),O] \, \rho_\beta
            \right) = \sum_m p_m \, \bigl\lVert [O(t),O] \lvert E_m\rangle
            \bigr\rVert^2 , \qquad p_m = e^{-\beta E_m} / Z(\beta) .

        In the energy eigenbasis ``rho_beta`` is diagonal, so the trace becomes a
        Boltzmann-weighted contraction over eigenstates: ``[O(t), O]`` is built
        from the cached eigenbasis matrix :meth:`_operator_in_eigenbasis` by two
        dense ``dim x dim`` products per time point, mirroring the blocked
        gemm philosophy of :meth:`_curve_spectral` - no eigenvector is ever
        propagated and no ``dim^2 x dim^2`` Liouville operator is assembled.
        The weights use ``exp(-beta * (E_m - E_0))``: the ground shift cancels
        exactly against ``Z(beta)``, cannot overflow, and underflows cleanly to
        zero, so large ``beta`` is numerically stable.

        Parameters
        ----------
        times : numpy.ndarray
            Times of the curve (any real values; ``t = 0`` yields exactly 0).
        beta : float
            Inverse temperature, finite and ``beta >= 0``. ``beta = 0`` is the
            maximally-mixed (infinite-temperature) state; large ``beta``
            approaches the ground state (see Examples).
        regularization : {"none", "symmetric"}, default "none"
            ``"none"``: the definition above - the Gibbs weight enters once,
            on the traced slot. ``"symmetric"``: the regularized convention
            (thesis Sec. 7.1, App. D), which shares the state weight
            symmetrically over the correlator: ``O_tilde = rho_beta^{1/8} O
            rho_beta^{1/8}`` has eigenbasis elements ``(p_n p_m)^{1/8} O_{nm}``
            and the curve is ``||[O_tilde(t), O_tilde]||_F^2``. Each external
            index then carries the eighth root of the weights from its own
            vertex (total ``(p_n p_m)^{1/4}`` per squared element) and every
            contracted intermediate eigenstate ``p_k^{1/4}``, so thermally
            unpopulated levels enter suppressed by their own occupation. As
            ``beta -> infinity`` the weighted operator collapses to
            ``<O>_0 |E_0><E_0|``, commuting with its own time evolution, and
            the regularized curve vanishes identically: the trivial
            identity/pure-state contribution - the part of the unregularized
            curve that survives as the ground-state curve at zero temperature -
            is removed, leaving only the genuinely mixed-state growth. At
            ``beta = 0`` both conventions coincide to machine precision.

        Returns
        -------
        numpy.ndarray of float
            Nonnegative curve, ``C_beta(t) >= 0`` (sum of weighted squared
            norms), ``C_beta(0) = 0`` exactly and identically zero when ``O``
            commutes with ``H``, for every ``beta`` and either regularization.

        Raises
        ------
        ValueError
            If the instance is not on the ``"eig"`` path (the thermal curve
            needs the full energy eigenbasis: rebuild with ``method="eig"``),
            or ``beta`` is not finite/nonnegative, or ``regularization`` is
            unknown, or ``times`` is not one-dimensional.

        Notes
        -----
        Cost per time point is two dense ``dim x dim`` complex matrix
        products: ``O(dim^3)`` flops, ``O(dim^2)`` memory - measured on a
        16-core CPU at 0.5 s ("none") / 0.9 s ("symmetric") for a 25-point
        curve at ``dim = 561`` (the interacting trimer at ``N = 32``). A
        literal mixed-state generalization of :meth:`curve` would instead
        propagate all ``dim`` eigenvectors per time point; the spectral
        contraction replaces those propagations entirely.

        Examples
        --------
        >>> import numpy as np
        >>> from bohnasa import FixedNFockSpace, Hamiltonian, Number
        >>> fock = FixedNFockSpace(6, 2)
        >>> t = np.array([[0.0, -2.0], [-2.0, 0.0]])
        >>> H = Hamiltonian(fock, one_body=t, onsite_u=0.8)
        >>> O = (Number(fock, 0) - Number(fock, 1)) * 0.5
        >>> sc = SquaredCommutator(H, O)
        >>> times = np.linspace(0.0, 2.0, 9)
        >>> float(sc.thermal_curve(times, beta=0.5)[0])
        0.0
        >>> float(np.round(sc.thermal_curve(times, 0.5)[-1], 10))
        5.2557003433
        >>> float(np.round(sc.thermal_curve(times, 0.5, regularization="symmetric")[-1], 10))
        1.1229170006
        >>> ground = np.linalg.eigh(H.matrix.toarray())[1][:, 0]
        >>> bool(np.allclose(sc.thermal_curve(times, 1000.0), sc.curve(ground, times)))
        True
        >>> bool(np.allclose(sc.thermal_curve(times, 0.0),
        ...                  sc.thermal_curve(times, 0.0, regularization="symmetric")))
        True
        >>> float(sc.thermal_curve(times, 1000.0, regularization="symmetric").max())
        0.0
        """
        if self.method != "eig":
            raise ValueError(
                "SquaredCommutator: thermal_curve requires the eig path (the full "
                f"energy eigenbasis), got method={self.method!r}; rebuild with method='eig'"
            )
        assert self._eps is not None  # set by __init__ for method == "eig"
        if regularization not in ("none", "symmetric"):
            raise ValueError(f"SquaredCommutator: unknown regularization {regularization!r}")
        b = float(beta)
        if not np.isfinite(b) or b < 0:
            raise ValueError(f"SquaredCommutator: beta must be finite and nonnegative, got {beta}")
        eps = self._eps
        times = np.asarray(times, dtype=float)
        if times.ndim != 1:
            raise ValueError(f"SquaredCommutator: times must be one-dimensional, got {times.shape}")
        p = np.exp(-b * (eps - eps[0]))
        p /= p.sum()
        o_e = self._operator_in_eigenbasis()
        if regularization == "symmetric":
            # geometric-mean weight (p_n p_m)^(1/8) on both slots of O
            o_e = o_e * (p**0.125)[:, None] * (p**0.125)[None, :]
        # "none": Gibbs weight sits fully on the traced slot (columns of A);
        # "symmetric": the weights are already inside o_e (plain Frobenius)
        col = np.sqrt(p) if regularization == "none" else None
        out = np.empty(len(times), dtype=float)
        for i, tt in enumerate(times):
            if tt == 0:
                out[i] = 0.0  # O(0) = O: the commutator vanishes exactly
                continue
            ph = np.exp(-1j * eps * tt)
            o_t = o_e * (np.conj(ph)[:, None] * ph[None, :])  # O(t) in eigenbasis
            d = o_t @ o_e - o_e @ o_t
            if col is not None:
                d = d * col[None, :]
            out[i] = float(np.sum(np.abs(d) ** 2))
        return out
