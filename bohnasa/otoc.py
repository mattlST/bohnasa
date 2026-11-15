"""Out-of-time-order squared-commutator growth (OTOC).

The squared commutator ``C(t) = || [O(t), O] |psi> ||^2`` with
``O(t) = exp(iHt) O exp(-iHt)`` is the scrambling diagnostic of quantum
chaos. Evaluated by the VECTOR TRICK,

    [O(t), O]|psi> = exp(iHt) O exp(-iHt) (O|psi>)
                     - O exp(iHt) O exp(-iHt) |psi>,

which needs only state propagations and never assembles the dim^2 x dim^2
Liouville machinery.

Scope of the operator form: the *same-operator* (auto-)commutator
``[O(t), O]``; the general asymmetric ``[O_1(t), O_2]`` and thermal
regularizations are not implemented (follow-up issues). Pure states
only; density-matrix input is rejected.
The classical reference line for the growth rates is
:mod:`bohnasa.classical_mean_field`.
"""

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .operators import Operator

__all__ = ["SquaredCommutator"]

_DENSE_EIG_LIMIT = 2000
# columns per spectral batch: caps the dim x block complex temporaries at
# ~130 MB for dim ~8200 while staying in the BLAS-3 regime
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
    probe state. Propagation paths:

    - ``method="eig"`` (default for ``dim <= 2000`` or tridiagonal ``H``):
      one eigendecomposition (STEMR for tridiagonal), curve evaluated as
      blocked spectral matrix-products, ``O(dim^2)`` memory;
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
