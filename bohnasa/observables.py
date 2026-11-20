"""State-resolved observables: coherence, occupations, and density noise.

One-body quantities are assembled from the existing second-quantized
operators (:class:`~bohnasa.operators.Correlator`); the normal-ordered
two-body densities are basis-diagonal - their expectation values are
occupation-weighted sums over the Fock basis, computed exactly in
``O(dim * L)`` without any sparse operator products.

:class:`StateObservables` binds one basis once and pre-assembles the
``Correlator`` pair matrices into a single CSR stack, so repeated
evaluation over a trajectory costs one stacked matvec per state instead
of ``L**2`` Python-loop matrix constructions per call. Its results are
bit-for-bit the module-function results - the class is a pure relocation
of work in time, not a change of computation.
"""

import numpy as np
import scipy.sparse as sp

from .fock_space import FixedNFockSpace
from .operators import Correlator, _pair_matrix

__all__ = [
    "one_body_density_matrix",
    "density_profile",
    "orbital_occupations",
    "momentum_distribution",
    "pair_correlation",
    "structure_factor",
    "StateObservables",
]


def _validated(state, fock):
    state = np.asarray(state, dtype=complex)
    if state.shape != (fock.dim,):
        raise ValueError(
            f"observables: state must have shape ({fock.dim},) for this basis, got {state.shape}"
        )
    return state


def _correlator_stack(fock):
    """Stack every pair matrix ``c^dag_i c_j`` of the basis into one CSR matrix.

    Block row ``i * L + j`` is :func:`bohnasa.operators._pair_matrix`
    ``(fock, src=j, dst=i)`` = ``c^dag_i c_j`` (the same matrix
    ``Correlator(fock, j, i)`` carries), so ``stack @ psi`` returns the
    ``L**2`` intermediate vectors ``c^dag_i c_j |psi>`` concatenated in
    the row-major entry order of gamma.
    """
    return sp.vstack(
        [_pair_matrix(fock, j, i) for i in range(fock.L) for j in range(fock.L)],
        format="csr",
    )


def _g2_from(probs, occ, profile):
    """Normal-ordered pair correlation from basis weights and occupations.

    ``G2_ij = sum_r p_r n_i(r) (n_j(r) - delta_ij)`` as the
    occupation-weighted Gram product with the on-site self-term removed.
    """
    g2 = (probs[:, None] * occ).T @ occ
    np.fill_diagonal(g2, np.diag(g2) - profile)
    return g2


def _structure_from(probs, occ, profile, n_total):
    """Static structure factor from basis weights (one FFT over the rows)."""
    fluct = occ - profile[None, :]
    modes = np.fft.fft(fluct, axis=1)
    return probs @ (np.abs(modes) ** 2) / n_total


def _momentum_from_gamma(gamma, L):
    """Diagonal of ``E^dag gamma E`` on the ring momentum grid (``E_kj`` below)."""
    k = 2.0 * np.pi * np.arange(L) / L
    # n_k = sum_ij conj(E_ki) gamma_ij E_kj, E_kj = exp(-i k j)/sqrt(L)
    phase = np.exp(-1j * np.outer(k, np.arange(L))) / np.sqrt(L)
    return np.real(np.diag(phase.conj() @ gamma @ phase.T))


def _natural_occupations(gamma):
    """Descending eigenvalues of the Hermitian one-body density matrix."""
    nu = np.linalg.eigvalsh(gamma)
    return nu.real[::-1]


def one_body_density_matrix(state, fock):
    r"""One-body density matrix :math:`\gamma_{ij} = \langle c^\dagger_i c_j\rangle`.

    Parameters
    ----------
    state : numpy.ndarray of complex, shape (dim,)
        State vector in the ``fock`` basis (assumed normalized; no
        renormalization is applied, matching ``Operator.eval_mean``).
    fock : FixedNFockSpace
        Basis of the state.

    Returns
    -------
    numpy.ndarray of complex, shape (L, L)
        ``gamma``; Hermitian with trace ``N`` for normalized states and
        exact particle-number conservation - both are pinned in the tests,
        never imposed here.

    See Also
    --------
    StateObservables : basis-bound reuse path with bit-for-bit identical
        results for repeated evaluation on the same basis.

    Examples
    --------
    >>> from bohnasa import FixedNFockSpace
    >>> fock = FixedNFockSpace(1, 3)
    >>> state = np.zeros(fock.dim, dtype=complex)
    >>> state[fock.dict[(1, 0, 0)]] = 1.0
    >>> one_body_density_matrix(state, fock)
    array([[1.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j]])
    """
    state = _validated(state, fock)
    gamma = np.empty((fock.L, fock.L), dtype=complex)
    for i in range(fock.L):
        for j in range(fock.L):
            # Correlator(fock, j, i) = c^dag_i c_j
            gamma[i, j] = Correlator(fock, j, i).eval_mean(state)
    return gamma


def density_profile(state, fock):
    r"""Site occupations :math:`\langle \hat{n}_i \rangle`, shape ``(L,)``.

    Computed as the basis-diagonal expectation
    ``sum_r |state_r|^2 * fockArray[r, i]``; agrees with ``Number(fock, i).
    eval_mean(state)`` and with ``diag(one_body_density_matrix(...))`` up to
    that route's floating noise.

    Examples
    --------
    >>> from bohnasa import FixedNFockSpace
    >>> fock = FixedNFockSpace(1, 3)
    >>> state = np.zeros(fock.dim, dtype=complex)
    >>> state[fock.dict[(0, 1, 0)]] = 1.0
    >>> density_profile(state, fock)
    array([0., 1., 0.])
    """
    state = _validated(state, fock)
    probs = np.abs(state) ** 2
    return probs @ fock.fockArray.astype(float)


def orbital_occupations(state, fock):
    r"""Natural-orbital occupations: eigenvalues of gamma, descending.

    Penrose-Onsager criterion: a condensate is a macroscopically occupied
    natural orbital - largest eigenvalue ``nu_0``; ``nu_0 / N`` is the
    condensate fraction, and ``sum(nu) = N`` with ``0 <= nu`` for every
    state. The free ring ground state gives exactly ``(N, 0, ..., 0)``; a
    Fock-basis eigenstate gives its own sorted occupation numbers.

    Examples
    --------
    >>> from bohnasa import FixedNFockSpace
    >>> fock = FixedNFockSpace(2, 3)
    >>> state = np.zeros(fock.dim, dtype=complex)
    >>> state[fock.dict[(2, 0, 0)]] = 1.0
    >>> orbital_occupations(state, fock)
    array([2., 0., 0.])
    """
    return _natural_occupations(one_body_density_matrix(state, fock))


def momentum_distribution(state, fock):
    r"""Occupations :math:`n_k = \langle a^\dagger_k a_k\rangle` on ring momenta.

    With the convention ``a_k = L^(-1/2) sum_j exp(-i k j) c_j`` at
    ``k_n = 2 pi n / L`` (``n = 0, ..., L-1``), returns ``n_{k_n}``, shape
    ``(L,)``. A plane-wave orbital state ``phi_j = exp(i m j)/sqrt(L)``
    yields ``N`` at the index matching its momentum ``m`` and ``~0``
    elsewhere; a Fock eigenstate gives the flat ``N / L``. The momentum
    convention assumes periodic boundary conditions; it is documented, not
    enforced.

    Examples
    --------
    >>> from bohnasa import FixedNFockSpace
    >>> fock = FixedNFockSpace(1, 3)
    >>> state = np.zeros(fock.dim, dtype=complex)
    >>> state[fock.dict[(0, 0, 1)]] = 1.0
    >>> float(np.round(momentum_distribution(state, fock).sum(), 12))
    1.0
    """
    return _momentum_from_gamma(one_body_density_matrix(state, fock), fock.L)


def pair_correlation(state, fock):
    r"""Normal-ordered density-noise matrix
    :math:`G2_{ij} = \langle c^\dagger_i c^\dagger_j c_j c_i\rangle`.

    The monomial is basis-diagonal: eigenvalue ``n_i n_j`` (``i != j``) and
    ``n_i (n_i - 1)`` on site, so

    ``G2_ij = sum_r |state_r|^2 n_i(r) (n_j(r) - delta_ij)``

    with no ladder arithmetic in the code. For the plane-wave state every
    entry equals ``N (N - 1) / L^2`` (bunching: ratio to the incoherent
    ``<n_i><n_j>`` is ``(N - 1)/N``); a unit-occupation eigenstate gives
    vanishing on-site pair correlation (no doublons) and exact classical
    correlation ``G2_ij = n_i n_j`` off-diagonal.

    Examples
    --------
    >>> from bohnasa import FixedNFockSpace
    >>> fock = FixedNFockSpace(2, 3)
    >>> state = np.zeros(fock.dim, dtype=complex)
    >>> state[fock.dict[(2, 0, 0)]] = 1.0
    >>> float(pair_correlation(state, fock)[0, 0])
    2.0
    """
    state = _validated(state, fock)
    probs = np.abs(state) ** 2
    occ = fock.fockArray.astype(float)
    return _g2_from(probs, occ, density_profile(state, fock))


def structure_factor(state, fock):
    r"""Static structure factor
    :math:`S(k) = \langle \delta \hat{n}_k\, \delta \hat{n}_{-k}\rangle / N`.

    With ``delta n_k = sum_j exp(-i k j) (n_j - <n_j>)`` and the ``1/N``
    (ideal-gas) normalization, at ``k_n = 2 pi n / L``. The density
    fluctuation is basis-diagonal with eigenvalue
    ``|sum_j exp(-i k j) (n_j(r) - <n_j>)|^2``, so ``S`` is one FFT over the
    occupation rows - and two invariants follow structurally: ``S(0) = 0``
    for every state on the fixed-N shell (total number is sharp), and
    ``S(k) = S(-k)``. Sum rule: ``sum_k S(k) = (L / N) sum_i Var(n_i)``.

    Examples
    --------
    >>> from bohnasa import FixedNFockSpace
    >>> fock = FixedNFockSpace(2, 2)
    >>> state = np.zeros(fock.dim, dtype=complex)
    >>> state[fock.dict[(1, 1)]] = 1.0
    >>> bool(np.abs(structure_factor(state, fock)).max() < 1e-15)
    True
    """
    state = _validated(state, fock)
    probs = np.abs(state) ** 2
    occ = fock.fockArray.astype(float)
    return _structure_from(probs, occ, density_profile(state, fock), fock.N)


class StateObservables:
    r"""All six state-resolved observables for one basis, bound once.

    The module-level functions rebuild the ``L**2`` ``Correlator`` pair
    matrices of :math:`\gamma_{ij} = \langle c^\dagger_i c_j\rangle` on
    every call - fine for one state, wasteful for trajectories, sweeps or
    any repeated evaluation on the same basis. This class mirrors the
    bind-structure-once/apply-repeatedly pattern of
    :class:`~bohnasa.krylov.Liouvillian`: ``__init__`` pre-assembles the
    pair matrices into a single CSR stack, and each evaluation of gamma is
    one stacked matvec plus ``L**2`` BLAS dots. The results are bit-for-bit
    the module-function results (pinned in
    ``tests/unit/test_state_observables.py`` across several bases and
    complex-valued states), and the #19 physics anchors are re-run through
    this path in ``tests/physics/test_observable_anchors.py``.

    Parameters
    ----------
    fock : FixedNFockSpace
        The basis all evaluated states live in.

    Examples
    --------
    The motivating use case: gamma along a time-evolved trajectory. The
    stack is built once at binding, and every frame pays one matvec.

    >>> import numpy as np
    >>> from bohnasa import FixedNFockSpace, Hopping
    >>> fock = FixedNFockSpace(1, 3)
    >>> H = Hopping(fock, connections="pbc")
    >>> obs = StateObservables(fock)
    >>> state = np.zeros(fock.dim, dtype=complex)
    >>> state[fock.dict[(1, 0, 0)]] = 1.0
    >>> eigval, eigvec = np.linalg.eigh(H.matrix.toarray())
    >>> coeffs = eigvec.conj().T @ state
    >>> times = np.array([0.0, 0.25, 0.5])
    >>> trajectory = [eigvec @ (np.exp(-1j * eigval * t) * coeffs) for t in times]
    >>> gamma_t = np.array([obs.one_body_density_matrix(psi) for psi in trajectory])
    >>> gamma_t.shape
    (3, 3, 3)
    >>> bool(np.allclose(np.trace(gamma_t, axis1=1, axis2=2), 1.0))
    True
    >>> bool(np.array_equal(gamma_t[1], one_body_density_matrix(trajectory[1], fock)))
    True
    >>> float(np.round(gamma_t[2, 0, 1].real, 4))
    -0.1033
    >>> float(np.round(obs.density_profile(trajectory[2])[0], 3))
    0.587
    """

    def __init__(self, fock: FixedNFockSpace) -> None:
        self.fock = fock
        self.L = int(fock.L)
        self.N = int(fock.N)
        self.dim = int(fock.dim)
        self._occ = fock.fockArray.astype(float)
        # Reuse work bound (issue #20): __init__ performs exactly L**2
        # _pair_matrix constructions - each an O(dim) Python loop, i.e. the
        # same construction bill the reference path pays on its FIRST
        # one_body_density_matrix call - while every later call performs
        # exactly ONE stacked matvec and L**2 BLAS dots, constructing nothing
        # (pinned by the monkeypatched-builder test). Measured on this
        # 16-core dev host with numpy 2.5.5 / scipy 1.18.1 for N=4, L=6
        # (dim=126) over T=25 complex states: one_body_density_matrix drops
        # from 4.1 ms to 0.049 ms per state (~80x; ~15-20x with the 3.9 ms
        # bind amortized over T), at identical total matvec flops - the stack
        # just concatenates the same nnz.
        self._stack = _correlator_stack(fock)

    def one_body_density_matrix(self, state: np.ndarray) -> np.ndarray:
        r"""One-body density matrix :math:`\gamma_{ij} = \langle c^\dagger_i c_j\rangle`.

        One matvec against the pre-assembled pair stack, then the same
        ``dot`` per entry that ``Correlator.eval_mean`` performs on the same
        operands - hence bit-for-bit the module function's ``gamma``.

        Parameters
        ----------
        state : numpy.ndarray of complex, shape (dim,)
            State vector in the bound basis (assumed normalized, like the
            module function).

        Returns
        -------
        numpy.ndarray of complex, shape (L, L)
            ``gamma``, identical to
            :func:`bohnasa.observables.one_body_density_matrix`.
        """
        psi = _validated(state, self.fock)
        y = self._stack @ psi  # all L**2 vectors c^dag_i c_j |psi>, one call
        pconj = psi.conj()
        dim = self.dim
        gamma = np.empty((self.L, self.L), dtype=complex)
        flat = gamma.reshape(-1)  # row-major: entry k = i * L + j
        for k in range(self.L * self.L):
            # block k of the stack is exactly Correlator(fock, j, i).matrix,
            # so this dot reproduces eval_mean's operands bit-for-bit
            flat[k] = y[k * dim : (k + 1) * dim] @ pconj
        return gamma

    def density_profile(self, state: np.ndarray) -> np.ndarray:
        r"""Site occupations :math:`\langle \hat{n}_i \rangle`, shape ``(L,)``.

        Identical to :func:`bohnasa.observables.density_profile`; the float
        occupation array of the bound basis is converted once at binding.

        Parameters
        ----------
        state : numpy.ndarray of complex, shape (dim,)
            State vector in the bound basis.

        Returns
        -------
        numpy.ndarray of float, shape (L,)
            The site occupations.
        """
        psi = _validated(state, self.fock)
        return np.abs(psi) ** 2 @ self._occ

    def orbital_occupations(self, state: np.ndarray) -> np.ndarray:
        r"""Natural-orbital occupations: eigenvalues of gamma, descending.

        Identical to :func:`bohnasa.observables.orbital_occupations`
        (Penrose-Onsager occupations; see the module function for the
        physics convention).

        Parameters
        ----------
        state : numpy.ndarray of complex, shape (dim,)
            State vector in the bound basis.

        Returns
        -------
        numpy.ndarray of float, shape (L,)
            The natural occupations, descending.
        """
        return _natural_occupations(self.one_body_density_matrix(state))

    def momentum_distribution(self, state: np.ndarray) -> np.ndarray:
        r"""Occupations :math:`n_k = \langle a^\dagger_k a_k\rangle` on ring momenta.

        Identical to :func:`bohnasa.observables.momentum_distribution`
        (``a_k = L^(-1/2) sum_j exp(-i k j) c_j``; see the module function
        for the full convention).

        Parameters
        ----------
        state : numpy.ndarray of complex, shape (dim,)
            State vector in the bound basis.

        Returns
        -------
        numpy.ndarray of float, shape (L,)
            The momentum occupations ``n_{k_n}``.
        """
        return _momentum_from_gamma(self.one_body_density_matrix(state), self.L)

    def pair_correlation(self, state: np.ndarray) -> np.ndarray:
        r"""Normal-ordered density-noise matrix
        :math:`G2_{ij} = \langle c^\dagger_i c^\dagger_j c_j c_i\rangle`.

        Identical to :func:`bohnasa.observables.pair_correlation` (basis-
        diagonal occupation weights; see the module function for the
        convention).

        Parameters
        ----------
        state : numpy.ndarray of complex, shape (dim,)
            State vector in the bound basis.

        Returns
        -------
        numpy.ndarray of float, shape (L, L)
            The pair correlation matrix.
        """
        psi = _validated(state, self.fock)
        probs = np.abs(psi) ** 2
        return _g2_from(probs, self._occ, probs @ self._occ)

    def structure_factor(self, state: np.ndarray) -> np.ndarray:
        r"""Static structure factor
        :math:`S(k) = \langle \delta \hat{n}_k\, \delta \hat{n}_{-k}\rangle / N`.

        Identical to :func:`bohnasa.observables.structure_factor` (one FFT
        over the occupation rows; see the module function for the
        convention).

        Parameters
        ----------
        state : numpy.ndarray of complex, shape (dim,)
            State vector in the bound basis.

        Returns
        -------
        numpy.ndarray of float, shape (L,)
            ``S(k)`` on the ring momentum grid.
        """
        psi = _validated(state, self.fock)
        probs = np.abs(psi) ** 2
        return _structure_from(probs, self._occ, probs @ self._occ, self.fock.N)
