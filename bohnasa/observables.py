"""State-resolved observables: coherence, occupations, and density noise.

One-body quantities are assembled from the existing second-quantized
operators (:class:`~bohnasa.operators.Correlator`); the normal-ordered
two-body densities are basis-diagonal - their expectation values are
occupation-weighted sums over the Fock basis, computed exactly in
``O(dim * L)`` without any sparse operator products.
"""

import numpy as np

from .operators import Correlator

__all__ = [
    "one_body_density_matrix",
    "density_profile",
    "orbital_occupations",
    "momentum_distribution",
    "pair_correlation",
    "structure_factor",
]


def _validated(state, fock):
    state = np.asarray(state, dtype=complex)
    if state.shape != (fock.dim,):
        raise ValueError(
            f"observables: state must have shape ({fock.dim},) for this basis, got {state.shape}"
        )
    return state


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
    gamma = one_body_density_matrix(state, fock)
    nu = np.linalg.eigvalsh(gamma)
    return nu.real[::-1]


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
    gamma = one_body_density_matrix(state, fock)
    k = 2.0 * np.pi * np.arange(fock.L) / fock.L
    # n_k = sum_ij conj(E_ki) gamma_ij E_kj, E_kj = exp(-i k j)/sqrt(L)
    phase = np.exp(-1j * np.outer(k, np.arange(fock.L))) / np.sqrt(fock.L)
    return np.real(np.diag(phase.conj() @ gamma @ phase.T))


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
    g2 = (probs[:, None] * occ).T @ occ
    np.fill_diagonal(g2, np.diag(g2) - density_profile(state, fock))
    return g2


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
    fluct = occ - density_profile(state, fock)[None, :]
    modes = np.fft.fft(fluct, axis=1)
    return probs @ (np.abs(modes) ** 2) / fock.N
