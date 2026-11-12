"""State vectors in the fixed-N Fock basis."""

import numpy as np


def multinomial(lst):
    """Multinomial coefficient ``N! / prod_i n_i!`` for occupations ``lst``.

    Examples
    --------
    >>> multinomial([2, 1])
    3
    """
    res, i = 1, int(np.sum(lst))
    lst_sorted = np.sort(lst)[::-1]
    for a in lst_sorted:
        for j in range(1, a + 1):
            res *= i
            res //= j
            i -= 1
    return res


def multinomial_sqrt(lst):
    """Square root of :func:`multinomial`, accumulated multiplicatively."""
    res, i = 1, np.sum(lst)
    lst_sorted = np.sort(lst)[::-1]
    for a in lst_sorted:
        for j in range(1, a + 1):
            res *= np.sqrt(i)
            res /= np.sqrt(j)
            i -= 1
    return res


def single_fockstate(fock, vec):
    """Basis state of ``fock`` with occupations ``vec`` as a complex vector.

    Parameters
    ----------
    fock : FixedNFockSpace
        Basis to embed the state in.
    vec : array-like of shape (fock.L,)
        Occupation numbers; must sum to ``fock.N``.

    Returns
    -------
    numpy.ndarray of complex, shape (fock.dim,)
        Unit vector with a single 1 at ``fock.dict[tuple(vec)]``.

    Raises
    ------
    ValueError
        If ``vec`` has the wrong shape or total particle number.
    """
    N = fock.N
    L = fock.L
    vec = np.asarray(vec)
    if vec.shape != (L,) or np.sum(vec) != N:
        raise ValueError(
            f"state vector must have shape ({L},) and particle number {N}, "
            f"got shape {vec.shape} and sum {int(np.sum(vec))}"
        )

    index = fock.dict[tuple(vec)]

    sta_vec = np.zeros(fock.dim, dtype=complex)
    sta_vec[index] = 1.0
    return sta_vec


def fixed_n_coherent_state(fock, phi):
    """Projection of a coherent state onto the fixed-N sector of ``fock``.

    ``phi`` holds the mode amplitudes (length L, or length 2L interpreted as
    real and imaginary parts) and is normalized internally; the caller's
    array is never modified.

    Parameters
    ----------
    fock : FixedNFockSpace
        Target basis with total particle number ``N``.
    phi : array-like of shape (L,) or (2L,)
        Coherent-state mode amplitudes.

    Returns
    -------
    numpy.ndarray of complex, shape (fock.dim,)
        Amplitudes ``sqrt(N! / prod n!) * prod_i phi_i^n_i``; unit norm for
        normalized ``phi``.

    Raises
    ------
    ValueError
        If ``phi`` has neither length ``L`` nor ``2L``.
    """
    L = fock.L
    phi = np.asarray(phi, dtype=complex)
    if len(phi) == 2 * L:
        phi = phi[:L] + 1j * phi[L:]
    elif len(phi) != L:
        raise ValueError(f"phi must have length {L} or {2 * L}, got {len(phi)}")
    phi = phi / np.linalg.norm(phi)

    preCalMulti = np.array([(multinomial_sqrt(x)) for x in fock.fockArray])
    return np.prod(phi[None, :] ** fock.fockArray, axis=1) * preCalMulti
