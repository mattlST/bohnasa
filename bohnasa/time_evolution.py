"""Exact time evolution via diagonalization."""

import numpy as np


def time_evolution_exact_ind(H, state, time, observables):
    """Exact time evolution via full diagonalization of ``H``.

    Diagonalizes ``H`` once and caches the result on the operator itself
    (``H.eigval``, ``H.eigvec`` attributes), then propagates ``state``
    step-by-step (``e^{-i H dt}`` per sample, so the spectrum may be
    non-equidistant only through ``time[1] - time[0]``) and records
    mean and variance of each observable at every time.

    Only feasible for small Hilbert spaces (dense diagonalization).

    Parameters
    ----------
    H : Operator
        Hamiltonian; assumed time-independent.
    state : numpy.ndarray
        Initial state vector.
    time : numpy.ndarray
        Equidistant time samples.
    observables : list of Operator
        Observables to measure; may be non-Hermitian.

    Returns
    -------
    numpy.ndarray of complex, shape (len(observables), 2, len(time))
        ``result[i, 0, t]`` is the expectation value, ``result[i, 1, t]``
        the variance of observable ``i`` at time index ``t``. Complex when
        the observable is non-Hermitian; imaginary parts are never dropped.
    """
    if not hasattr(H, "eigval"):
        eigval, eigvec = np.linalg.eigh(H.matrix.toarray())
        H.eigval = np.asarray(eigval)
        H.eigvec = np.asarray(eigvec)
    state = np.asarray(state, dtype=complex)
    state_eval = H.eigvec.T.conj() @ state

    dt = time[1] - time[0]
    dE = np.exp(-1j * dt * H.eigval)
    results_observables = np.zeros((len(observables), 2, len(time)), dtype=complex)
    for iob, ob in enumerate(observables):
        results_observables[iob, :, 0] = ob.eval_mean_var(state)
    for it in range(len(time) - 1):
        state_eval *= dE
        ret = H.eigvec @ state_eval
        for iob, ob in enumerate(observables):
            results_observables[iob, :, it + 1] = ob.eval_mean_var(ret)
    return results_observables
