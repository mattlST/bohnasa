import numpy as np
import pytest

import bohnasa


def test_happy_breakdown_on_eigenstate():
    N, L = 5, 3
    fock = bohnasa.fockStates_fixed_particle_number(N, L)
    sta = bohnasa.single_fockstate(fock, np.array([3, 1, 1]))
    op = bohnasa.number(fock, 0)
    v, T_m = bohnasa.lanczos(sta, op, 10)
    assert v.shape == (1, fock.dim)
    assert T_m.shape == (1, 1)
    assert np.isclose(T_m[0, 0], 3.0)


def test_lanczos_vectors_stay_orthonormal():
    N, L = 8, 4
    fock = bohnasa.fockStates_fixed_particle_number(N, L)
    op = bohnasa.hopping(fock) + bohnasa.interaction(fock) * 0.3
    rng = np.random.default_rng(1)
    sta = rng.normal(size=fock.dim) + 1j * rng.normal(size=fock.dim)
    sta /= np.linalg.norm(sta)

    v, T_m = bohnasa.lanczos(sta, op, 40)
    norms = np.linalg.norm(v, axis=1)
    m = int(np.count_nonzero(norms > 1e-12))
    assert m >= 20, "expected a non-trivial Krylov basis"

    gram = v[:m] @ v[:m].conj().T
    assert np.linalg.norm(gram - np.eye(m)) < 1e-8
    assert np.allclose(norms[:m], 1.0, atol=1e-8)


def test_lanczos_T_matrix_is_symmetric_tridiagonal():
    N, L = 8, 4
    fock = bohnasa.fockStates_fixed_particle_number(N, L)
    op = bohnasa.hopping(fock) + bohnasa.interaction(fock) * 0.3
    rng = np.random.default_rng(2)
    sta = rng.normal(size=fock.dim) + 1j * rng.normal(size=fock.dim)
    sta /= np.linalg.norm(sta)

    v, T_m = bohnasa.lanczos(sta, op, 40)
    m = int(np.count_nonzero(np.linalg.norm(v, axis=1) > 1e-12))
    T = T_m[:m, :m]

    assert np.allclose(T, T.T.conj())
    off = T.copy()
    idx = np.arange(m)
    off[idx, idx] = 0.0
    off[idx[:-1], idx[1:]] = 0.0
    off[idx[1:], idx[:-1]] = 0.0
    assert np.allclose(off, 0.0)


def test_lanczos_ground_state_energy_matches_exact():
    N, L = 8, 4
    fock = bohnasa.fockStates_fixed_particle_number(N, L)
    op = bohnasa.hopping(fock) + bohnasa.interaction(fock) * 0.3
    rng = np.random.default_rng(4)
    sta = rng.normal(size=fock.dim) + 1j * rng.normal(size=fock.dim)
    sta /= np.linalg.norm(sta)

    v, T_m = bohnasa.lanczos(sta, op, 40)
    m = int(np.count_nonzero(np.linalg.norm(v, axis=1) > 1e-12))
    e0_krylov = np.linalg.eigvalsh(T_m[:m, :m])[0]
    e0_exact = np.linalg.eigvalsh(op.matrix.toarray())[0]
    assert np.isclose(e0_krylov, e0_exact, atol=1e-6)
