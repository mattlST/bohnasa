"""Unit tests for the Liouvillian superoperator and operator growth."""

import numpy as np
import pytest
import scipy.sparse as sp

import bohnasa


def _random_matrix(dim, rng, hermitian=True):
    x = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    if hermitian:
        x = x + x.conj().T
    return x


def _operator_with(fock, matrix):
    op = bohnasa.Operator(fock)
    op.matrix = sp.csr_matrix(matrix)
    return op


@pytest.fixture
def small_system():
    fock = bohnasa.FixedNFockSpace(2, 3)  # dim 10
    rng = np.random.default_rng(7)
    hd = _random_matrix(fock.dim, rng)
    return fock, _operator_with(fock, hd), hd, rng


def test_action_matches_kron_representation(small_system):
    fock, ham, hd, rng = small_system
    liouvillian = bohnasa.Liouvillian(ham)
    xd = _random_matrix(fock.dim, rng, hermitian=False)
    d = fock.dim
    superop = np.kron(hd, np.eye(d)) - np.kron(np.eye(d), hd.T)
    assert np.allclose(liouvillian(xd.reshape(-1)), superop @ xd.reshape(-1), atol=1e-10)


def test_superoperator_is_hermitian_under_hilbert_schmidt(small_system):
    _, ham, _, rng = small_system
    liouvillian = bohnasa.Liouvillian(ham)
    for _ in range(3):
        a = rng.normal(size=liouvillian.dim**2) + 1j * rng.normal(size=liouvillian.dim**2)
        b = rng.normal(size=liouvillian.dim**2) + 1j * rng.normal(size=liouvillian.dim**2)
        assert np.vdot(a, liouvillian(b)) == pytest.approx(np.vdot(liouvillian(a), b), rel=1e-9)


def test_commutators_are_traceless(small_system):
    _, ham, _, rng = small_system
    liouvillian = bohnasa.Liouvillian(ham)
    x = rng.normal(size=liouvillian.dim**2) + 1j * rng.normal(size=liouvillian.dim**2)
    assert abs(np.trace(liouvillian.unvectorize(liouvillian(x)))) < 1e-10


def test_identity_spans_the_kernel(small_system):
    fock, ham, _, _ = small_system
    liouvillian = bohnasa.Liouvillian(ham)
    eye_op = _operator_with(fock, np.eye(fock.dim))
    assert np.abs(liouvillian(liouvillian.vectorize(eye_op))).max() < 1e-12


def test_vectorize_roundtrip(small_system):
    fock, ham, _, _ = small_system
    liouvillian = bohnasa.Liouvillian(ham)
    number = bohnasa.Number(fock, 0)
    back = liouvillian.unvectorize(liouvillian.vectorize(number))
    assert np.allclose(back, number.matrix.toarray())


@pytest.mark.parametrize("size", [3, 9, 101])
def test_wrong_vector_size_raises(small_system, size):
    _, ham, _, _ = small_system
    with pytest.raises(ValueError, match="expected length"):
        bohnasa.Liouvillian(ham)(np.zeros(size))


def test_missing_matrix_raises():
    fock = bohnasa.FixedNFockSpace(2, 3)
    bare = bohnasa.Operator(fock)
    with pytest.raises(ValueError, match="no matrix"):
        bohnasa.Liouvillian(bare)
    with pytest.raises(ValueError, match="no matrix"):
        bohnasa.Liouvillian(bohnasa.Hopping(fock)).vectorize(bare)


def test_seed_shape_mismatch_raises():
    fock = bohnasa.FixedNFockSpace(2, 3)
    other = bohnasa.FixedNFockSpace(1, 3)
    liouvillian = bohnasa.Liouvillian(bohnasa.Hopping(fock))
    with pytest.raises(ValueError, match="does not match"):
        bohnasa.operator_growth(liouvillian, bohnasa.Number(other, 0), np.linspace(0, 1, 5), 10)


def test_identity_seed_is_rejected():
    fock = bohnasa.FixedNFockSpace(2, 3)
    liouvillian = bohnasa.Liouvillian(bohnasa.Hopping(fock))
    eye_op = _operator_with(fock, np.eye(fock.dim))
    with pytest.raises(ValueError, match="proportional to the identity"):
        bohnasa.operator_growth(liouvillian, eye_op, np.linspace(0, 1, 5), 10)


def test_hamiltonian_itself_stops_the_chain_at_one_level():
    fock = bohnasa.FixedNFockSpace(2, 3)
    ham = bohnasa.Hopping(fock) * (-1.0) + bohnasa.Interaction(fock) * 0.5
    liouvillian = bohnasa.Liouvillian(ham)
    times = np.linspace(0, 2, 11)
    krylov, t_m = bohnasa.operator_growth(liouvillian, ham, times, 30)
    assert t_m.shape == (1, 1)
    assert np.allclose(krylov, 0)
