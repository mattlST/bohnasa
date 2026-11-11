"""Physics invariants: canonical commutation relations and operator identities."""

import numpy as np
import pytest

import bohnasa


def test_correlator_adjoint(fock53):
    i, j = 0, 1
    c = bohnasa.Correlator(fock53, i, j)
    cdag = c.adjoint()
    assert np.allclose(cdag.matrix.toarray(), c.matrix.toarray().T.conj())
    # and adjoint(c^dag_j c_i) == c^dag_i c_j == Correlator(j, i)
    assert np.allclose(cdag.matrix.toarray(), bohnasa.Correlator(fock53, j, i).matrix.toarray())


def test_correlator_diagonal_equals_number():
    N, L = 3, 3
    fock = bohnasa.fockStates_fixed_particle_number(N, L)
    for i in range(L):
        expected = bohnasa.number(fock, i).matrix.toarray()
        assert np.allclose(bohnasa.correlator(fock, i, i).matrix.toarray(), expected)


@pytest.mark.parametrize("i,j,k,ell", [(0, 1, 1, 2), (0, 1, 2, 0), (0, 2, 2, 0)])
def test_bosonic_commutator_of_correlators(fock53, i, j, k, ell):
    # [c_j^dag c_i, c_l^dag c_k] = delta_il c_k^dag c_j - delta_jk c_i^dag c_l
    def A(p, q):
        return bohnasa.correlator(fock53, p, q).matrix.toarray()

    lhs = A(i, j) @ A(k, ell) - A(k, ell) @ A(i, j)
    rhs = (int(i == ell)) * A(k, j) - (int(j == k)) * A(i, ell)
    assert np.allclose(lhs, rhs)
