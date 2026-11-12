"""Unit tests: correctness of individual operators and the operator API."""

import numpy as np
import pytest

import bohnasa


def test_number_diagonal_is_occupation(fock53):
    for i in range(3):
        n_i = bohnasa.Number(fock53, i)
        assert np.allclose(n_i.matrix.diagonal(), fock53.fockArray[:, i])
        assert np.allclose(n_i.matrix.toarray(), n_i.matrix.toarray().T.conj())


def test_interaction_diagonal(fock53):
    u = bohnasa.Interaction(fock53)
    expected = np.sum(fock53.fockArray * (fock53.fockArray - 1), axis=1)
    assert np.allclose(u.matrix.diagonal(), expected)


def test_chemical_mu_diagonal(fock53):
    rng = np.random.default_rng(3)
    mu = rng.normal(size=3)
    op = bohnasa.ChemicalPotential(fock53, mu)
    assert np.allclose(op.matrix.diagonal(), fock53.fockArray @ mu)


def test_chemical_potential_accepts_list_and_validates_shape(fock53):
    op = bohnasa.ChemicalPotential(fock53, [1.0, 2.0, 3.0])
    assert np.allclose(op.matrix.diagonal(), fock53.fockArray @ np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        bohnasa.ChemicalPotential(fock53, [1.0, 2.0])


def test_correlator_is_annihilate_create(fock53):
    # correlator(i, j) = c_j^dag c_i : moves one particle i -> j
    i, j = 0, 2
    op = bohnasa.Correlator(fock53, i, j)
    M = op.matrix.toarray()
    for index, state in enumerate(fock53.fockSpace):
        if state[i] == 0:
            continue
        occ = list(state)
        occ[i] -= 1
        occ[j] += 1
        target = fock53.dict[tuple(occ)]
        expected = np.sqrt(state[i] * (state[j] + 1))
        assert np.isclose(M[target, index], expected)
        assert np.count_nonzero(M[:, index]) == 1


def test_eval_mean_matches_vdot(fock53):
    rng = np.random.default_rng(5)
    psi = rng.normal(size=fock53.dim) + 1j * rng.normal(size=fock53.dim)
    psi /= np.linalg.norm(psi)
    op = bohnasa.Number(fock53, 1)
    assert np.isclose(op.eval_mean(psi), np.vdot(psi, op.matrix @ psi))


def test_eval_mean_var_consistency(fock53):
    rng = np.random.default_rng(6)
    psi = rng.normal(size=fock53.dim) + 1j * rng.normal(size=fock53.dim)
    psi /= np.linalg.norm(psi)
    op = bohnasa.Hopping(fock53)
    mean, var = op.eval_mean_var(psi)
    a2 = (op.matrix @ op.matrix) @ psi
    assert np.isclose(mean, np.vdot(psi, op.matrix @ psi))
    assert np.isclose(var, np.vdot(psi, a2) - mean**2)


def test_operator_addition(fock53):
    a = bohnasa.Number(fock53, 0)
    b = bohnasa.Number(fock53, 1)
    c = a + b
    assert np.allclose(
        c.matrix.toarray(),
        a.matrix.toarray() + b.matrix.toarray(),
    )


def test_operator_subtraction(fock53):
    a = bohnasa.Number(fock53, 0)
    b = bohnasa.Number(fock53, 1)
    c = a - b
    assert np.allclose(
        c.matrix.toarray(),
        a.matrix.toarray() - b.matrix.toarray(),
    )


def test_scalar_multiplication(fock53):
    a = bohnasa.Number(fock53, 0)
    c = a * 2.5
    assert np.allclose(c.matrix.toarray(), 2.5 * a.matrix.toarray())
    # original untouched (deepcopy semantics)
    assert np.allclose(a.matrix.toarray(), a.matrix.toarray())
