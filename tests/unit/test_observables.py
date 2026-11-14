"""Unit tests for state-resolved observables (issue #19)."""

import numpy as np
import pytest

import bohnasa
from bohnasa import observables as obs


@pytest.fixture
def system():
    fock = bohnasa.FixedNFockSpace(3, 5)
    rng = np.random.default_rng(7)
    state = rng.normal(size=fock.dim) + 1j * rng.normal(size=fock.dim)
    state /= np.linalg.norm(state)
    return fock, state


def test_state_shape_is_validated():
    fock = bohnasa.FixedNFockSpace(2, 3)
    for fn in (
        obs.one_body_density_matrix,
        obs.density_profile,
        obs.momentum_distribution,
        obs.pair_correlation,
        obs.structure_factor,
    ):
        with pytest.raises(ValueError, match=r"state must have shape"):
            fn(np.zeros(fock.dim + 1, dtype=complex), fock)


def test_gamma_entries_are_correlator_means(system):
    fock, state = system
    gamma = obs.one_body_density_matrix(state, fock)
    for i in range(fock.L):
        for j in range(fock.L):
            assert gamma[i, j] == pytest.approx(
                bohnasa.Correlator(fock, j, i).eval_mean(state), abs=1e-12
            )


def test_profile_agrees_with_number_ops_and_gamma_diag(system):
    fock, state = system
    profile = obs.density_profile(state, fock)
    gamma_diag = np.diag(obs.one_body_density_matrix(state, fock)).real
    assert profile.shape == (fock.L,)
    assert np.allclose(profile, gamma_diag, atol=1e-12)
    for i in range(fock.L):
        assert profile[i] == pytest.approx(bohnasa.Number(fock, i).eval_mean(state).real)


def test_occupations_sum_to_n_and_match_gamma_spectrum(system):
    fock, state = system
    nu = obs.orbital_occupations(state, fock)
    assert nu.shape == (fock.L,)
    assert nu.sum() == pytest.approx(fock.N)
    assert np.all(np.diff(nu) <= 1e-12)  # descending
    gamma = obs.one_body_density_matrix(state, fock)
    assert np.allclose(nu, np.sort(np.linalg.eigvalsh(gamma).real)[::-1], atol=1e-12)


def test_momentum_matches_independent_fourier_of_gamma(system):
    fock, state = system
    gamma = obs.one_body_density_matrix(state, fock)
    k = 2.0 * np.pi * np.arange(fock.L) / fock.L
    e = np.exp(-1j * k[:, None] * np.arange(fock.L)[None, :]) / np.sqrt(fock.L)
    # a_k = sum_j e_kj c_j  =>  n_k = sum_ij conj(e_ki) gamma_ij e_kj
    manual = np.einsum("ki,ij,kj->k", e.conj(), gamma, e).real
    assert np.allclose(obs.momentum_distribution(state, fock), manual, atol=1e-12)


def test_pair_correlation_matches_number_operator_products(system):
    fock, state = system
    g2 = obs.pair_correlation(state, fock)
    n_ops = [bohnasa.Number(fock, i).matrix.toarray() for i in range(fock.L)]
    eye = np.eye(fock.dim)
    for i in range(fock.L):
        for j in range(fock.L):
            op = n_ops[i] @ (n_ops[j] - (eye if i == j else 0.0))
            assert g2[i, j] == pytest.approx(np.vdot(state, op @ state).real, abs=1e-11)


def test_structure_factor_matches_density_fluctuation_operators(system):
    fock, state = system
    profile = obs.density_profile(state, fock)
    n_ops = [bohnasa.Number(fock, i).matrix.toarray() for i in range(fock.L)]
    k = 2.0 * np.pi * np.arange(fock.L) / fock.L
    s_k = obs.structure_factor(state, fock)
    for mode, kk in enumerate(k):
        delta = sum(
            np.exp(-1j * kk * j) * (n_ops[j] - profile[j] * np.eye(fock.dim)) for j in range(fock.L)
        )
        reference = np.vdot(state, delta @ delta.conj().T @ state).real / fock.N
        assert s_k[mode] == pytest.approx(reference, abs=1e-10)


def test_structure_factor_symmetry_and_sum_rule(system):
    fock, state = system
    s_k = obs.structure_factor(state, fock)
    # S(k) = S(-k): index symmetry n <-> L-n on the ring momentum grid
    assert np.allclose(s_k, s_k[(fock.L - np.arange(fock.L)) % fock.L])
    probs = np.abs(state) ** 2
    occ = fock.fockArray.astype(float)
    profile = obs.density_profile(state, fock)
    var_i = (probs[:, None] * occ**2).sum(0) - profile**2
    assert s_k.sum() == pytest.approx(fock.L * var_i.sum() / fock.N, abs=1e-11)
