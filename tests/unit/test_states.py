from math import factorial, sqrt

import numpy as np
import pytest

import bohnasa


def test_single_fockstate_has_one_at_correct_index():
    N, L = 5, 3
    fock = bohnasa.FixedNFockSpace(N, L)
    vec = np.array([2, 0, 3])
    psi = bohnasa.single_fockstate(fock, vec)
    assert psi.shape == (fock.dim,)
    assert np.isclose(np.linalg.norm(psi), 1.0)
    index = fock.dict[tuple(vec)]
    assert psi[index] == 1.0
    assert np.count_nonzero(psi) == 1


def test_multinomial_known_values():
    assert bohnasa.multinomial([2, 1]) == 3
    assert bohnasa.multinomial([1, 1, 1]) == 6
    assert bohnasa.multinomial([0]) == 1
    assert np.isclose(bohnasa.multinomial_sqrt([2, 1]), sqrt(3))


def test_coherent_state_matches_analytic_formula():
    rng = np.random.default_rng(42)
    N, L = 4, 3
    fock = bohnasa.FixedNFockSpace(N, L)
    phi = rng.normal(size=L) + 1j * rng.normal(size=L)

    psi = bohnasa.fixed_n_coherent_state(fock, phi.copy())
    phi_n = phi / np.linalg.norm(phi)

    # <n|psi> = sqrt(N! / prod n_i!) * prod_i phi_i^{n_i}
    expected = np.array(
        [
            sqrt(factorial(N) / np.prod([factorial(n) for n in state]))
            * np.prod(phi_n ** np.array(state))
            for state in fock.fockSpace
        ]
    )
    assert np.allclose(psi, expected)


def test_coherent_state_is_normalized_for_unit_phi():
    # sum over the fixed-N sector of a normalized coherent amplitude has
    # total weight 1 (multinomial distribution)
    rng = np.random.default_rng(7)
    N, L = 6, 3
    fock = bohnasa.FixedNFockSpace(N, L)
    phi = rng.normal(size=L) + 1j * rng.normal(size=L)
    phi /= np.linalg.norm(phi)
    psi = bohnasa.fixed_n_coherent_state(fock, phi)
    assert np.isclose(np.linalg.norm(psi), 1.0)


def test_coherent_state_global_phase_covariance():
    # phi -> phi * e^{i th}  =>  psi -> psi * e^{i th N}
    rng = np.random.default_rng(11)
    N, L = 4, 3
    fock = bohnasa.FixedNFockSpace(N, L)
    phi = rng.normal(size=L) + 1j * rng.normal(size=L)
    theta = 0.7
    psi1 = bohnasa.fixed_n_coherent_state(fock, phi.copy())
    psi2 = bohnasa.fixed_n_coherent_state(fock, phi * np.exp(1j * theta))
    assert np.allclose(psi2, psi1 * np.exp(1j * theta * N))


def test_coherent_state_large_N_is_finite_and_normalized():
    """Log-space construction: the multinomial prefactor never
    materializes, so N = 4096 (sqrt(N!) alone exceeds float64) builds
    exact amplitudes instead of inf * 0 = NaN."""
    N = 4096
    fock = bohnasa.FixedNFockSpace(N, 2)
    psi = bohnasa.fixed_n_coherent_state(fock, np.array([1.0, -1.0]) / np.sqrt(2.0))
    assert np.all(np.isfinite(psi))
    assert np.linalg.norm(psi) == pytest.approx(1.0, abs=1e-12)


def test_coherent_state_empty_mode_is_exactly_zero():
    fock = bohnasa.FixedNFockSpace(20, 3)
    psi = bohnasa.fixed_n_coherent_state(fock, np.array([1.0, 0.0, 1.0]))
    occ = fock.fockArray
    assert psi[occ[:, 1] > 0].size > 0
    assert np.all(psi[occ[:, 1] > 0] == 0.0)
    assert np.linalg.norm(psi) == pytest.approx(1.0)
