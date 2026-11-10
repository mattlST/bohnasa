import numpy as np
import pytest
from math import factorial, sqrt

import bohnasa


def test_single_fockstate_has_one_at_correct_index():
    N, L = 5, 3
    fock = bohnasa.fockStates_fixed_particle_number(N, L)
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
    fock = bohnasa.fockStates_fixed_particle_number(N, L)
    phi = rng.normal(size=L) + 1j * rng.normal(size=L)

    psi = bohnasa.coherent_state_number_projected(fock, phi.copy())
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
    fock = bohnasa.fockStates_fixed_particle_number(N, L)
    phi = rng.normal(size=L) + 1j * rng.normal(size=L)
    phi /= np.linalg.norm(phi)
    psi = bohnasa.coherent_state_number_projected(fock, phi)
    assert np.isclose(np.linalg.norm(psi), 1.0)


def test_coherent_state_global_phase_covariance():
    # phi -> phi * e^{i th}  =>  psi -> psi * e^{i th N}
    rng = np.random.default_rng(11)
    N, L = 4, 3
    fock = bohnasa.fockStates_fixed_particle_number(N, L)
    phi = rng.normal(size=L) + 1j * rng.normal(size=L)
    theta = 0.7
    psi1 = bohnasa.coherent_state_number_projected(fock, phi.copy())
    psi2 = bohnasa.coherent_state_number_projected(fock, phi * np.exp(1j * theta))
    assert np.allclose(psi2, psi1 * np.exp(1j * theta * N))
