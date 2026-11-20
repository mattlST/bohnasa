"""Exact finite-size physics of the observables module (issue #19).

Every anchor here is analytic on the nose - plane-wave condensates, Fock
eigenstates and ring momenta - so the tests double as a validation of the
Fourier and normal-ordering conventions themselves.

Each anchor runs against both evaluation paths: the one-shot module
functions and the reusable ``StateObservables`` pair-stack of issue #20,
pinning the physics through the pre-assembled gamma as well.
"""

import functools
import types

import numpy as np
import pytest

import bohnasa
from bohnasa import observables as obs

_SIX = (
    "one_body_density_matrix",
    "density_profile",
    "orbital_occupations",
    "momentum_distribution",
    "pair_correlation",
    "structure_factor",
)


def _bind(fock, api):
    """The six observables as ``fn(state)``, on the requested path."""
    if api == "class":
        return bohnasa.StateObservables(fock)
    return types.SimpleNamespace(
        **{name: functools.partial(getattr(obs, name), fock=fock) for name in _SIX}
    )


@pytest.fixture(params=["module", "class"])
def api(request):
    """Evaluation path: one-shot module functions or the bound reuse stack."""
    return request.param


def _ring_hamiltonian(fock):
    t = np.zeros((fock.L, fock.L))
    for site in range(fock.L):
        nxt = (site + 1) % fock.L
        t[nxt, site] = t[site, nxt] = -1.0
    return bohnasa.Hamiltonian(fock, one_body=t)


def _mode_orbital(fock, mode):
    """N = 1 state with all amplitude in ring momentum mode `mode`."""
    k = 2.0 * np.pi * mode / fock.L
    phi = np.exp(1j * k * np.arange(fock.L)) / np.sqrt(fock.L)
    state = np.zeros(fock.dim, dtype=complex)
    for site in range(fock.L):
        occ = tuple(1 if s == site else 0 for s in range(fock.L))
        state[fock.dict[occ]] = phi[site]
    return state


@pytest.fixture
def ring():
    fock = bohnasa.FixedNFockSpace(3, 6)
    ground = np.linalg.eigh(_ring_hamiltonian(fock).matrix.toarray())[1][:, 0]
    return fock, ground


def test_plane_wave_gamma_is_flat_and_condensate_is_perfect(ring, api):
    fock, gs = ring
    ev = _bind(fock, api)
    gamma = ev.one_body_density_matrix(gs)
    n, L = fock.N, fock.L
    # every entry N * conj(phi_i) phi_j = N/L, including off-diagonal coherence
    assert np.allclose(gamma, n / L, atol=1e-12)
    assert np.allclose(gamma, gamma.conj().T, atol=1e-12)
    assert np.trace(gamma).real == pytest.approx(n)

    nu = ev.orbital_occupations(gs)
    assert np.allclose(nu, np.r_[n, np.zeros(L - 1)], atol=1e-12)
    assert np.allclose(ev.density_profile(gs), n / L, atol=1e-12)


def test_plane_wave_momentum_and_structure_factor_anchors(ring, api):
    fock, gs = ring
    ev = _bind(fock, api)
    n, L = fock.N, fock.L
    n_k = ev.momentum_distribution(gs)
    assert np.allclose(n_k, np.r_[n, np.zeros(L - 1)], atol=1e-12)
    assert np.allclose(n_k.sum(), n)

    s_k = ev.structure_factor(gs)
    # fixed-N shell kills k = 0; the ideal condensate sits at exactly
    # shot-noise level 1 everywhere else (1/N convention)
    assert s_k[0] == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(s_k[1:], 1.0, atol=1e-12)


def test_plane_wave_pair_correlation_is_pure_bunching(ring, api):
    fock, gs = ring
    ev = _bind(fock, api)
    n, L = fock.N, fock.L
    g2 = ev.pair_correlation(gs)
    assert np.allclose(g2, n * (n - 1) / L**2, atol=1e-12)
    profile = ev.density_profile(gs)
    ratio = g2 / np.outer(profile, profile)
    # bosonic bunching factor, (N-1)/N < 1: enhanced probability to find pairs
    assert np.allclose(ratio, (n - 1) / n, atol=1e-10)


def test_fock_eigenstates_are_classical(ring, api):
    fock, _ = ring
    ev = _bind(fock, api)
    config = (1, 1, 1, 0, 0, 0)
    state = np.zeros(fock.dim, dtype=complex)
    state[fock.dict[config]] = 1.0

    nu = ev.orbital_occupations(state)
    assert np.allclose(nu, np.sort(np.array(config, dtype=float))[::-1], atol=1e-12)
    # a Fock configuration has zero inter-site coherence
    gamma = ev.one_body_density_matrix(state)
    assert np.abs(gamma - np.diag(np.diag(gamma))).max() < 1e-14
    # momentum is the flat distribution of distinguishable particles
    assert np.allclose(ev.momentum_distribution(state), fock.N / fock.L, atol=1e-12)
    # no doublons on site...
    g2 = ev.pair_correlation(state)
    assert np.abs(np.diag(g2)).max() == pytest.approx(0.0)
    # ...but exact classical pair correlations between occupied sites
    assert g2[0, 1] == pytest.approx(1.0)
    # a number eigenstate has zero density fluctuations
    assert np.abs(ev.structure_factor(state)).max() < 1e-15


def test_directed_momentum_eigenstates_pin_the_convention(api):
    """exp(+i k_m j) amplitude -> n_k = delta_{km} at the lattice-allowed mode;
    a mode NOT on the ring lattice leaks exactly the windowing weights."""
    fock = bohnasa.FixedNFockSpace(1, 6)
    ev = _bind(fock, api)
    for mode in range(fock.L):
        n_k = ev.momentum_distribution(_mode_orbital(fock, mode))
        expected = np.zeros(fock.L)
        expected[mode] = 1.0
        assert np.allclose(n_k, expected, atol=1e-12), mode


def test_momentum_distribution_reads_out_superposition_weights(api):
    fock = bohnasa.FixedNFockSpace(1, 6)
    ev = _bind(fock, api)
    weights = {1: 0.4, 2: 0.3, 4: 0.2, 5: 0.1}  # sum to 1: orthogonal orbitals
    state = np.zeros(fock.dim, dtype=complex)
    for mode, weight in weights.items():
        state += np.sqrt(weight) * _mode_orbital(fock, mode)
    n_k = ev.momentum_distribution(state)
    expected = np.zeros(fock.L)
    for mode, weight in weights.items():
        expected[mode] = weight
    assert np.allclose(n_k, expected, atol=1e-12)


def test_interactions_deplete_the_condensate_and_suppress_s_k(api):
    """The example storyline as a test: growing U monotonically lowers nu_0
    below the perfect-condensate anchor and S(k) below the shot-noise line."""
    fock = bohnasa.FixedNFockSpace(3, 6)
    h_ring = _ring_hamiltonian(fock)
    ev = _bind(fock, api)
    nu0, s1 = [], []
    for u in (0.0, 1.0, 3.0, 8.0):
        ground = np.linalg.eigh((h_ring + bohnasa.Hamiltonian(fock, onsite_u=u)).matrix.toarray())[
            1
        ][:, 0]
        nu0.append(ev.orbital_occupations(ground)[0])
        s1.append(ev.structure_factor(ground)[1])
    assert nu0[0] == pytest.approx(3.0, abs=1e-12)
    assert np.all(np.diff(nu0) < 0)
    assert s1[0] == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.diff(s1) < 0)
