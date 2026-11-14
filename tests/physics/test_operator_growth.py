"""Physics invariants of operator-growth Krylov complexity (issue #15).

The ground-truth check mirrors example 05's philosophy: compare the Lanczos
chain against brute-force Heisenberg evolution with the *dense* superoperator
- and anchor what can be solved analytically (single-particle L̂ spectra, the
universal short-time beta_1^2 t^2 law, integrable closure).
"""

import numpy as np
import pytest
from scipy.linalg import expm

import bohnasa


def _traceless_flat(operator):
    x = operator.matrix.toarray().astype(complex)
    d = x.shape[0]
    x -= np.trace(x) / d * np.eye(d)
    return (x / np.linalg.norm(x)).reshape(-1)


def test_single_particle_liouvillian_spectrum_is_level_differences():
    fock = bohnasa.FixedNFockSpace(1, 6)
    ham = bohnasa.Hopping(fock, connections="pbc") * (-1.0)
    hd = ham.matrix.toarray()
    d = fock.dim
    dense = np.kron(hd, np.eye(d)) - np.kron(np.eye(d), hd.T)
    eps = np.linalg.eigvalsh(hd)
    expected = np.sort((eps[:, None] - eps[None, :]).ravel())
    assert np.allclose(np.sort(np.linalg.eigvalsh(dense)), expected, atol=1e-12)


def test_operator_growth_curve_matches_exact_heisenberg_evolution():
    fock = bohnasa.FixedNFockSpace(2, 4)
    ham = bohnasa.Hopping(fock) * (-1.0) + bohnasa.Interaction(fock) * 0.5
    liouvillian = bohnasa.Liouvillian(ham)
    d = liouvillian.dim
    seed = _traceless_flat(bohnasa.Number(fock, 0))
    times = np.linspace(0.0, 3.0, 40)

    curve, t_m = bohnasa.operator_growth(liouvillian, bohnasa.Number(fock, 0), times, d * d - 1)

    # brute force: dense superoperator evolution + K-hat from the same chain
    hd = ham.matrix.toarray()
    dense = np.kron(hd, np.eye(d)) - np.kron(np.eye(d), hd.T)
    v, _ = bohnasa.lanczos(seed, liouvillian, d * d - 1)
    k_op = v.conj().T @ np.diag(np.arange(v.shape[0])) @ v
    exact = np.array(
        [np.vdot(o, k_op @ o).real for o in map(lambda t: expm(1j * t * dense) @ seed, times)]
    )
    assert t_m.shape[0] == v.shape[0]
    assert np.abs(curve - exact).max() < 1e-10


def test_universal_quadratic_short_time_growth():
    fock = bohnasa.FixedNFockSpace(2, 4)
    ham = bohnasa.Hopping(fock) * (-1.0) + bohnasa.Interaction(fock) * 0.5
    liouvillian = bohnasa.Liouvillian(ham)
    times = np.array([1e-4, 2e-4, 3e-4])
    curve, t_m = bohnasa.operator_growth(liouvillian, bohnasa.Number(fock, 0), times, 60)
    beta1 = abs(t_m[1, 0])
    assert np.allclose(curve, (beta1 * times) ** 2, rtol=1e-5)


def test_free_density_closes_early_interacting_growth_sustains():
    """Integrable closure vs interacting climb - the ex03-sibling story."""
    fock = bohnasa.FixedNFockSpace(6, 4)
    times = np.linspace(0.0, 4.0, 150)
    h_free = bohnasa.Hopping(fock, connections="pbc") * (-1.0)
    h_int = h_free + bohnasa.Interaction(fock) * 4.0
    seed = bohnasa.Number(fock, 0)

    k_free, t_free = bohnasa.operator_growth(bohnasa.Liouvillian(h_free), seed, times, 60)
    k_int, t_int = bohnasa.operator_growth(bohnasa.Liouvillian(h_int), seed, times, 60)

    # the on-site interaction commutes with the density seed: first step is
    # coupling-independent
    assert abs(t_free[1, 0]) == pytest.approx(abs(t_int[1, 0]), rel=1e-9)
    # free theory: finite Krylov closure, happy breakdown far below the cap
    assert t_free.shape[0] < 60
    assert k_free.max() < 5.0
    # interacting: chain fills the requested depth and reaches much higher
    # Krylov levels (pointwise curves oscillate; maxima are the stable claim)
    assert t_int.shape[0] == 61  # numVec + 1 rows: no happy breakdown yet
    assert k_int.max() > 3 * k_free.max()


def test_complex_seed_pins_the_heisenberg_time_direction():
    """<K>(t) is even in t for every seed (real-symmetric T_m), so the curve
    cannot see the sign of exp(+iLt). The chain-RECONSTRUCTED operator O(t)
    can: compare it against brute-force dense evolution of a genuinely
    complex (non-Hermitian, non-phase-rotatable) seed, and check the wrong
    time direction demonstrably disagrees."""
    fock = bohnasa.FixedNFockSpace(2, 4)
    ham = bohnasa.Hopping(fock) * (-1.0) + bohnasa.Interaction(fock) * 0.5
    liouvillian = bohnasa.Liouvillian(ham)
    d = liouvillian.dim

    # c^dag_0 c_1 + 1j c^dag_2 c_3: off-diagonal monomials are traceless
    seed = bohnasa.Correlator(fock, 1, 0).matrix + 1j * bohnasa.Correlator(fock, 3, 2).matrix
    seed_flat = np.asarray(seed.toarray(), dtype=complex).reshape(-1)
    seed_flat /= np.linalg.norm(seed_flat)

    v, t_m = bohnasa.lanczos(seed_flat, liouvillian, d * d - 1)
    eigval, eigvec = np.linalg.eigh(t_m)
    coeffs0 = eigvec[0, :].conj()

    hd = ham.matrix.toarray()
    dense_superop = np.kron(hd, np.eye(d)) - np.kron(np.eye(d), hd.T)
    t_probe = 0.7
    forward = expm(1j * t_probe * dense_superop) @ seed_flat
    backward = expm(-1j * t_probe * dense_superop) @ seed_flat

    chain_coeffs = eigvec @ (np.exp(1j * eigval * t_probe) * coeffs0)
    reconstructed = chain_coeffs @ v
    assert np.abs(reconstructed - forward).max() < 1e-10
    # the seed is not real up to a global phase, so the sign is observable:
    assert np.abs(reconstructed - backward).max() > 1e-3
