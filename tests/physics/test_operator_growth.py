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
