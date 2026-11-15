"""Unit tests for the squared-commutator OTOC machinery (issue #22)."""

import numpy as np
import pytest
import scipy.linalg as sla

import bohnasa
from bohnasa import classical_mean_field as mf
from bohnasa import otoc as oc


@pytest.fixture
def dimer():
    N = 6
    fock = bohnasa.FixedNFockSpace(N, 2)
    t = np.array([[0.0, -2.0], [-2.0, 0.0]])
    H = bohnasa.Hamiltonian(fock, one_body=t, onsite_u=0.1)
    op = (bohnasa.Number(fock, 0) - bohnasa.Number(fock, 1)) * 0.5
    state = np.zeros(fock.dim, dtype=complex)
    state[fock.dict[(1, 5)]] = 1.0
    return fock, H, op, state


def _beta2(H, op, psi):
    hd, od = H.matrix.toarray(), op.matrix.toarray()
    ho = hd @ od - od @ hd
    v = ho @ od @ psi - od @ ho @ psi
    return np.vdot(v, v).real


def test_curve_matches_brute_force_matrix_commutator(dimer):
    fock, H, op, state = dimer
    times = np.linspace(0.0, 1.7, 7)
    sc = oc.SquaredCommutator(H, op)
    curve = sc.curve(state, times)
    hd, od = H.matrix.toarray(), op.matrix.toarray()
    brute = np.array(
        [
            np.linalg.norm(
                (sla.expm(1j * t * hd) @ od @ sla.expm(-1j * t * hd)) @ od @ state
                - od @ (sla.expm(1j * t * hd) @ od @ sla.expm(-1j * t * hd)) @ state
            )
            ** 2
            for t in times
        ]
    )
    assert np.allclose(curve, brute, rtol=1e-9, atol=1e-12)


def test_eig_and_krylov_paths_agree(dimer):
    _, H, op, state = dimer
    times = np.linspace(0.0, 1.7, 7)
    a = oc.SquaredCommutator(H, op, method="eig").curve(state, times)
    b = oc.SquaredCommutator(H, op, method="krylov").curve(state, times)
    assert np.allclose(a, b, rtol=1e-8, atol=1e-12)


def test_zero_time_is_exactly_zero(dimer):
    _, H, op, state = dimer
    sc = oc.SquaredCommutator(H, op)
    assert sc.curve(state, np.array([0.0]))[0] == 0.0


def test_operator_commuting_with_h_does_not_grow(dimer):
    _, H, _, state = dimer
    sc = oc.SquaredCommutator(H, H)
    assert sc.curve(state, np.linspace(0, 3, 13)).max() < 1e-20


def test_short_time_quadratic_coefficient_is_the_double_commutator(dimer):
    _, H, op, state = dimer
    sc = oc.SquaredCommutator(H, op)
    t = 1e-3
    beta2 = _beta2(H, op, state)
    assert sc.curve(state, np.array([t]))[0] / t**2 == pytest.approx(beta2, rel=1e-4)


def test_normalize_option(dimer):
    _, H, op, state = dimer
    sc = oc.SquaredCommutator(H, op)
    times = np.linspace(0.0, 1.0, 5)
    weight = np.vdot(op.matrix @ state, op.matrix @ state).real
    assert np.allclose(sc.curve(state, times, normalize=True), sc.curve(state, times) / weight)


def test_floquet_equals_stepwise_hamiltonian_evolution(dimer):
    _, H, op, state = dimer
    dt = 0.31
    u = sla.expm(-1j * dt * H.matrix.toarray())
    sc_f = oc.SquaredCommutator.from_propagator(u, op)
    sc_h = oc.SquaredCommutator(H, op)
    kicks = np.arange(0, 5)
    assert np.allclose(
        sc_f.curve(state, kicks), sc_h.curve(state, kicks * dt), rtol=1e-9, atol=1e-14
    )
    with pytest.raises(ValueError, match="integers"):
        sc_f.curve(state, np.array([0.0, 1.5]))


def test_input_validation(dimer):
    fock, H, op, state = dimer
    with pytest.raises(ValueError, match="no matrix"):
        oc.SquaredCommutator(bohnasa.Operator(fock), op)
    with pytest.raises(ValueError, match="unknown method"):
        oc.SquaredCommutator(H, op, method="magic")
    sc = oc.SquaredCommutator(H, op)
    with pytest.raises(ValueError, match="state must have shape"):
        sc.curve(np.zeros(fock.dim + 2), np.array([0.5]))
    other = bohnasa.FixedNFockSpace(4, 3)
    with pytest.raises(ValueError, match="operator shape"):
        oc.SquaredCommutator(H, bohnasa.Number(other, 0))


def test_tan_units_match_legacy_cos_sin_form():
    """dimer_lambda is expressed in tan(theta) units (PR #23 review); this
    pins it against the original cos/sin form to machine precision across
    the parameter line."""
    grid = np.linspace(0.6, 1.55, 200)

    def legacy_lambda(th):
        c, u = np.cos(th), np.sin(th)
        mu2 = 8.0 * c * (u - 2.0 * c)
        return float(np.sqrt(mu2)) if mu2 > 0.0 else 0.0

    assert max(abs(mf.dimer_lambda(th) - legacy_lambda(th)) for th in grid) < 1e-13


def test_density_matrix_input_is_rejected(dimer):
    """Pure-state scope: a density matrix must fail with the explicit mixed-
    state message, not the generic shape error (which invites flattening)."""
    fock, H, op, state = dimer
    sc = oc.SquaredCommutator(H, op)
    rho = np.outer(state, state.conj())
    with pytest.raises(ValueError, match="mixed states are not supported"):
        sc.curve(rho, np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="mixed states are not supported"):
        sc.commuted(rho, 1.0)


def test_dimer_fixed_points_structure_across_pitchfork():
    below = mf.dimer_fixed_points(1.0)  # tan(1.0) < 2: only the two s=0 points
    assert len(below) == 2
    assert np.all(below[:, 2] == 0.0)
    above = mf.dimer_fixed_points(1.35)
    assert len(above) == 4
    saddle = above[1]
    assert saddle[0] == 0.0 and saddle[1] == pytest.approx(np.pi)
    assert saddle[2] == pytest.approx(mf.dimer_lambda(1.35), rel=1e-12)
    broken = above[2:, 0]
    assert np.allclose(broken, [broken[0], -broken[0]])
    assert broken[0] == pytest.approx(
        np.sqrt(1 - 4 * np.cos(1.35) ** 2 / np.sin(1.35) ** 2), rel=1e-12
    )
    # pitchfork exactly at tan(theta) = 2: lambda turns on with zero broken pair
    th_c = np.arctan(2.0)
    assert mf.dimer_lambda(th_c) == pytest.approx(0.0, abs=1e-15)
    assert len(mf.dimer_fixed_points(th_c + 1e-9)) == 4


def test_jacobian_eigenvalues_are_p_minus_lambda():
    theta = 1.35
    jac = mf.dimer_jacobian(theta, 0.0, np.pi)
    eig = np.linalg.eigvals(jac)
    lam = mf.dimer_lambda(theta)
    assert np.allclose(np.sort(eig.real), [-lam, lam], atol=1e-12)
    assert np.abs(eig.imag).max() < 1e-12


def test_mean_field_flow_rejects_complex_initial():
    with pytest.raises(ValueError, match="real pair"):
        mf.mean_field_flow(1.35, (1e-9 + 1e-3j, np.pi), np.linspace(0, 1, 5))


def test_batched_spectral_curve_matches_pointwise_recursion():
    """The blocked matrix-product evaluation must equal the per-time-point
    vector recursion (which itself matches the brute-force matrix
    commutator - pinned in a separate test)."""
    N = 7
    fock = bohnasa.FixedNFockSpace(N, 2)
    t = np.array([[0.0, -1.7], [-1.7, 0.0]])
    H = bohnasa.Hamiltonian(fock, one_body=t, onsite_u=0.2)
    op = (bohnasa.Number(fock, 0) - bohnasa.Number(fock, 1)) * 0.5
    state = np.zeros(fock.dim, dtype=complex)
    state[fock.dict[(3, 4)]] = 1.0
    times = np.linspace(0.0, 2.5, 23)
    sc = bohnasa.SquaredCommutator(H, op, method="eig")
    batched = sc.curve(state, times)
    pointwise = np.array([np.vdot(v, v).real for v in (sc.commuted(state, t) for t in times)])
    assert np.allclose(batched, pointwise, rtol=1e-9, atol=1e-14)
    assert batched[0] == 0.0  # exact-zero mask survives batching


def test_method_auto_uses_stemr_for_tridiagonal_above_the_dense_limit():
    """Tridiagonal Hamiltonians (the dimer) diagonalize in O(dim^2), so
    auto must NOT downgrade them to the per-point krylov recursion above
    _DENSE_EIG_LIMIT - and non-tridiagonal ones still must."""
    import bohnasa.otoc as oc_module

    N = 2000  # dim 2001 > _DENSE_EIG_LIMIT = 2000
    fock = bohnasa.FixedNFockSpace(N, 2)
    t = np.array([[0.0, -2.0], [-2.0, 0.0]])
    H = bohnasa.Hamiltonian(fock, one_body=t, onsite_u=0.01)
    op = (bohnasa.Number(fock, 0) - bohnasa.Number(fock, 1)) * 0.5
    assert bohnasa.SquaredCommutator(H, op).method == "eig"
    assert oc_module._is_tridiagonal(H.matrix)

    ring = bohnasa.FixedNFockSpace(10, 6)  # dim 3003, banded but not width 1
    H_ring = bohnasa.Hopping(ring, connections="pbc") * (-1.0)
    assert not oc_module._is_tridiagonal(H_ring.matrix)
    sc_ring = bohnasa.SquaredCommutator(H_ring, bohnasa.Number(ring, 0))
    assert sc_ring.method == "krylov"
