"""Example 13: the kicked Bose-Hubbard ring - Floquet spectra, OTOC, classical map.

Question: what does the issue #29 kicked-ring API show that the hand-assembled
Trotterization of ex10 could not state as cleanly - quasienergy branches with
their gaps, a Floquet OTOC comparable to its continuous-time twin, and a
classical kicked map beside the quantum step?

Protocol. KickedBoseHubbard(L, tau, J, U) drives one period as
exp(-i tau H_J) exp(-i tau H_U) (module docstring conventions): a plane-wave
coherent quench on the L = 3 ring (N = 6 for the spectra, N = 10 with
dim = 66 for the OTOC) feeds SquaredCommutator.from_propagator, the
quasienergy panel sweeps U at fixed tau through the blocked
floquet_spectrum, the density panel propagates a Gaussian packet
(sigma = 1.2 sites) step by step against exact diagonalization, and the
classical panel projects 200-step orbits of KickedMeanFieldMap onto the
two relative phases. Everything runs headless and is byte-deterministic
(no random draws, no wall-clock values in the output).

Every printed number below is a measured deviation, asserted against a
rounding-headroom bound: (1) the library step equals the issue's
hand-assembled expm workaround to machine precision, (2) the Floquet OTOC
round trip against explicit Heisenberg matrix powers, (3) unitarity at the
strong-kick corner (on-site phase winds 91 rad across the basis), (4) the
classical dimer map at tau = 0.05 tracking mean_field_flow (the validation
of issue #29, full Richardson study in tests/physics/test_kicked_physics.py),
(5) exact-norm conservation of the ring map and the small-tau agreement of
kicked density / OTOC curves with the continuous drive, and (6) the
quasienergy branches staying inside the half-open Floquet zone including the
zone-edge relabelling at U = 0.
"""

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla

from bohnasa import (
    FixedNFockSpace,
    KickedBoseHubbard,
    Number,
    SquaredCommutator,
    fixed_n_coherent_state,
)
from bohnasa.mean_field import dimer as mf

THETA = 1.35  # same unstable-saddle angle as ex09/ex10 (for the dimer check)

# --- guard 1: the library step IS the hand-assembled workaround -------------
k = KickedBoseHubbard(L=3, tau=0.7, J=1.0, U=0.9)
fock = FixedNFockSpace(6, 3)
h_j = k.hopping_hamiltonian(fock).matrix.toarray()
h_u = k.interaction_hamiltonian(fock).matrix.toarray()
u_hand = sla.expm(-1j * 0.7 * h_j) @ np.diag(np.exp(-1j * 0.7 * np.diag(h_u)))
dev_hand = np.abs(k.propagator(fock) - u_hand).max()
assert dev_hand < 1e-12  # observed 6.2e-17
print(f"library step == hand-assembled expm workaround, max dev {dev_hand:.1e}")

# --- guard 2: Floquet OTOC round trip (validation 3) -------------------------
L, N, J, U = 3, 10, 1.0, 0.9
fock = FixedNFockSpace(N, L)
psi = fixed_n_coherent_state(
    fock, np.array([1.0, np.exp(2j * np.pi / 3), np.exp(-2j * np.pi / 3)]) / np.sqrt(3.0)
)
op = Number(fock, 0)
k4 = KickedBoseHubbard(L, 0.4, J, U)
u4 = k4.propagator(fock)
n_probe = 8
lib = SquaredCommutator.from_propagator(u4, op).curve(psi, np.arange(n_probe + 1))
od = op.matrix.toarray()
brute = np.empty(n_probe + 1)
for n in range(n_probe + 1):
    un = np.linalg.matrix_power(u4, n)
    o_n = un.conj().T @ od @ un
    w = o_n @ (od @ psi) - od @ (o_n @ psi)
    brute[n] = np.vdot(w, w).real
dev_rt = np.abs(lib - brute).max()
assert dev_rt < 1e-10  # observed 1.1e-13
assert lib[0] == 0.0 and brute[0] == 0.0
print(f"from_propagator curve == explicit Heisenberg growth, max dev {dev_rt:.1e}")

# --- guard 3: unitarity at the strong-kick corner (validation 1) -------------
k_strong = KickedBoseHubbard(L=3, tau=2.5, J=0.7, U=1.3)
u_strong = k_strong.propagator(FixedNFockSpace(8, 3))
dev_uni = np.abs(u_strong.conj().T @ u_strong - np.eye(u_strong.shape[0])).max()
assert dev_uni < 5e-15  # observed 1.11e-15 (on-site phase winds 91 rad)
print(f"strong-kick unitarity |U^dag U - I|max = {dev_uni:.1e}")

# --- guard 4: classical dimer map -> mean_field_flow at small tau (valid. 2) -
k_dim = KickedBoseHubbard(L=2, tau=0.05, J=np.cos(THETA), U=2 * np.sin(THETA) / 20.0)
m_dim = k_dim.classical_map()
phi_d, occ_d = m_dim.iterate((np.array([0.3, -1.1]), np.array([12.0, 8.0])), 80)
s_d, ph_d = m_dim.dimer_variables(phi_d, occ_d)
ref_d = mf.mean_field_flow(THETA, (float(s_d[0]), float(ph_d[0])), 0.05 * np.arange(81))
ds = np.abs(s_d - ref_d[:, 0]).max()
dphi = np.abs((ph_d - ref_d[:, 1] + np.pi) % (2 * np.pi) - np.pi).max()
assert ds < 1.5e-2 and dphi < 4e-2  # observed 9.82e-03 / 2.95e-02 (first order in tau)
print(f"classical map at tau = 0.05 tracks mean_field_flow: |ds| {ds:.1e}, |dphi| {dphi:.1e}")

# --- panel A data: quasienergy branches vs U (blocked spectrum) --------------
tau_spec, fold = 0.7, 2 * np.pi / 0.7
u_grid = np.linspace(0.0, 2.0, 401)
fock_spec = FixedNFockSpace(6, 3)
branches = np.array(
    [
        np.sort(KickedBoseHubbard(3, tau_spec, 1.0, u_val).floquet_spectrum(fock_spec)[0])
        for u_val in u_grid
    ]
)
assert np.all(branches < fold) and np.all(branches >= 0.0)
print(
    f"quasienergy branches inside [0, 2 pi / tau) for all {len(u_grid)} sweep points "
    f"(max {branches.max():.6f} < {fold:.6f}; zone-edge states relabelled to 0)"
)

# --- panel B data: Floquet vs continuous OTOC --------------------------------
t_end = 6.0
tg = np.linspace(0.0, t_end, 601)
c_cont = SquaredCommutator(
    k4.hopping_hamiltonian(fock) + k4.interaction_hamiltonian(fock), op
).curve(psi, tg)
oto = {}
for tau in (0.05, 0.4, 1.0):
    steps = np.arange(int(round(t_end / tau)) + 1)
    c = SquaredCommutator.from_propagator(
        KickedBoseHubbard(L, tau, J, U).propagator(fock), op
    ).curve(psi, steps)
    ref = np.interp(steps * tau, tg, c_cont)
    dev = np.abs(c[1:] / ref[1:] - 1.0).max()
    oto[tau] = (steps * tau, c, dev)
assert oto[0.05][2] < 5e-2  # observed 3.2e-02
print(f"tau = 0.05 OTOC tracks the continuous curve, max relative dev {oto[0.05][2]:.1e}")

# --- panel C data: kicked density propagation vs exact evolution -------------
tau_rho = 0.1
k_rho = KickedBoseHubbard(3, tau_rho, J, U)
fock_rho = FixedNFockSpace(8, 3)
dj = np.minimum(np.arange(3), 3 - np.arange(3))
psi_rho = fixed_n_coherent_state(fock_rho, np.exp(-(dj**2) / (4 * 1.2**2)))
u_rho = k_rho.propagator(fock_rho)
n0_mat = Number(fock_rho, 0).matrix.toarray()
h_dense = (
    k_rho.hopping_hamiltonian(fock_rho) + k_rho.interaction_hamiltonian(fock_rho)
).matrix.toarray()
eps_h, V = np.linalg.eigh(h_dense)
v, n_kick = psi_rho.copy(), []
for _ in range(61):
    n_kick.append(float(np.vdot(v, n0_mat @ v).real))
    v = u_rho @ v
n_kick = np.array(n_kick)
times_rho = tau_rho * np.arange(61)
c0 = V.conj().T @ psi_rho


def n_exact(t):
    w = V @ (np.exp(-1j * eps_h * t) * c0)
    return float(np.vdot(w, n0_mat @ w).real)


n_cont_fine = np.array([n_exact(t) for t in tg])
dev_rho = np.abs(n_kick - np.array([n_exact(t) for t in times_rho])).max()
assert dev_rho < 1e-1  # observed 8.1e-02
dev_norm = abs(np.vdot(v, v) - 1.0)
assert dev_norm < 1e-12  # observed 3.3e-15 (norm of the 61-step chain)
print(f"kicked density (tau = 0.1) within {dev_rho:.1e} of the exact evolution over t <= 6")

# --- panel D data: classical kicked ring orbits -------------------------------
m_ring = KickedBoseHubbard(L=3, tau=0.7, J=1.0, U=1.5).classical_map()
orbits = []
norm_dev = 0.0
for phi0 in (
    np.array([0.0, 0.1, -0.1]),
    np.array([0.0, 1.9, -1.1]),
    np.array([1e-3, np.pi, np.pi]),
):
    p, o = m_ring.iterate((phi0, np.full(3, 10 / 3)), 200)
    norm_dev = max(norm_dev, np.abs(o.sum(axis=1) - 10.0).max())
    rel = (p[:, 1:] - p[:, :1] + np.pi) % (2 * np.pi) - np.pi
    orbits.append((rel[:, 0], rel[:, 1], o[:, 0]))
assert norm_dev < 1e-11  # observed 1.0e-12
print(
    f"map conserves the norm shell to {norm_dev:.1e}; orbits roam "
    f"n_0 in [{min(orb[2].min() for orb in orbits):.2f}, "
    f"{max(orb[2].max() for orb in orbits):.2f}] of the N = 10 shell"
)

# --- figure -------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4))
ax_eps, ax_otoc, ax_rho, ax_map = axes.ravel()

for row in branches.T:
    ax_eps.plot(u_grid, row, color="navy", lw=0.5)
ax_eps.axhline(0.0, color="0.6", lw=0.8)
ax_eps.axhline(fold, color="0.6", lw=0.8, ls="--")
ax_eps.set_xlim(0.0, 2.0)
ax_eps.set_ylim(-0.3, fold + 0.3)
ax_eps.set_xlabel(r"$U$")
ax_eps.set_ylabel(r"quasienergy $\epsilon \in [0, 2\pi/\tau)$")
ax_eps.set_title(rf"blocked Floquet branches, $U$-sweep ($\tau = {tau_spec}$)")

ax_otoc.semilogy(tg, c_cont, color="black", lw=1.2, label="continuous")
for tau, style in ((0.05, "-"), (0.4, "-o"), (1.0, "-s")):
    times, c, _ = oto[tau]
    ax_otoc.semilogy(
        times, np.maximum(c, 1e-300), style, ms=3, lw=1.0, label=f"kicked $\\tau = {tau}$"
    )
ax_otoc.set_xlabel(r"time $t = n\tau$ (kick counts $n$)")
ax_otoc.set_ylabel(r"$C = \Vert[O(n), O]\vert\psi\rangle\Vert^2$")
ax_otoc.set_title(r"Floquet OTOC of the kicked ring ($L = 3$, $N = 10$, $U = 0.9$)")
ax_otoc.legend(fontsize=8)

times_rho = tau_rho * np.arange(61)
ax_rho.plot(tg, n_cont_fine, lw=1.0, color="0.55", label=r"exact $\exp(-iHt)$")
ax_rho.plot(times_rho, n_kick, "o", ms=3.5, color="firebrick", label=f"kicked $\tau = {tau_rho}$")
ax_rho.set_xlabel(r"$t = n\tau$")
ax_rho.set_ylabel(r"$\langle n_0\rangle_n$")
ax_rho.set_title(r"density propagation, Gaussian packet ($\sigma = 1.2$)")
ax_rho.legend(fontsize=8)

for x, y, n0 in orbits:
    sc = ax_map.scatter(x, y, c=n0, s=4, cmap="viridis")
ax_map.axhline(0.0, color="0.7", lw=0.6)
ax_map.axvline(0.0, color="0.7", lw=0.6)
ax_map.set_xlim(-np.pi, np.pi)
ax_map.set_ylim(-np.pi, np.pi)
ax_map.set_xlabel(r"$\varphi_1 - \varphi_0$")
ax_map.set_ylabel(r"$\varphi_2 - \varphi_0$")
ax_map.set_title(r"classical kicked ring map, 200 steps ($U = 1.5$, colour $n_0$)")
fig.colorbar(sc, ax=ax_map, label=r"$n_0$", shrink=0.8)

fig.tight_layout()


out_dir = Path(__file__).resolve().parent
path = out_dir / "ex13_kicked_ring.png"
fig.savefig(path, dpi=80)
hash_first = hashlib.sha256(path.read_bytes()).hexdigest()
fig.savefig(out_dir / "ex13_kicked_ring.tmp.png", dpi=80)
hash_second = hashlib.sha256((out_dir / "ex13_kicked_ring.tmp.png").read_bytes()).hexdigest()
assert hash_first == hash_second, "figure is not deterministic"
(out_dir / "ex13_kicked_ring.tmp.png").unlink()
plt.close(fig)
print(f"saved {path.name} (sha256 {hash_first[:16]}..., byte-deterministic)")

print(
    "issue #29 checks passed: step == hand-assembled workaround, OTOC round trip, "
    "strong-kick unitarity, classical map -> mean_field_flow, branch zone closed, "
    "density and norm conservation"
)
