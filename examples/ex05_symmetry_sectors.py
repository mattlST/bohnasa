"""Example 05: symmetry sectors - same physics, much smaller matrices.

Question: how far do the spatial symmetries of the chain shrink exact
diagonalization?

Reassembles the full spectrum of a 4-site ring from its momentum/parity
blocks (issue #13), then takes the staggered quench of example 03 to a
6-site ring (1287-dimensional): the state lives entirely in one
126-dimensional sector, so the Krylov chain is built there and checked
against exact full-space time evolution. A long full-space Lanczos chain
is shown to leak across sectors after a few dozen steps - inside the
block, where it belongs, there is nowhere to leak to.
"""

import matplotlib.pyplot as plt
import numpy as np

from bohnasa import (
    FixedNFockSpace,
    Hopping,
    Interaction,
    SectorLabel,
    SymmetrySectors,
    fixed_n_coherent_state,
    lanczos,
)

# --- a small ring where all blocks can be checked against full ED ---
N, L = 6, 4
fock = FixedNFockSpace(N, L)
H = Hopping(fock, connections="pbc") * (-1.0) + Interaction(fock) * 0.5
sectors = SymmetrySectors(fock, connections="pbc")

block_energies, block_minima = [], []
print(f"ring N={N}, L={L}: dim {fock.dim}, {len(sectors.labels)} sectors")
for label in sectors.labels:
    eig = np.linalg.eigvalsh(sectors.restrict(H, label).toarray())
    block_energies.append(eig)
    block_minima.append(eig[0])
    print(f"  {label}: dim {sectors.dim(label):3d}, block minimum {eig[0]:.6f}")

full_spectrum = np.linalg.eigvalsh(H.matrix.toarray())
union = np.sort(np.concatenate(block_energies))
print(f"spectrum union vs full ED: max difference {np.abs(union - full_spectrum).max():.1e}")
assert np.allclose(union, full_spectrum, atol=1e-10)
assert abs(block_minima[0] - full_spectrum[0]) < 1e-12  # ground state in the (0,+) block

# --- the quench of example 03, now on a 6-site ring ---
N, L = 8, 6
fock6 = FixedNFockSpace(N, L)
H6 = Hopping(fock6, connections="pbc") * (-1.0) + Interaction(fock6) * 4.0
sectors6 = SymmetrySectors(fock6, connections="pbc")
phi = 1j * np.array([(-1) ** j for j in range(L)], dtype=complex)
sta = fixed_n_coherent_state(fock6, phi)
sta = sta / np.linalg.norm(sta)

support = {
    str(label): float(np.linalg.norm(sectors6.to_sector(sta, label)) ** 2)
    for label in sectors6.labels
}
biggest = max(sectors6.dim(lab) for lab in sectors6.labels)
print(f"\nring N={N}, L={L}: dim {fock6.dim}, largest block {biggest}")
print("sector support of the staggered coherent state:")
for key, value in support.items():
    if value > 1e-12:
        print(f"  {key}: {value:.12f}")
assert max(support.values()) > 1 - 1e-12, "quench should live in a single sector"

sector = SectorLabel(0, 1)
psi_s = sectors6.to_sector(sta, sector)
psi_s = psi_s / np.linalg.norm(psi_s)
block = sectors6.restrict(H6, sector)

# Krylov chain in the 126-dimensional block ...
v, T = lanczos(psi_s, block.dot, block.shape[1] - 1)
n = v.shape[0]
K_chain = v.conj().T @ np.diag(np.arange(n)) @ v
times = np.linspace(0.0, 4.0, 150)
E, W = np.linalg.eigh(T)
seed = np.zeros(n, dtype=complex)
seed[0] = 1.0
krylov_curve = np.array(
    [
        np.dot(np.arange(n), np.abs(W @ (np.exp(-1j * E * t) * (W.conj().T @ seed))) ** 2)
        for t in times
    ]
)

# ... and the exact full-space evolution as ground truth for it
u_sector = sectors6.isometry(sector).toarray()
K_full = u_sector @ K_chain @ u_sector.conj().T
E_f, W_f = np.linalg.eigh(H6.matrix.toarray())
c_f = W_f.conj().T @ sta
krylov_exact = []
for t in times:
    psi_t = W_f @ (np.exp(-1j * E_f * t) * c_f)
    krylov_exact.append(np.vdot(psi_t, K_full @ psi_t).real)
krylov_exact = np.array(krylov_exact)

dev = np.abs(krylov_curve - krylov_exact).max()
print(f"chain in the block: {n} Krylov vectors")
print(f"block-chain vs exact full-space evolution: max deviation {dev:.1e}")
assert dev < 1e-9

# a naive full-space Lanczos of comparable length leaks across sectors:
_, T_full = lanczos(sta, H6, block.shape[1] - 1)
k = min(T_full.shape[0], n)
beta = np.abs(np.diag(T, -1))
leak_at = int(np.argmax(np.abs(beta - np.abs(np.diag(T_full, -1))[:k]) > 1e-6))
print(
    f"full-space Lanczos starts to deviate at step {leak_at} (1/beta amplification of sector noise)"
)

block_work = sum(sectors6.dim(lab) ** 3 for lab in sectors6.labels)
print(f"dense-ED work of all blocks vs full space: {block_work / fock6.dim**3:.1%}")

fig, (ax_blocks, ax_k) = plt.subplots(1, 2, figsize=(11, 4))
names = [str(lab) for lab in sectors.labels]
ax_blocks.bar(range(len(names)), [sectors.dim(lab) for lab in sectors.labels])
ax_blocks.set_xticks(range(len(names)), names, rotation=45, ha="right")
ax_blocks.set_ylabel("block dimension")
ax_blocks.set_title(f"sectors of the N={fock.N}, L={fock.L} ring (dim {fock.dim})")
ax_k.plot(times, krylov_curve, label="126-dim block chain")
ax_k.plot(times, krylov_exact, ":", lw=2, label="exact full space")
ax_k.set_xlabel("time")
ax_k.set_ylabel(r"$\langle \hat{K} \rangle(t)$")
ax_k.legend()
ax_k.set_title("staggered quench, N=8, L=6 ring")
fig.tight_layout()
plt.show()

print(f"<K> saturates at {krylov_curve.max():.2f} of possible {n - 1}")
