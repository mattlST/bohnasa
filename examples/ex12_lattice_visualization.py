"""Example 12: what does a quenched Bose-Hubbard ring *look like*?

Question: can the lattice view of issue #40 turn the numeric tables of the
observables module into a picture that is simultaneously (a) faithful - the
drawn bonds are literally the Hamiltonian's bond list, and the edge color
is the bond current pinned by the continuity equation - and (b)
reproducible - the saved GIF and the PNG frame strip are byte-identical
across repeated builds?

Protocol: N = 6 bosons on an L = 6 ring; all particles start on site 0
(Fock state |6,0,0,0,0,0>, a maximally unbalanced quench), evolving under
-t sum(c^dag_j c_i + h.c.) + (U/2) sum n_i(n_i-1) with t = 1, U = 1
(``Interaction * 0.5`` in the module's unit convention). Nodes pulse with
the on-site occupation (area linear in <n_i>), edges light up with the bond
current from ``one_body_density_matrix``, and square outlines track
doublons ``<n_i(n_i-1)>``. The same arrays feed the 2-D fallback heatmaps.
Exact diagonalization propagates the state (dim = 462, all headless).
"""

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from bohnasa import (
    Correlator,
    FixedNFockSpace,
    Hopping,
    Interaction,
    Number,
    Operator,
    density_profile,
    one_body_density_matrix,
    time_evolution_exact_ind,
)
from bohnasa.visualization import (
    animate_lattice,
    bond_current,
    draw_lattice,
    lattice_bonds,
    lattice_geometry,
    momentum_heatmap,
    occupation_heatmap,
)

N, L = 6, 6
U_HALF = 0.5  # Interaction * 0.5 is the (U/2) n(n-1) term, i.e. U = 1
N_TIME = 49

fock = FixedNFockSpace(N, L)
H = Hopping(fock, connections="pbc") * (-1.0) + Interaction(fock) * U_HALF

psi0 = np.zeros(fock.dim, dtype=complex)
psi0[fock.dict[(N, 0, 0, 0, 0, 0)]] = 1.0
times = np.linspace(0.0, 6.0, N_TIME)

# Observables measured inside one time_evolution_exact_ind call:
# per-site Number, the L^2 Correlators (Correlator(fock, j, i) = c^dag_i c_j
# is gamma_ij), doublon operators n_i(n_i-1) as bare diagonal Operators,
# and the momentum-mode operators a^dag_k a_k assembled from Correlators
# with the convention a_k = L^(-1/2) sum_j exp(-i k j) c_j of
# momentum_distribution.
number_ops = [Number(fock, i) for i in range(L)]
correlator_ops = [Correlator(fock, j, i) for i in range(L) for j in range(L)]
doublon_ops = []
for i in range(L):
    op = Operator(fock)
    occ = fock.fockArray[:, i].astype(float)
    op.matrix = sp.diags(occ * (occ - 1.0))
    doublon_ops.append(op)
momentum_ops = []
for mode in range(L):
    k = 2.0 * np.pi * mode / L
    mat = sp.csr_matrix((fock.dim, fock.dim), dtype=complex)
    for i in range(L):
        for j in range(L):
            mat = mat + np.exp(1j * k * (j - i)) / L * Correlator(fock, j, i).matrix
    op = Operator(fock)
    op.matrix = mat
    momentum_ops.append(op)

result = time_evolution_exact_ind(
    H, psi0, times, number_ops + correlator_ops + doublon_ops + momentum_ops
)
n_obs = result.shape[0]
assert n_obs == 2 * L + L * L + L

# Number, doublon and momentum observables are Hermitian, so their
# imaginary parts are numerical noise only and are dropped with .real.
# result rows: Number (0:L), Correlators (L:L+L^2), doublons, momenta
n_i_t = result[:L, 0, :].real.T  # (T, L)
gamma_t = np.empty((N_TIME, L, L), dtype=complex)
for i in range(L):
    for j in range(L):
        gamma_t[:, i, j] = result[L + i * L + j, 0, :]
dbl_t = result[L + L * L : 2 * L + L * L, 0, :].real.T
n_k_t = result[2 * L + L * L :, 0, :].real.T

print(f"dimension {fock.dim}, {N_TIME} frames, final time t = {times[-1]:g}")

# --- physics anchors -------------------------------------------------------
norm_dev = float(np.abs(n_i_t.sum(axis=1) - N).max())
assert norm_dev < 1e-9, norm_dev
assert np.abs(n_i_t[0] - np.array([N] + [0.0] * (L - 1))).max() < 1e-12
g0 = one_body_density_matrix(psi0, fock)
assert np.allclose(g0, np.diag(n_i_t[0].astype(complex)))
# the drawn bonds ARE the Hamiltonian's bond list (issue #40 requirement 2)
np.testing.assert_array_equal(lattice_bonds(L, "pbc"), H.connections)
bonds = lattice_bonds(L, "pbc")
j_t = np.array([bond_current(g, bonds) for g in gamma_t])
assert j_t.dtype == float  # the encoded current is real by construction
assert np.abs(j_t).max() > 0.1  # the quench genuinely circulates
mid = N_TIME // 2
# the two measured routes must agree: diag(gamma) == <n_i> per frame
assert np.abs(np.real(np.diag(gamma_t[mid])) - n_i_t[mid]).max() < 1e-10
# the momentum-operator means equal the FFT of the measured gamma (same
# convention, independent code paths): n_k = sum_ij conj(E_ki) gamma_ij E_kj
k_vec = 2.0 * np.pi * np.arange(L) / L
phase = np.exp(-1j * np.outer(k_vec, np.arange(L))) / np.sqrt(L)
n_k_from_gamma = np.einsum("ki,tij,kj->tk", phase.conj(), gamma_t, phase).real
assert np.abs(n_k_from_gamma - n_k_t).max() < 1e-9
print(f"number conservation: max deviation {norm_dev:.1e}; |j|max = {np.abs(j_t).max():.3f}")

# --- trajectory-wide normalization (never per frame) ----------------------
node_bounds = (float(n_i_t.min()), float(n_i_t.max()))
edge_abs_max = float(np.abs(j_t).max())
edge_bounds = (-edge_abs_max, edge_abs_max)

out_dir = Path(__file__).resolve().parent
xy = lattice_geometry(L, "pbc")
strip_frames = [0, 10, 20, 29, 38, 48]


def build_strip(path: Path) -> str:
    """Render the six-frame strip with the pinned trajectory bounds."""
    fig, axes = plt.subplots(1, len(strip_frames), figsize=(3.0 * len(strip_frames), 3.4))
    for ax, frame in zip(axes, strip_frames, strict=True):
        draw_lattice(
            ax,
            xy,
            node_sizes=n_i_t[frame],
            node_colors=n_i_t[frame],
            edge_widths=j_t[frame],
            edge_colors=j_t[frame],
            node_vmin=node_bounds[0],
            node_vmax=node_bounds[1],
            edge_vmin=edge_bounds[0],
            edge_vmax=edge_bounds[1],
            node_label="",
            edge_label="",
        )
        ax.set_title(f"t = {times[frame]:g}")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return hashlib.sha256(path.read_bytes()).hexdigest()


strip_path = out_dir / "ex12_lattice_frames.png"
hash_first = build_strip(strip_path)
hash_second = build_strip(out_dir / "ex12_lattice_frames.tmp.png")
assert hash_first == hash_second, "frame strip is not deterministic"
(out_dir / "ex12_lattice_frames.tmp.png").unlink()


def build_gif(path: Path) -> str:
    """Render the replay once and save it headless."""
    ax = plt.subplots(figsize=(5.2, 4.4), dpi=80)[1]
    anim = animate_lattice(times, n_i_t, gamma_t, doublons=dbl_t, ax=ax)
    anim.save(path, writer="pillow", fps=12)
    plt.close("all")  # saving marks the animation as rendered
    return hashlib.sha256(path.read_bytes()).hexdigest()


gif_path = out_dir / "ex12_lattice.gif"
gif_hash_a = build_gif(gif_path)
gif_hash_b = build_gif(out_dir / "ex12_lattice.tmp.gif")
assert gif_hash_a == gif_hash_b, "animation replay is not deterministic"
(out_dir / "ex12_lattice.tmp.gif").unlink()

print(f"saved {strip_path.name} (sha256 {hash_first[:16]}...)")
print(f"saved {gif_path.name} (sha256 {gif_hash_a[:16]}..., byte-deterministic)")

# --- 2-D fallback views ----------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.6))
occupation_heatmap(times, n_i_t, ax=ax1)
ax1.set_title("occupation heatmap n_i(t)")
momentum_heatmap(times, n_k_t, ax=ax2)
ax2.set_title("momentum heatmap n_k(t)")
fig.tight_layout()
fig.savefig(out_dir / "ex12_heatmaps.png", dpi=80)
plt.close(fig)
print("saved ex12_heatmaps.png")

# sanity of the fallback arrays against the density anchor
assert np.abs(density_profile(psi0, fock) - n_i_t[0]).max() < 1e-12
assert np.abs(n_k_t.sum(axis=1) - N).max() < 1e-9  # sum_k n_k = N

print("issue #40 checks passed: bonds == Hopping bond list, strip and GIF byte-identical")
