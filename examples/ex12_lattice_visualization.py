"""Example 12: what does a quenched Bose-Hubbard ring *look like*?

Question: can the lattice view of issue #40 (amended grammar) turn the
numeric tables of the observables module into a picture that is
simultaneously (a) faithful - the drawn bonds are literally the
Hamiltonian's bond list, the edge base encodes the symmetric bond energy
``e_ij = 2 Re gamma_ij`` whose bond sum equals ``<Hopping>`` exactly, and
the arrowheads encode the current ``j_ij = 2 Im gamma_ij`` pinned by the
continuity equation - and (b) reproducible: the saved GIF and PNG frame
strip are byte-identical across repeated builds?

Protocol: N = 6 bosons on an L = 6 ring, starting as a Gaussian packet
(occupation widths sigma = 1.5 sites around site 0) with winding k0 =
2 pi / 6 - a CURRENT-CARRYING quench: the parity-symmetric density-only
initial states (Fock |6,0,...> or the k = 0 plane wave) have gamma real
and hence j_ij = 0 everywhere, which would leave the arrow channel empty.
The packet circulates with group velocity 2 t sin(k0) while spreading
under U = 1 (``Interaction * 0.5`` in the module's unit convention,
t = 1). Node area pulses with ``<n_i>``, edges shade with the bond
energy, arrowheads show the current, squares track doublons. Exact
diagonalization propagates the state (dim = 462, all headless).
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
    fixed_n_coherent_state,
    one_body_density_matrix,
    time_evolution_exact_ind,
)
from bohnasa.visualization import (
    animate_lattice,
    bond_current,
    bond_energy,
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
hop = Hopping(fock, connections="pbc")
H = hop * (-1.0) + Interaction(fock) * U_HALF

# Gaussian packet with winding k0 = 2 pi / 6 (wrapped distance from site 0)
k0 = 2.0 * np.pi / L
dj = np.minimum(np.arange(L), L - np.arange(L))
phi0 = np.exp(-(dj**2) / (4.0 * 1.5**2) + 1j * k0 * np.arange(L))
psi0 = fixed_n_coherent_state(fock, phi0)
times = np.linspace(0.0, 6.0, N_TIME)

# Observables measured inside one time_evolution_exact_ind call:
# per-site Number, the L^2 Correlators (Correlator(fock, j, i) = c^dag_i c_j
# is gamma_ij), doublon operators n_i(n_i-1) as bare diagonal Operators,
# the momentum-mode operators a^dag_k a_k assembled from Correlators with
# the convention a_k = L^(-1/2) sum_j exp(-i k j) c_j of
# momentum_distribution, and the bare Hopping (the bond-energy sum rule).
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
    # a^dag_k a_k = (1/L) sum_ij exp(i k (i - j)) c^dag_i c_j for
    # a_k = L^(-1/2) sum_j exp(-i k j) c_j  (the momentum_distribution convention)
    for i in range(L):
        for j in range(L):
            mat = mat + np.exp(1j * k * (i - j)) / L * Correlator(fock, j, i).matrix
    op = Operator(fock)
    op.matrix = mat
    momentum_ops.append(op)

result = time_evolution_exact_ind(
    H, psi0, times, number_ops + correlator_ops + doublon_ops + momentum_ops + [hop]
)
assert result.shape[0] == 3 * L + L * L + 1

# Number, doublon, momentum and Hopping are Hermitian observables, so their
# imaginary parts are numerical noise only and are dropped with .real.
n_i_t = result[:L, 0, :].real.T  # (T, L)
gamma_t = np.empty((N_TIME, L, L), dtype=complex)
for i in range(L):
    for j in range(L):
        gamma_t[:, i, j] = result[L + i * L + j, 0, :]
dbl_t = result[L + L * L : 2 * L + L * L, 0, :].real.T
n_k_t = result[2 * L + L * L : 3 * L + L * L, 0, :].real.T
hop_t = result[3 * L + L * L, 0, :].real  # <Hopping>(t)

print(f"dimension {fock.dim}, {N_TIME} frames, final time t = {times[-1]:g}")

# --- physics anchors -------------------------------------------------------
norm_dev = float(np.abs(n_i_t.sum(axis=1) - N).max())
assert norm_dev < 1e-9, norm_dev
assert np.abs(n_i_t[0] - density_profile(psi0, fock)).max() < 1e-12
assert np.abs(n_i_t[0, 0] - 1.6859) < 1e-4  # packet peak on site 0
g0 = one_body_density_matrix(psi0, fock)
assert np.allclose(g0.real.diagonal(), n_i_t[0])
# the drawn bonds ARE the Hamiltonian's bond list (issue #40 requirement 2)
np.testing.assert_array_equal(lattice_bonds(L, "pbc"), hop.connections)
bonds = lattice_bonds(L, "pbc")
j_t = np.array([bond_current(g, bonds) for g in gamma_t])
e_t = np.array([bond_energy(g, bonds) for g in gamma_t])
assert j_t.dtype == float and e_t.dtype == float

# the bond-energy sum rule: sum of e_ij over the bond list is <Hopping>
sum_rule_dev = float(np.abs(e_t.sum(axis=1) - hop_t).max())
assert sum_rule_dev < 1e-10, sum_rule_dev

# the quench carries current: measured over this protocol the trajectory
# max |j| is 2.6848..., well above the 5%-of-max no-arrow threshold
j_max = float(np.abs(j_t).max())
assert j_max > 2.5, j_max
arrows_visible = (np.abs(j_t) > 0.05 * j_max).sum(axis=1)
assert np.all(arrows_visible >= 1)  # every frame shows at least one arrowhead
e_max = float(np.abs(e_t).max())
mid = N_TIME // 2
assert np.abs(np.real(np.diag(gamma_t[mid])) - n_i_t[mid]).max() < 1e-10
k_vec = 2.0 * np.pi * np.arange(L) / L
phase = np.exp(-1j * np.outer(k_vec, np.arange(L))) / np.sqrt(L)
n_k_from_gamma = np.einsum("ki,tij,kj->tk", phase.conj(), gamma_t, phase).real
assert np.abs(n_k_from_gamma - n_k_t).max() < 1e-9
print(f"number conservation: max deviation {norm_dev:.1e}")
print(f"bond energy |e|max = {e_max:.4f}; sum-rule vs <Hopping> err {sum_rule_dev:.1e}")
n_arrows = int((arrows_visible > 0).sum())
print(f"bond current |j|max = {j_max:.4f}; arrows visible in {n_arrows}/{N_TIME} frames")

# --- trajectory-wide normalization (never per frame) ----------------------
node_bounds = (float(n_i_t.min()), float(n_i_t.max()))
edge_bounds = (-e_max, e_max)  # symmetric: zero bond energy is the color center

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
            edge_widths=e_t[frame],
            edge_colors=e_t[frame],
            edge_currents=j_t[frame],
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
assert np.abs(n_k_t.sum(axis=1) - N).max() < 1e-9  # sum_k n_k = N

print(
    "issue #40 checks passed: bond list shared with Hopping, e-sum rule, "
    "arrows drawn, strip and GIF byte-identical"
)
