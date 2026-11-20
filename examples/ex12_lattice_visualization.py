"""Example 12: what does a quenched Bose-Hubbard ring *look like*?

Question: can the lattice view of issue #40 (amended grammar) turn the
numeric tables of the observables module into a picture that is
simultaneously (a) faithful - the drawn bonds are literally the
Hamiltonian's bond list, the edge base encodes the symmetric bond energy
``e_ij = 2 Re gamma_ij`` whose bond sum equals ``<Hopping>`` (= -<H_kin>)
exactly, and the arrowheads encode the current ``j_ij = 2 Im gamma_ij``
pinned by the continuity equation - and (b) reproducible: the annotated
GIF and PNG frame strip are byte-identical across repeated builds?

Protocol: N = 6 bosons on an L = 6 ring or open chain (switch with ``CONN``
below), starting as a Gaussian packet
(width sigma = 1.5 sites around site 0) with winding k0 = 2 pi / 6 - a
CURRENT-CARRYING quench: parity-symmetric density-only initial states
(Fock |6,0,...> or the k = 0 plane wave) have gamma real and hence
j_ij = 0 everywhere, which would leave the arrow channel empty. Under
H = -t sum(c^dag_j c_i + h.c.) + (U/2) sum n_i(n_i-1) with t = 1, U = 1
(``Interaction * 0.5`` in the module's unit convention) the packet
circulates: its measured bond-flux velocity at t = 0 is 1.6590 sites/s,
below the monochromatic group velocity 2 t sin(k0) = 1.7321 sites/s
(the finite momentum spread of sigma = 1.5 shifts weight toward smaller
|sin k|), and it spreads as it goes. Nodes pulse with ``<n_i>``, edges
shade with the bond energy, arrowheads show the current, squares track
doublons; the header/legend/caption blocks state the norms actually used.
Exact diagonalization propagates the state (dim = 462, all headless). The
frame count follows the ``BOHNASA_EXAMPLES_SLOW`` profile: 13 frames by
default (cheap pipeline run), 49 for the full replay.
"""

import hashlib
import os
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
from bohnasa.visualization import (  # _ARROW_THRESHOLD / _legend_lines are
    # module-private on purpose: the printed cutoff and the legend wording must
    # come from the exact constants/formatter the renderer uses (#40 round-3 B7)
    _ARROW_THRESHOLD,
    _legend_lines,
    add_captions,
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
# Pipeline profile: the default samples the same quench coarsely (13
# frames) so example runs stay cheap in CI; set BOHNASA_EXAMPLES_SLOW=1
# for the full 49-frame replay. Both profiles are headless and byte-
# deterministic; every anchor below holds at both frame counts.
SLOW = os.environ.get("BOHNASA_EXAMPLES_SLOW", "0") == "1"
N_TIME = 49 if SLOW else 13
CONN = "pbc"  # "pbc" (ring) or "obc" (open chain) - geometry AND bonds

fock = FixedNFockSpace(N, L)
hop = Hopping(fock, connections=CONN)
H = hop * (-1.0) + Interaction(fock) * U_HALF

# Gaussian packet with winding k0 = 2 pi / 6 (wrapped distance from site 0)
k0 = 2.0 * np.pi / L
dj = np.minimum(np.arange(L), L - np.arange(L))
SIGMA = 1.5
phi0 = np.exp(-(dj**2) / (4.0 * SIGMA**2) + 1j * k0 * np.arange(L))
psi0 = fixed_n_coherent_state(fock, phi0)
times = np.linspace(0.0, 6.0, N_TIME)

# Observables measured inside one time_evolution_exact_ind call: per-site
# Number, the L^2 Correlators (Correlator(fock, j, i) = c^dag_i c_j is
# gamma_ij), doublon operators n_i(n_i-1) as bare diagonal Operators, the
# momentum-mode operators a^dag_k a_k assembled from Correlators with the
# convention a_k = L^(-1/2) sum_j exp(-i k j) c_j of momentum_distribution,
# and the bare Hopping (the bond-energy sum rule).
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
    # a_k = L^(-1/2) sum_j exp(-i k j) c_j (the momentum_distribution convention)
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
np.testing.assert_array_equal(lattice_bonds(L, CONN), hop.connections)
bonds = lattice_bonds(L, CONN)
j_t = np.array([bond_current(g, bonds) for g in gamma_t])
e_t = np.array([bond_energy(g, bonds) for g in gamma_t])
assert j_t.dtype == float and e_t.dtype == float

# bond-energy sum rule: sum of e_ij over the bond list is <Hopping>(t) =
# -<H_kin>(t); the energy channels then conserve H = H_hop + H_int.
sum_rule_dev = float(np.abs(e_t.sum(axis=1) - hop_t).max())
assert sum_rule_dev < 1e-10, sum_rule_dev
h_hop_t = -hop_t  # <H_hop> = <-t Hopping> with t = 1
h_int_t = U_HALF * dbl_t.sum(axis=1)  # <(U/2) sum n(n-1)>
e_dev = float(np.abs(h_hop_t + h_int_t - (h_hop_t[0] + h_int_t[0])).max())
assert e_dev < 1e-9, e_dev

# the quench carries current: measured trajectory max |j| is 2.6848
# (pbc) / 2.6129 (obc), far above the 5%-of-max no-arrow cutoff
j_max = float(np.abs(j_t).max())
cutoff = _ARROW_THRESHOLD * j_max
assert j_max > 2.5, j_max
arrows_visible = np.abs(j_t) > cutoff
assert np.all(arrows_visible.sum(axis=1) >= 1)  # every frame shows >= 1 arrow
# packet flux velocity at t = 0 (mean bond current / site density, sites/s):
# 1.6590 (pbc) / 1.4682 (obc) measured, vs the monochromatic
# 2 t sin(k0) = 1.7321 it approximates; the open chain loses the periodic
# bonds and reflects the packet, so its mean current is smaller.
v0 = float(L * j_t[0].mean() / N)
v_mono = 2.0 * 1.0 * np.sin(k0)
assert 1.4 < abs(v0) < v_mono, (v0, v_mono)
mid = N_TIME // 2
assert np.abs(np.real(np.diag(gamma_t[mid])) - n_i_t[mid]).max() < 1e-10
# the momentum-operator means equal the FFT of the measured gamma (same
# convention, independent code paths): n_k = sum_ij conj(E_ki) gamma_ij E_kj
k_vec = 2.0 * np.pi * np.arange(L) / L
phase = np.exp(-1j * np.outer(k_vec, np.arange(L))) / np.sqrt(L)
n_k_from_gamma = np.einsum("ki,tij,kj->tk", phase.conj(), gamma_t, phase).real
assert np.abs(n_k_from_gamma - n_k_t).max() < 1e-9
e_max = float(np.abs(e_t).max())
print(f"number conservation: max deviation {norm_dev:.1e}; energy: {e_dev:.1e}")
print(f"bond energy |e|max = {e_max:.4f}; sum-rule vs <Hopping> err {sum_rule_dev:.1e}")
print(
    f"bond current |j|max = {j_max:.4f}; cutoff {cutoff:.4f}; arrows in "
    f"{int(arrows_visible.any(axis=1).sum())}/{N_TIME} frames"
)
print(f"packet flux velocity at t=0: {abs(v0):.4f} sites/s (monochromatic {v_mono:.4f})")

# --- trajectory-wide normalization (never per frame) ----------------------
node_bounds = (float(n_i_t.min()), float(n_i_t.max()))
edge_bounds = (-e_max, e_max)  # symmetric: zero bond energy is the color center

# annotation blocks (all deterministic: simulated data only, no wall clock)
_GEOM = "ring " if CONN == "pbc" else "chain"
header_lines = [
    f"Bose-Hubbard {_GEOM}  L={L}  N={N}  nu=N/L={N / L:.1f}  {CONN}",
    f"t_hop = 1.0   U/t = {2 * U_HALF:.1f}   init: Gaussian packet k0=2pi/6 sigma={SIGMA:.1f}",
    f"evolution: t in [0, {times[-1]:g}], {N_TIME} frames   "
    f"arrow cutoff: {_ARROW_THRESHOLD:.0%} of max|j| = {cutoff:.3f}",
]
legend_lines_gif = _legend_lines(node_bounds, "energy", edge_bounds, j_max, cutoff, True)
# the strip has no doublon layer, so its legend must not claim one
legend_lines_strip = _legend_lines(node_bounds, "energy", edge_bounds, j_max, cutoff, False)
live_fields = [
    ("t", times, "5.2f"),
    ("<H_hop>", h_hop_t, "+.3f"),
    ("<H_int>", h_int_t, "+.3f"),
    ("max|j|", np.abs(j_t).max(axis=1), ".3f"),
    ("sum n_i - N", n_i_t.sum(axis=1) - N, "+.1e"),
]

out_dir = Path(__file__).resolve().parent
xy = lattice_geometry(L, CONN)
strip_frames = [round(k * (N_TIME - 1) / 5) for k in range(6)]  # six evenly spread


def build_strip(path: Path) -> str:
    """Six-frame strip: panels without own colorbars, ONE shared pair at
    the figure right, plus the same header/legend captions as the GIF."""
    fig, axes = plt.subplots(1, len(strip_frames), figsize=(2.9 * len(strip_frames) + 1.8, 4.4))
    for ax, frame in zip(axes, strip_frames, strict=True):
        draw_lattice(
            ax,
            xy,
            connections=CONN,
            node_sizes=n_i_t[frame],
            node_colors=n_i_t[frame],
            edge_widths=e_t[frame],
            edge_colors=e_t[frame],
            edge_currents=j_t[frame],
            node_vmin=node_bounds[0],
            node_vmax=node_bounds[1],
            edge_scale=e_max,
            current_scale=j_max,
            colorbar=False,  # the shared pair is added once for the figure
        )
        ax.set_title(f"t = {times[frame]:g}", family="monospace", size=9)
    add_captions(fig, header=header_lines, legend=legend_lines_strip)
    # reserve the header/legend space at the top; the colorbar calls shrink
    # the panel row themselves (setting right= here would re-open the
    # panels into the colorbar axes and overlap them)
    fig.subplots_adjust(top=0.80, bottom=0.04, left=0.01)
    node_bar = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(*node_bounds))
    node_bar.set_array([])
    edge_bar = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(*edge_bounds))
    edge_bar.set_array([])
    # ONE shared colorbar pair placed once at the figure right (round-3 A3a):
    # the node call first reserves the far-right slot, the edge call is then
    # inserted to its left, giving the approved edge-then-node reading order
    fig.colorbar(node_bar, ax=list(axes), label=r"$\langle \hat{n}_i \rangle$", shrink=0.8)
    fig.colorbar(edge_bar, ax=list(axes), label=r"$e_{ij}=2\mathrm{Re}\,\gamma_{ij}$", shrink=0.8)
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return hashlib.sha256(path.read_bytes()).hexdigest()


strip_path = out_dir / "ex12_lattice_frames.png"
hash_first = build_strip(strip_path)
hash_second = build_strip(out_dir / "ex12_lattice_frames.tmp.png")
assert hash_first == hash_second, "frame strip is not deterministic"
(out_dir / "ex12_lattice_frames.tmp.png").unlink()


def build_gif(path: Path) -> str:
    """Render the annotated replay once and save it headless."""
    ax = plt.subplots(figsize=(6.4, 5.4), dpi=80)[1]
    anim = animate_lattice(
        times,
        n_i_t,
        gamma_t,
        doublons=dbl_t,
        connections=CONN,
        header=header_lines,
        legend=legend_lines_gif,
        live=live_fields,
        ax=ax,
    )
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
    "annotated arrows drawn, strip and GIF byte-identical"
)
