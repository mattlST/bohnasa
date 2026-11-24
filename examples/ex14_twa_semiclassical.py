"""Example 14: the truncated-Wigner bridge between quantum scrambling and
classical Lyapunov growth.

Question: can a classical ensemble sampled from the Wigner representation
reproduce the exact quantum squared commutator at short time, follow the
engine's own Lyapunov growth at intermediate time, and where does it
break?

Setup: the L = 2 interacting dimer above the pitchfork (theta = 1.35, the
unstable-saddle regime of examples/ex09 and the OTOC physics anchors),
quantized with N_phys = 16 bosons on the fixed-N shell, so the TWA
expansion parameter is hbar_eff = 1/16. The truncated-Wigner ensemble
propagates each sample with the mean-field engine's own Radau flow; the
quantum reference is SquaredCommutator (dense eigensolve, dim 17) plus
StateObservables for the state-resolved line, and the single-trajectory
mean-field reference is the Poisson-bracket point value at the
Gross-Pitaevskii saddle. Conventions are derived once in bohnasa/twa.py.

Panel 1 shows the Ehrenfest structure: C(t) from quantum and TWA agree
at short time (measured relative deviation ~ hbar_eff, pinned in
tests/physics/test_twa_physics.py), both run through the hyperbolic
rates of the issue #34 crossover (log sqrt(C) exponents 2 lambda_s then
lambda_s), and the TWA curve then holds the classical growth while the
quantum one bends toward saturation. Panel 2 is the validation-1
statement on a REGULAR (below-pitchfork) orbit: the sample mean
collapses onto the deterministic ensemble average at the Monte-Carlo
1/sqrt(samples) rate; the gap that survives is the O(hbar_eff)
ensemble-vs-mean-field back-reaction, drawn as the plateau. Panel 3 is
the geometric view (bohnasa.visualization): occupation heatmaps of the
TWA sample mean, the single mean-field orbit, and an exact-quantum ED
reference (fixed-N ring, N_phys = L = 4, dim 35) on an L = 4 ring.

Headless and byte-reproducible: fixed seeds everywhere, and the figure
pair is rendered twice from the same data and hash-compared (Agg).
"""

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bohnasa import (
    FixedNFockSpace,
    Hamiltonian,
    MeanFieldLattice,
    Number,
    SquaredCommutator,
    StateObservables,
    TruncatedWigner,
    dimer_lambda,
    fixed_n_coherent_state,
)
from bohnasa.visualization import occupation_heatmap

THETA = 1.35  # unstable dimer saddle regime (examples/ex09 anchor point)
N_PHYS = 16
# The ensemble sizes used below (240 / 800 / 384 / 64 samples, plus the
# dim-35 ED eigensolve) keep the full-script runtime at ~1 min 50 s on
# this 16-core dev host (numpy 2.5.5 / scipy 1.18.1, cpufreq powersave
# governor, measured twice with `time`: 1m48.9s and 1m51.4s); the
# physics values they produce are re-pinned with independent margins in
# tests/physics/test_twa_physics.py, and no timing enters the printed
# output (CODING_RULES performance-path justification channel).
HBAR = 1.0 / N_PHYS
LAT = 2
LAM = dimer_lambda(THETA)
print(f"classical saddle exponent lambda_s = {LAM:.6f}, hbar_eff = {HBAR:.4f}")

lat = MeanFieldLattice(LAT, THETA)
# dimer saddle in engine coordinates: phi = (1, -1) on the N = L shell
ZGP = np.array([np.sqrt(2.0), -np.sqrt(2.0), 0.0, 0.0])

fock = FixedNFockSpace(N_PHYS, LAT)
coup = np.cos(THETA)
t = np.array([[0.0, -2 * coup], [-2 * coup, 0.0]])
H = Hamiltonian(fock, one_body=t, onsite_u=np.sin(THETA) / N_PHYS)
op = (Number(fock, 0) - Number(fock, 1)) * 0.5
psi = fixed_n_coherent_state(fock, np.array([1.0, -1.0]) / np.sqrt(2.0))
sc = SquaredCommutator(H, op)


# symbol partner (b) of the quantum probe: (n0 - n1)/2  <->  (N/L) s_engine
def probe(z: np.ndarray) -> float:
    return (N_PHYS / LAT) * 0.25 * (z[0] ** 2 + z[2] ** 2 - z[1] ** 2 - z[3] ** 2)


times = np.linspace(0.0, 2.0, 41)
c_q = sc.curve(psi, times)
sim = TruncatedWigner(lat, ZGP, seed=3031, hbar_eff=HBAR)
c_twa, c_se = sim.otoc(240, times, probe)

# classical point value: rule (c) evaluated at the unsampled GP point;
# the gradient of this quadratic symbol is analytic
grad_probe = (N_PHYS / LAT) * 0.5 * np.array([ZGP[0], -ZGP[1], ZGP[2], -ZGP[3]])
x_o = np.concatenate((grad_probe[2:], -grad_probe[:2]))  # Omega grad O
eps = 1e-4
br_point = np.array(
    [
        probe(zp) - probe(zm)
        for zp, zm in zip(
            lat.trajectory(ZGP + eps * x_o, times),
            lat.trajectory(ZGP - eps * x_o, times),
            strict=True,
        )
    ]
) / (2 * eps)
hbar_moyal = LAT * HBAR
c_point = (hbar_moyal * br_point) ** 2

# --- anchors ---------------------------------------------------------------
# quantum state-resolved reference through StateObservables (issue #20 path):
# the saddle coherent state is stationary under ED, so any TWA motion of the
# sampled occupations is sampling + truncation, not physics.
obs = StateObservables(fock)
eigval, eigvec = np.linalg.eigh(H.matrix.toarray())
coef = eigvec.conj().T @ psi


def phys_n(z: np.ndarray) -> np.ndarray:
    return (N_PHYS / LAT) * (0.5 * (z[:LAT] ** 2 + z[LAT:] ** 2)) - 0.5


t_anchor = 0.4
psi_t = eigvec @ (np.exp(-1j * eigval * t_anchor) * coef)
n_q = obs.density_profile(psi_t)
mean_n, se_n = sim.observable(800, np.array([t_anchor]), phys_n)
print(
    f"occupations at t={t_anchor}: quantum {np.round(n_q, 4)}, "
    f"TWA {np.round(mean_n[0], 4)} +- {np.round(se_n[0], 4)}"
)
assert np.abs(n_q - n_q[0]).max() < 1e-9  # ED saddle stationary
assert np.all(np.abs(mean_n[0] - n_q) < 3.0 * se_n[0])  # TWA within 3 sigma

i_short = 9  # t = 0.45
rel_short = (c_twa[i_short] - c_q[i_short]) / c_q[i_short]
print(
    f"C(t={times[i_short]:.2f}): quantum {c_q[i_short]:.4f}, TWA {c_twa[i_short]:.4f}"
    f" (dev {rel_short:+.4f} = {rel_short / HBAR:+.2f} x hbar_eff)"
)
assert 0.5 * HBAR < rel_short < 0.25


def slope_sqrt_c(vals, t0, t1):
    m = (times >= t0) & (times <= t1) & (vals > 0)
    return float(np.polyfit(times[m], np.log(vals[m]), 1)[0] / 2.0)


w_early, w_mid = slope_sqrt_c(c_twa, 0.3, 0.8), slope_sqrt_c(c_twa, 1.0, 1.5)
q_early, q_mid = slope_sqrt_c(c_q, 0.3, 0.8), slope_sqrt_c(c_q, 1.0, 1.5)
p_early, p_mid = slope_sqrt_c(c_point, 0.3, 0.8), slope_sqrt_c(c_point, 1.0, 1.5)
print("sqrt-C exponents, early / mid window:")
print(f"  TWA      {w_early:.4f} / {w_mid:.4f}")
print(f"  quantum  {q_early:.4f} / {q_mid:.4f}")
print(f"  point    {p_early:.4f} / {p_mid:.4f}")
print(f"  targets  2*lambda_s = {2 * LAM:.4f} / lambda_s = {LAM:.4f}")
assert abs(w_early / (2 * LAM) - 1) < 0.08
assert abs(w_mid / LAM - 1) < 0.08
assert abs(q_early / (2 * LAM) - 1) < 0.08
# NOTE: the single-trajectory point value carries a larger early/mid
# transient (+7.5 % / +20 %) - it holds the pure unstable-manifold growth
# without the packet-average that forms the crossover's lambda_s side.
# Documented in bohnasa/twa.py; not asserted here (measured, informational).

# --- panel 1: quantum vs TWA vs classical point value ----------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
ax1.semilogy(times[1:], c_q[1:], label="quantum $C(t)$")
ax1.semilogy(times[1:], c_twa[1:], label="TWA (240 samples)")
ax1.semilogy(times[1:], c_point[1:], "--", lw=1.0, label="classical point value")
tau_e = np.log(N_PHYS) / LAM
ax1.axvline(tau_e, color="grey", lw=0.8, ls=":")
ax1.text(
    tau_e * 1.03,
    float(c_q[1:].min()) * 2,
    f"$\\tau_E=\\ln N/\\lambda_s$ = {tau_e:.2f}",
    fontsize=8,
    color="grey",
)
ax1.set_xlabel("time")
ax1.set_ylabel("$C(t)$")
ax1.set_title(f"dimer $\\theta$={THETA}, $N_{{phys}}$={N_PHYS}")
ax1.legend(fontsize=8, loc="upper left")

# --- panel 2: sample convergence on a regular orbit ------------------------
lat_reg = MeanFieldLattice(2, 1.0)  # below the pitchfork: bounded, regular
z_reg = np.array([np.sqrt(2.0 * 1.3), np.sqrt(2.0 * 0.7), 0.2, 0.2])
z_reg *= 2.0 / np.linalg.norm(z_reg)  # engine shell, non-stationary IC
conv_times = np.linspace(0.0, 1.5, 10)
scale = N_PHYS / LAT


def phys_occ(z: np.ndarray) -> float:
    return scale * (0.5 * (z[0] ** 2 + z[2] ** 2)) - 0.5


mf_traj = lat_reg.trajectory(z_reg, conv_times)
gp_n = scale * (0.5 * mf_traj[:, 0] ** 2 + 0.5 * mf_traj[:, 2] ** 2)
sim_reg = TruncatedWigner(lat_reg, z_reg, seed=5, hbar_eff=HBAR)
ref, _ = sim_reg.observable(384, conv_times, phys_occ)
ks = [8, 16, 32, 64, 128, 256]
devs = []
for k in ks:
    mean_k, _ = sim_reg.observable(k, conv_times, phys_occ)
    devs.append(float(np.abs(mean_k - ref).max()))
devs = np.array(devs)
rate_fit = float(np.polyfit(np.log(ks[1:]), np.log(devs[1:]), 1)[0])
ax2.loglog(ks, devs, "o-", label=r"$\max_t|\mathrm{mean}_K - \mathrm{mean}_{384}|$")
ax2.loglog(
    ks,
    devs[-1] * np.sqrt(ks[-1] / np.asarray(ks, dtype=float)),
    "--",
    lw=1.0,
    label="$1/\\sqrt{K}$ guide",
)
plateau = float(np.abs(ref - gp_n).max())
ax2.loglog(ks, [plateau] * len(ks), ":", color="grey", label="ensemble vs mean field")
ax2.set_xlabel("samples")
ax2.set_ylabel("deviation (occupation units)")
ax2.set_title(f"regular orbit, $\\hbar_{{eff}}$={HBAR}: MC slope {rate_fit:.2f}")
ax2.legend(fontsize=8)
print(
    f"sample-mean convergence: log-log slope {rate_fit:.3f} "
    f"(Monte-Carlo expectation -0.5); ensemble-vs-GP plateau {plateau:.4f}"
)
assert rate_fit < -0.35
assert plateau < 0.5  # O(hbar_eff) on the (N/L)-scaled occupation scale

fig.tight_layout()

# --- panel 3: geometric view, L = 4 ring -----------------------------------
lat4 = MeanFieldLattice(4, THETA)
z4 = np.array(
    [
        np.sqrt(2.0 * 1.2),
        np.sqrt(2.0 * 0.9),
        np.sqrt(2.0 * 1.0),
        np.sqrt(2.0 * 0.9),
        0.4,
        -0.3,
        0.2,
        -0.1,
    ]
)
z4 *= np.sqrt(2.0 * 4) / np.linalg.norm(z4)  # engine shell sum n_j = 4
times4 = np.linspace(0.0, 10.0, 41)
sim4 = TruncatedWigner(lat4, z4, seed=1234)  # engine shell: hbar_M = 1
trajs4 = sim4.propagate(64, times4)
occ_sites = np.arange(4)


def occupations(traj: np.ndarray) -> np.ndarray:
    return 0.5 * (traj[..., occ_sites] ** 2 + traj[..., occ_sites + 4] ** 2)


occ_mean = occupations(trajs4).mean(axis=0) - 0.5  # occupation estimator (b)
occ_mf = occupations(lat4.trajectory(z4, times4))  # GP occupations, no shift

# exact quantum reference: dense ED on the fixed-N ring, N_phys = L = 4
# (dim binom(7, 4) = 35), the dictionary-(b) partner of the engine shell
# state z4. The engine GP flow carries nonlinear coefficient sin(theta)
# on the shell, so the matching quantum onsite coupling is
# onsite_u = sin(theta) * L / (2 * N_phys) - the dimer block's
# sin(theta) / N_phys is its L = 2 special case, NOT the general rule.
# Hopping is -cos(theta) per directed bond (the dimer's -2c is the 2-site
# double-visit of its single bond, visited once here).
fock4 = FixedNFockSpace(4, 4)
t4 = np.zeros((4, 4))
for j4 in range(4):
    t4[j4, (j4 + 1) % 4] = -coup
    t4[(j4 + 1) % 4, j4] = -coup
H4 = Hamiltonian(fock4, one_body=t4, onsite_u=np.sin(THETA) / 2)  # sin(theta)*L/(2*N_phys)
phi4 = (z4[:4] + 1j * z4[4:]) / np.sqrt(2.0)  # physical GP amplitudes
psi4 = fixed_n_coherent_state(fock4, phi4 / np.linalg.norm(phi4))
obs4 = StateObservables(fock4)
ev4, evec4 = np.linalg.eigh(H4.matrix.toarray())
coef4 = evec4.conj().T @ psi4
n_ed4 = np.array(
    [np.real(obs4.density_profile(evec4 @ (np.exp(-1j * ev4 * tt) * coef4))) for tt in times4]
)
vmin3 = float(min(occ_mean.min(), occ_mf.min(), n_ed4.min()))
vmax3 = float(max(occ_mean.max(), occ_mf.max(), n_ed4.max()))
fig3, (ax3, ax4, ax5) = plt.subplots(1, 3, figsize=(16.5, 3.4), sharey=True)
occupation_heatmap(times4, occ_mean, ax=ax3, vmin=vmin3, vmax=vmax3)
ax3.set_title("TWA occupation estimator (64 samples)")
occupation_heatmap(times4, occ_mf, ax=ax4, vmin=vmin3, vmax=vmax3)
ax4.set_title("single mean-field orbit $|\\phi_j(t)|^2$")
occupation_heatmap(times4, n_ed4, ax=ax5, vmin=vmin3, vmax=vmax3)
ax5.set_title(r"exact quantum ED $\langle\hat{n}_j(t)\rangle$ (dim 35)")
fig3.tight_layout()
# All three rows are occupation numbers now: the -1/2 Weyl shift is kept
# only where it is physics (the TWA estimator absorbs the ensemble's
# vacuum half-quanta - its raw engine mean would sum to N + L/2 = 6), so
# every row sums to N = 4 and the panels are directly comparable.
diff = float(np.abs(occ_mean - occ_mf).max())
diff_q = float(np.abs(occ_mean - n_ed4).max())
diff_mq = float(np.abs(occ_mf - n_ed4).max())
print(
    f"L=4 ring (chaotic orbit): max|TWA mean - MF| = {diff:.3f}, "
    f"max|TWA mean - ED| = {diff_q:.3f}, max|MF - ED| = {diff_mq:.3f} "
    "occupation units"
)
assert np.abs(n_ed4.sum(axis=1) - 4.0).max() < 1e-9  # ED norm fixed-N
# Regime note (issue #30 review): the engine shell fixes N_phys = L, so
# this panel runs at hbar_eff = 1/4 and t = 10 spans ~7 Ehrenfest windows
# (t_E = ln 4 / lambda_s ~ 1.43): the absolute deviations are
# breakdown-sized, but the ORDERING is the semiclassical statement - the
# Wigner ensemble (max|TWA - ED| = 0.218, measured) sits closer to the
# exact quantum shell than a single GP orbit (max|MF - ED| = 0.451) even
# deep past the window. max|TWA - MF| = 0.360: the ensemble agrees with
# quantum better than with its own center trajectory - the vacuum noise,
# not the GP point, carries the agreement with ED.
plt.show()

out_dir = Path(__file__).resolve().parent


def render(tag: str) -> str:
    """Render the figure pair once; byte-identical output across builds."""
    main = out_dir / f"ex14_twa_semiclassical{tag}.png"
    grid = out_dir / f"ex14_twa_lattice{tag}.png"
    fig.savefig(main, dpi=80)
    fig3.savefig(grid, dpi=80)
    digest = hashlib.sha256()
    digest.update(main.read_bytes())
    digest.update(grid.read_bytes())
    main.unlink()
    grid.unlink()
    return digest.hexdigest()


first, second = render(""), render(".tmp")
assert first == second, "example figures are not byte-deterministic"
print(f"figure pair sha256 {first}")
