"""Example 11: the general L-site mean-field lattice engine.

Question: does the closed-form dimer survive as the L = 2 corner of the
general discrete-nonlinear-Schrodinger (mean-field Bose-Hubbard) engine,
and what do chaos and fixed-point structure look like once L > 2?

Part one is the reduction check: on the L = 2 ring the engine's flow,
fixed points and rotating-frame stability spectrum must reproduce
``dimer_fixed_points`` / ``dimer_lambda`` - the exact reference line of
the OTOC module (issue #22). Part two moves to L = 4: the branch search
counts mean-field fixed points by energy, and the maximal Lyapunov
exponent of a generic orbit is computed by three INDEPENDENT algorithms
(Benettin's tangent-flow renormalization, the QR method and a literal
two-trajectory separation) - agreement between them is the numerical
evidence, the dimer saddle exponent the analytic one. Conventions are
stated once in ``bohnasa/mean_field_lattice.py`` (thesis Sec. 10.1,
App. A.1).
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as si

from bohnasa import MeanFieldLattice
from bohnasa import classical_mean_field as mf

THETA = 1.35  # above the dimer pitchfork tan(theta) = 2


def embed(s: float, phi: float) -> np.ndarray:
    """Dimer (s, phi) of the classical_mean_field module into lattice
    coordinates z = (q0, q1, p0, p1) at L = 2 (occupation shell = 2)."""
    n0, n1 = 1.0 + s, 1.0 - s
    q = np.sqrt(2.0) * np.array([np.sqrt(n0) * np.cos(-phi / 2), np.sqrt(n1) * np.cos(phi / 2)])
    p = np.sqrt(2.0) * np.array([np.sqrt(n0) * np.sin(-phi / 2), np.sqrt(n1) * np.sin(phi / 2)])
    return np.concatenate((q, p))


def project(z: np.ndarray) -> tuple[float, float]:
    """Inverse map: s = (n0 - n1)/2, phi = arg(phi1) - arg(phi0)."""
    n = 0.5 * (z[:2] ** 2 + z[2:] ** 2)
    s = float((n[0] - n[1]) / 2.0)
    phi = float(np.angle(z[1] + 1j * z[3]) - np.angle(z[0] + 1j * z[2]))
    return s, phi


# ---------- L = 2: the engine must be the dimer ----------
lat = MeanFieldLattice(2, THETA)
pts = lat.fixed_points()
table = mf.dimer_fixed_points(THETA)


def key(s: float, p: float) -> tuple[float, float]:
    return (round(abs(s), 7), round(abs(p), 7))


got = sorted(key(*project(z)) for z in pts)
ref = sorted(key(float(row[0]), float(row[1])) for row in table)
assert got == ref, (got, ref)
print(f"L=2 reduction: {len(pts)} fixed points, positions agree with the dimer table")

saddle = next(z for z in pts if abs(abs(project(z)[1]) - np.pi) < 1e-6)
eig_lattice = max(np.linalg.eigvals(lat.stability_matrix(saddle)).real)
print(f"  saddle exponent {eig_lattice:.10f} vs dimer_lambda {mf.dimer_lambda(THETA):.10f}")
assert abs(eig_lattice - mf.dimer_lambda(THETA)) < 1e-7

# orbit-level check against a tight-tolerance integration of the dimer
# sphere equations (the closed forms of the classical_mean_field docstring)
c, u = np.cos(THETA), np.sin(THETA)


def dimer_rhs(_t: float, y: np.ndarray) -> list[float]:
    s, phi = y
    root = np.sqrt(max(1.0 - s * s, 1e-300))
    return [-4 * c * root * np.sin(phi), 4 * c * s * np.cos(phi) / root + 2 * u * s]


times2 = np.linspace(0.0, 8.0, 801)
y0 = (0.3, 2.0)
ref2 = si.solve_ivp(dimer_rhs, (0.0, 8.0), y0, t_eval=times2, rtol=1e-12, atol=1e-14).y
traj2 = lat.trajectory(embed(*y0), times2)
proj2 = np.array([project(z) for z in traj2])
# phi is an angle: compare unwrapped phases (the orbit crosses the +-pi wrap)
err2 = float(
    np.abs(proj2[:, 0] - ref2[0]).max() + np.abs(np.unwrap(proj2[:, 1]) - np.unwrap(ref2[1])).max()
)
print(f"  orbit reduction: max |Delta(s, phi)| = {err2:.1e} over t in [0, 8]")
assert err2 < 1e-8

# ---------- L = 4: branch search and Lyapunov cross-check ----------
lat4 = MeanFieldLattice(4, THETA)
pts4 = lat4.fixed_points()
energies = [lat4.hamiltonian(z) for z in pts4]
print(
    f"L=4 PBC ring: {len(pts4)} fixed points (search, not certified complete), "
    f"energies {energies[0]:.3f} ... {energies[-1]:.3f}"
)

rng = np.random.default_rng(0)
z0 = rng.normal(size=8)
z0 *= np.sqrt(8.0) / np.linalg.norm(z0)  # particle number = L shell
times = np.linspace(0.0, 40.0, 401)
lam_b = lat4.lyapunov(z0, times)
lam_q = lat4.lyapunov(z0, times, method="qr")
lam_t = lat4.lyapunov(z0, times, method="two_trajectory")
print(
    f"maximal Lyapunov exponent: Benettin {lam_b:.6f}, QR {lam_q:.6f}, two-trajectory {lam_t:.6f}"
)
# chaotic finite-time estimates: Benettin vs two-trajectory agree within the
# finite-time + threading spread (not a one-stack 1% band); QR is looser and
# tracked separately. See issue #26.
assert lam_b > 0.1 and abs(lam_b - lam_t) / lam_b < 0.05

traj4 = lat4.trajectory(z0, times)
occ = 0.5 * traj4[:, :4] ** 2 + 0.5 * traj4[:, 4:] ** 2
drift_e = float(np.ptp([lat4.hamiltonian(z) for z in traj4]))
drift_n = float(np.ptp([lat4.particle_number(z) for z in traj4]))
print(f"conservation over T = 40: ptp(H) = {drift_e:.1e}, ptp(N) = {drift_n:.1e}")
assert drift_e < 1e-6 and drift_n < 1e-6

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.0))
for j in range(4):
    ax1.plot(times, occ[:, j], lw=1.0, label=f"$n_{j}(t)$")
ax1.set_xlabel("time")
ax1.set_ylabel("occupation")
ax1.set_title(f"L=4 DNLS orbit, theta = {THETA}, lambda ~ {lam_b:.3f}")
ax1.legend(fontsize=8, ncol=2)
s_orb = proj2[:, 0]
p_orb = proj2[:, 1] % (2.0 * np.pi)  # plot phi in [0, 2pi): the orbit swings around pi
ax2.plot(p_orb, s_orb, lw=1.0, color="C1")
for row in table:
    ax2.plot(row[1], row[0], "kx" if row[2] > 0 else "ko", ms=8)
ax2.set_xlabel("$\\phi$")
ax2.set_ylabel("$s$")
ax2.set_title("L=2 orbit on the dimer sphere (x: saddle, o: center)")
fig.tight_layout()
plt.show()
