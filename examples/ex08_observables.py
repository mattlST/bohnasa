"""Example 08: what does the state actually look like? Observables.

Question: how do interactions show up in measurable single- and two-body
quantities of the ring ground state?

The N=3, L=6 free ring is an exact plane-wave condensate: perfect coherence
(gamma = N/L everywhere), all particles at k=0, pair bunching (N-1)/N, and a
structure factor pinned between S(0) = 0 (sharp total number) and S(k) = 1
(shot noise) for k != 0. Turning on the on-site U depletes the condensate
and suppresses density fluctuations - the whole story lives on the analytic
anchors computed from the observables module.

The U sweep below evaluates three observables on six sweep states plus the
free ground state, so it binds the basis once with
:class:`~bohnasa.StateObservables` (issue #20): the pair-operator stack is
built a single time instead of per call, while the module-level functions
stay the reference path - the two agree bit-for-bit, asserted and printed.
"""

import matplotlib.pyplot as plt
import numpy as np

from bohnasa import (
    FixedNFockSpace,
    Hamiltonian,
    StateObservables,
    density_profile,
    momentum_distribution,
    orbital_occupations,
    pair_correlation,
    structure_factor,
)

N, L = 3, 6
fock = FixedNFockSpace(N, L)
t = np.zeros((L, L))
for site in range(L):
    nxt = (site + 1) % L
    t[nxt, site] = t[site, nxt] = -1.0
h_ring = Hamiltonian(fock, one_body=t)
obs = StateObservables(fock)  # bind once: every obs.* call reuses the stack


def ground_state(u):
    matrix = (h_ring + Hamiltonian(fock, onsite_u=u)).matrix.toarray()
    return np.linalg.eigh(matrix)[1][:, 0]


# --- the free limit: everything exact at once ---
gs = ground_state(0.0)
gamma = orbital_occupations(gs, fock)
n_k = momentum_distribution(gs, fock)
s_k = structure_factor(gs, fock)
g2 = pair_correlation(gs, fock)

print(f"occupations: {np.round(gamma, 10)} - perfect condensate nu_0 = {N}")
print(f"momentum n_k: {np.round(n_k, 10)} - all weight at k = 0")
print(f"S(k): {np.round(s_k, 10)} - pinned: 0 at k=0, shot noise 1 elsewhere")
assert abs(gamma[0] - N) < 1e-10 and abs(gamma[1]) < 1e-10
assert abs(n_k[0] - N) < 1e-10 and np.abs(n_k[1:]).max() < 1e-10
assert abs(s_k[0]) < 1e-10 and np.abs(s_k[1:] - 1.0).max() < 1e-10
bunching = g2[0, 1] / (density_profile(gs, fock)[0] ** 2)
print(f"bosonic bunching G2_ij/<n_i><n_j> = {bunching:.6f} = (N-1)/N = {(N - 1) / N:.6f}")
assert abs(bunching - (N - 1) / N) < 1e-10

# bound path vs module reference path: bit-for-bit identical (issue #20
# acceptance - the fast stack reproduces the rebuilt-operator values exactly)
parity = max(
    np.abs(obs.orbital_occupations(gs) - orbital_occupations(gs, fock)).max(),
    np.abs(obs.momentum_distribution(gs) - momentum_distribution(gs, fock)).max(),
    np.abs(obs.structure_factor(gs) - structure_factor(gs, fock)).max(),
    np.abs(obs.pair_correlation(gs) - pair_correlation(gs, fock)).max(),
    np.abs(obs.density_profile(gs) - density_profile(gs, fock)).max(),
)
print(f"StateObservables vs module functions: bit-for-bit, max abs diff {parity:.1e}")
assert parity == 0.0

# --- turn on U: depletion and fluctuation suppression ---
u_values = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
nu0_over_n = np.array([obs.orbital_occupations(ground_state(u))[0] / N for u in u_values])
s_k_by_u = np.array([obs.structure_factor(ground_state(u)) for u in u_values])
n_k_by_u = np.array([obs.momentum_distribution(ground_state(u)) for u in u_values])

for u, depl, s_one in zip(u_values, nu0_over_n, s_k_by_u[:, 1], strict=True):
    print(f"U = {u:4.1f}: condensate fraction {depl:.4f}, S(k=2pi/L) = {s_one:.4f}")
print(f"U = 8: depleted n_k = {np.round(n_k_by_u[-1], 4)}")

assert np.all(np.diff(nu0_over_n) < 0)
assert np.all(np.diff(s_k_by_u[:, 1], axis=0) < 0)

# --- plots ---
fig, (ax_occ, ax_s, ax_n) = plt.subplots(1, 3, figsize=(13, 4))
k = np.arange(L)
ax_occ.bar(k - 0.04, obs.orbital_occupations(gs), width=0.08, label="U = 0")
ax_occ.bar(k + 0.04, obs.orbital_occupations(ground_state(8.0)), width=0.08, label="U = 8")
ax_occ.set_xlabel("ring momentum mode")
ax_occ.set_ylabel(r"natural occupation $\nu_k$")
ax_occ.set_title("occupations: condensate depletes")
ax_occ.legend()

ax_s.plot(k, s_k_by_u[0], "o-", label="U = 0")
ax_s.plot(k, s_k_by_u[-1], "s-", label="U = 8")
ax_s.axhline(1.0, color="grey", ls="--", lw=0.8, label="shot noise")
ax_s.set_xlabel(r"$k = 2\pi n/L$")
ax_s.set_ylabel(r"$S(k)$")
ax_s.set_title("density fluctuations suppressed")
ax_s.legend()

ax_n.plot(k, n_k_by_u[0], "o-", label="U = 0")
ax_n.plot(k, n_k_by_u[-1], "s-", label="U = 8")
ax_n.set_xlabel(r"$k = 2\pi n/L$")
ax_n.set_ylabel(r"$n_k$")
ax_n.set_title("momentum distribution spreads")
ax_n.legend()
fig.tight_layout()
plt.show()
