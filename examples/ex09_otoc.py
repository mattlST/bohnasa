"""Example 09: operator scrambling - the squared commutator and its Ehrenfest window.

Question: how fast does an operator stop commuting with its own time
evolution, and how long does that exponential last?

The N-boson dimer with quarter-circle coupling J = cos(theta),
U = 2 sin(theta)/N: classically the (0, pi) saddle is unstable with exact
growth rate lambda_s; quantum mechanically the squared commutator C(t)
grows at rate 2*lambda_s in the early window and saturates on a time scale
ln N / lambda_s - the Ehrenfest time. Since the dimer is only N+1
dimensional, the sweep reaches the largest systems for free.
"""

import matplotlib.pyplot as plt
import numpy as np

from bohnasa import (
    FixedNFockSpace,
    Hamiltonian,
    Number,
    SquaredCommutator,
    dimer_fixed_points,
    dimer_lambda,
    fixed_n_coherent_state,
)

THETA = 1.35  # unstable-saddle regime, theta* = 1.339 nearby
lam = dimer_lambda(THETA)
print(f"classical saddle exponent lambda_s({THETA}) = {lam:.6f} (closed form)")
print(f"fixed points at theta: s = {np.round(dimer_fixed_points(THETA)[:, 0], 4)}")


def dimer_case(N):
    fock = FixedNFockSpace(N, 2)
    t = np.array([[0.0, -2 * np.cos(THETA)], [-2 * np.cos(THETA), 0.0]])
    H = Hamiltonian(fock, one_body=t, onsite_u=np.sin(THETA) / N)
    op = (Number(fock, 0) - Number(fock, 1)) * 0.5
    psi = fixed_n_coherent_state(fock, np.array([1.0, -1.0]) / np.sqrt(2.0))
    return fock, H, op, psi


def beta2(H, op, psi):
    ho = (H.matrix @ op.matrix - op.matrix @ H.matrix).toarray()
    v = ho @ (op.matrix @ psi) - op.matrix @ (ho @ psi)
    return np.vdot(v, v).real


# --- quantum growth across N ---
fig, (ax_curve, ax_rate) = plt.subplots(1, 2, figsize=(11, 4.2))
ns = np.linspace(0.002, 30.0, 900)
t_stars = {}
ratio_by_n = {}
rate_by_n = {}
for N in np.exp2(np.arange(6, 13)).astype(int):
    fock, H, op, psi = dimer_case(N)
    sc = SquaredCommutator(H, op)
    c = sc.curve(psi, ns)
    b2 = beta2(H, op, psi)
    assert abs(c[1] / ns[1] ** 2 - b2) / b2 < 1e-2  # quadratic onset, per system
    ratio_by_n[N] = c / b2
    ax_curve.semilogy(ns, ratio_by_n[N], label=f"N = {N}")
    rate = np.gradient(np.log(c / b2), ns)  # instantaneous effective rate
    rate_by_n[N] = rate
    above = rate > lam
    i = np.searchsorted(ns, 2.0)
    while i + 1 < len(ns) and above[i + 1]:
        i += 1
    t_stars[N] = ns[i]
# slope guides anchored to the N = 128 curve at t = 2, where the exponential
# regime sets in (a line through the origin would miss the quadratic onset)
t_ref = 2.0
v_ref = np.interp(t_ref, ns, ratio_by_n[128])
for rate_name, rate_val in ((r"$2\lambda_s$ guide", 2 * lam), (r"$\lambda_s$ guide", lam)):
    ax_curve.plot(
        ns,
        v_ref * np.exp(rate_val * (ns - t_ref)),
        ls="--",
        lw=0.9,
        color="grey",
        label=rate_name,
    )
ax_curve.set_ylim(1e-1, 1e7)
ax_curve.set_xlabel("time")
ax_curve.set_ylabel(r"$C(t)/\beta^2$")
ax_curve.set_title("squared commutator, dimer")
ax_curve.legend(fontsize=8)

for N, t_star in t_stars.items():
    print(
        f"N = {N:4d}: growth above lambda_s until t = {t_star:5.2f} "
        f"= {t_star * lam / np.log(N):.2f} ln N / lambda_s"
    )
ratios = np.array([t * lam / np.log(n) for n, t in t_stars.items()])
assert np.all(np.abs(ratios - 0.81) < 0.06)

# --- effective rate d log(C/beta^2)/dt: approaches 2 lambda_s from above
# as N grows, crosses lambda_s at the t* table above, decays to saturation ---
view = ns > 0.5
for N in ratio_by_n:
    ax_rate.plot(ns[view], rate_by_n[N][view], label=f"N = {N}")
ax_rate.axhline(2 * lam, color="black", ls="--", lw=0.9, label=r"$2\lambda_s$")
ax_rate.axhline(lam, color="grey", ls="--", lw=0.9, label=r"$\lambda_s$")
ax_rate.set_xlim(0.5, 12.0)
ax_rate.set_ylim(-0.2, 3.0)
ax_rate.set_xlabel("time")
ax_rate.set_ylabel(r"$\mathrm{d} \log (C/\beta^2) / \mathrm{d}t$")
ax_rate.set_title("effective rate")
ax_rate.legend(fontsize=8)
fig.tight_layout()
plt.show()
