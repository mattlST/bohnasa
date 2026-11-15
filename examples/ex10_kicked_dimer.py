"""Example 10: the kicked dimer - breaking integrability by Trotterized driving.

Question: what happens to operator growth when the continuous dimer is
driven periodically instead?

Floquet form of the dimer with fixed coupling (theta = 1.35): one period is
a kick of the full on-site interaction followed by a hop, both of duration
tau, so U_F = exp(-i tau H_hop) exp(-i tau H_int) with the kick PERIOD as
control parameter. This example reproduces the schematic and parameter
set of Figure 6.4 of the author's doctoral thesis (see CITATION.cff) -
tau = 0.05 ... 2.0, time to 100, continuous reference to t = 20 where the
Trotterizations still coincide - at N = 1000 instead of the figure's
N = 10^4.

The curve is evaluated with a batched vector trick: forward chains F_n =
U_F^n s are grown column by column and all backward chains U_F^†n (O F_n)
are propagated simultaneously as shrinking-width matrix products - turning
the naive O(n^2) chain of matvecs into a few gemms. The helper is
cross-checked against SquaredCommutator.from_propagator on a small system,
so speed here cannot cost correctness.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla

from bohnasa import (
    FixedNFockSpace,
    Hamiltonian,
    Number,
    SquaredCommutator,
    dimer_lambda,
    fixed_n_coherent_state,
)

THETA = 1.35
J = np.cos(THETA)


def kicked_case(N):
    fock = FixedNFockSpace(N, 2)
    h_hop = Hamiltonian(fock, one_body=np.array([[0.0, -2 * J], [-2 * J, 0.0]]))
    h_int = Hamiltonian(fock, onsite_u=np.sin(THETA) / N)
    op = (Number(fock, 0) - Number(fock, 1)) * 0.5
    psi = fixed_n_coherent_state(fock, np.array([1.0, -1.0]) / np.sqrt(2.0))
    return fock, h_hop, h_int, op, psi


def kicked_curve_batched(N, tau, n_max):
    """C(n) = ||[O(n), O]|psi>||^2 for the kicked dimer, batched in the chains."""
    fock, h_hop, h_int, op, psi = kicked_case(N)
    eps, V = sla.eigh_tridiagonal(
        np.asarray(h_hop.matrix.diagonal(), dtype=float),
        np.asarray(h_hop.matrix.diagonal(1)),
        check_finite=False,
    )
    hop_phase = np.exp(-1j * tau * eps)
    kick = np.exp(-1j * tau * np.asarray(h_int.matrix.diagonal(), dtype=float))
    od = (fock.fockArray[:, 0] - fock.fockArray[:, 1]) / 2.0

    def hop_apply(X, forward):
        p = hop_phase if forward else hop_phase.conj()
        return V @ (p[:, None] * (V.T @ X))

    curve = {}
    for name, seed in (("first", od * psi), ("second", psi)):
        F = np.empty((psi.size, n_max + 1), dtype=complex)
        F[:, 0] = seed
        for n in range(1, n_max + 1):  # F[:, n] = U_F^n |seed>, U_F = H K (kick first)
            F[:, n] = hop_apply((F[:, n - 1] * kick)[:, None], True)[:, 0]
        B = od[:, None] * F  # column n starts the backward chain from O F_n
        for j in range(1, n_max + 1):  # all chains U_F^† = K^† H^†, one gemm per round
            B[:, j:] = hop_apply(B[:, j:], False) * kick.conj()[:, None]
        curve[name] = B
    first = curve["first"]  # U_F^†n O U_F^n (O psi)
    second = od[:, None] * curve["second"]  # O U_F^†n O U_F^n psi
    return (np.abs(first - second) ** 2).sum(axis=0)


# correctness guard: batched == library pointwise path (small system)
Nsmall, tausmall, nsmall = 40, 0.4, 12
fock, h_hop, h_int, op, psi = kicked_case(Nsmall)
U_small = sla.expm(-1j * tausmall * h_hop.matrix.toarray()) @ sla.expm(
    -1j * tausmall * h_int.matrix.toarray()
)
lib = SquaredCommutator.from_propagator(U_small, op).curve(psi, np.arange(nsmall + 1))
bat = kicked_curve_batched(Nsmall, tausmall, nsmall)
rel = np.abs(bat[1:] / lib[1:] - 1.0).max()
assert rel < 1e-9, rel
print(f"batched chains reproduce the library path to {100 * rel:.2e}% relative")

lam = dimer_lambda(THETA)
fig, (ax_c, ax_dev) = plt.subplots(1, 2, figsize=(11.5, 4.2))

T_END, T_CONT = 100.0, 20.0
N = 1000
_, h_hop, h_int, op, psi = kicked_case(N)
tg = np.linspace(0.0, T_CONT, 801)
c_cont = SquaredCommutator(h_hop + h_int, op).curve(psi, tg)
ax_c.semilogy(tg, c_cont, color="black", lw=1.2, label="continuous (t < 20)")

y_max = c_cont.max()
for tau in (0.05, 0.5, 1.0, 1.5, 2.0):
    t_stop = T_CONT if tau == 0.05 else T_END
    n_max = int(t_stop / tau)
    c = kicked_curve_batched(N, tau, n_max)
    times = np.arange(n_max + 1) * tau
    ax_c.semilogy(
        times, np.maximum(c, 1e-300), "-o" if tau > 0.05 else "-", ms=3, label=f"tau = {tau}"
    )
    y_max = max(y_max, c.max())
    if tau == 0.05:
        dev = np.abs(c[1:] / np.interp(times[1:], tg, c_cont) - 1.0)
        assert dev.max() < 0.1, dev.max()
        ax_dev.semilogy(times[1:], np.maximum(dev, 1e-16), label="tau = 0.05 vs continuous")
        print(f"tau = 0.05 tracks the continuous curve, max deviation {dev.max():.3f}")
    else:
        print(f"tau = {tau:4.1f}: C(end) {c[-1]:.3e}, Cmax {c.max():.3e}")

ax_dev.set_xlabel("time $t = n\\tau$")
ax_dev.set_ylabel("relative deviation")
ax_dev.set_title("the smallest period is the continuous dimer")
ax_dev.legend(fontsize=8)
ax_c.set_ylim(1e-5, 10 * y_max)
ax_c.set_xlabel("time $t = n\\tau$")
ax_c.set_ylabel(r"$C(t)$")
ax_c.set_title(f"kicked dimer OTOC, N = {N}")
ax_c.legend(fontsize=8)
fig.tight_layout()
plt.show()
