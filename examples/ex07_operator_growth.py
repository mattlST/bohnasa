"""Example 07: operator growth - how a local operator spreads in Liouville space.

Question: how fast does an operator grow under the Liouvillian L̂ = [H, ·]?

The operator-growth sibling of example 03's state-based curve. A local
density n̂₀ on the N=6, L=4 ring seeds the Krylov chain of Liouvillian(H);
we track <K̂>(t) for the free ring (U=0) and the strongly interacting one
(U=4). Same first Krylov step (the interaction commutes with n̂₀), then two
fates: the free operator grows only inside a five-level closed Krylov space
- it oscillates, but its <K̂> is bounded by the closure - while the
interacting one spreads over levels three times higher before the chain cap.
"""

import matplotlib.pyplot as plt
import numpy as np

from bohnasa import (
    FixedNFockSpace,
    Hopping,
    Interaction,
    Liouvillian,
    Number,
    operator_growth,
)

N, L = 6, 4
fock = FixedNFockSpace(N, L)
num_vec = 60
times = np.linspace(0.0, 4.0, 200)
seed = Number(fock, 0)

h_free = Hopping(fock, connections="pbc") * (-1.0)
h_int = h_free + Interaction(fock) * 4.0
k_free, t_free = operator_growth(Liouvillian(h_free), seed, times, num_vec)
k_int, t_int = operator_growth(Liouvillian(h_int), seed, times, num_vec)

beta1_free, beta1_int = abs(t_free[1, 0]), abs(t_int[1, 0])
print(f"Liouville space: {fock.dim**2} dimensions, chain capped at {num_vec} levels")
print(f"first beta, free: {beta1_free:.6f} | interacting: {beta1_int:.6f}")
assert np.isclose(beta1_free, beta1_int, rtol=1e-9)  # [Interaction, n_0] = 0

assert k_free[0] < 1e-12 and k_int[0] < 1e-12  # traceless seed starts at level 0
print(f"free: Krylov space closes after {t_free.shape[0]} levels, <K> max {k_free.max():.2f}")
print(f"interacting: no breakdown, chain fills to {t_int.shape[0]}, <K> max {k_int.max():.2f}")
assert t_free.shape[0] < num_vec
assert k_int.max() > 3 * k_free.max()
# in the closed free system the operator can never visit levels beyond the
# Krylov subspace itself - an oscillation bounded by construction
assert k_free.max() <= t_free.shape[0] - 1 + 1e-9

fig, (ax_growth, ax_rate) = plt.subplots(1, 2, figsize=(11, 4))
ax_growth.plot(times, k_free, label="free ring (U = 0)")
ax_growth.plot(times, k_int, label="interacting (U = 4)")
ax_growth.axhline(num_vec - 1, color="grey", ls="--", lw=0.8, label="chain cap")
ax_growth.set_xlabel("time")
ax_growth.set_ylabel(r"$\langle \hat{K} \rangle(t)$")
ax_growth.legend()
ax_growth.set_title("operator growth of $\hat{n}_0$")

ax_rate.plot(times, np.gradient(k_free, times))
ax_rate.plot(times, np.gradient(k_int, times))
ax_rate.set_xlabel("time")
ax_rate.set_ylabel(r"$\mathrm{d}\langle \hat{K}\rangle/\mathrm{d}t$")
ax_rate.set_title("growth rate: bounded oscillation vs wide excursions")
fig.tight_layout()
plt.show()

print(f"free oscillation respects its closure bound K <= {t_free.shape[0] - 1}")
late_rate = np.gradient(k_int, times)[times > 3]
print(f"interacting rate at late times spans [{late_rate.min():.2f}, {late_rate.max():.2f}]")
