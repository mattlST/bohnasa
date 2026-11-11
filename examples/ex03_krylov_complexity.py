"""Example 03: Krylov complexity after a staggered coherent quench.

Question: how fast does a state spread through Krylov space?

Descendant of t04_gram.ipynb: a projected coherent state with staggered
amplitudes phi = i*(-1)**j quenches under H = -T + 4U; we track the mean
Krylov level <K>(t) and its logarithmic growth rate. The orthogonality
part of that notebook lives on as a permanent physics test.
"""

import matplotlib.pyplot as plt
import numpy as np

from bohnasa import (
    FixedNFockSpace,
    Hopping,
    Interaction,
    Number,
    fixed_n_coherent_state,
    krylov_complexity,
    time_evolution_exact_ind,
)

N, L = 30, 2
fock = FixedNFockSpace(N, L)
phi = 1j * np.array([(-1) ** j for j in range(L)], dtype=complex)
sta = fixed_n_coherent_state(fock, phi)
print(f"dim: {fock.dim}, coherent-state norm: {np.linalg.norm(sta):.6f}")

H = Hopping(fock) * -1.0 + Interaction(fock) * 4.0
K_op = krylov_complexity(fock, H, sta, fock.dim - 1)

time = np.linspace(0.0, 4.0, 200)
res = time_evolution_exact_ind(H, sta, time, [K_op, Number(fock, 0)])
krylov = res[0, 0, :].real
n0 = res[1, 0, :].real

# the chain is 0-based: the seed state itself sits entirely on level 0
assert krylov[0] < 1e-10
assert krylov.max() > 5.0, "expected the quench to climb the Krylov chain"

growth_rate = np.gradient(np.log(krylov[1:]), time[1:])

fig, (ax_k, ax_rate) = plt.subplots(1, 2, figsize=(11, 4))
ax_k.semilogy(time[1:], krylov[1:], label=r"$\langle \hat K \rangle(t)$")
ax_k.semilogy(time[1:], n0[1:], "--", label=r"$\langle n_0 \rangle(t)$")
ax_k.set_xlabel("time")
ax_k.legend()
ax_k.set_title("Krylov spread after staggered quench")
ax_rate.plot(time[1:], growth_rate)
ax_rate.set_xlabel("time")
ax_rate.set_title(r"growth rate $\mathrm{d}\log\langle \hat K\rangle/\mathrm{d}t$")
fig.tight_layout()
plt.show()

print(f"<K> saturates at {krylov.max():.2f} of possible {fock.dim - 1}")
