"""Example 04: attractive bosons on a disordered chain (transmon-boson model).

Question: where does an eigenstate live in Fock space?

Descendant of t05_transmon_boson.ipynb (same physics, smaller system so it
runs in CI): H = 0.005 T - 0.25 U + sum mu_i n_i with random on-site
potentials. Negative attraction plus disorder pushes eigenstates towards
localized patterns in the occupation basis; the inverse participation ratio
(IPR) over the spectrum exposes this.
"""

import matplotlib.pyplot as plt
import numpy as np

from bohnasa import ChemicalPotential, FixedNFockSpace, Hopping, Interaction

N, L = 4, 8
fock = FixedNFockSpace(N, L)
print(f"dim of the fixed-N sector: {fock.dim}")

rng = np.random.default_rng(42)
H = (
    Hopping(fock) * 0.005
    + Interaction(fock) * -0.25
    + ChemicalPotential(fock, 0.125 * rng.random(L))
)

energy, states = np.linalg.eigh(H.matrix.toarray())

# IPR_n = sum_k |<k|psi_n>|^4 in the Fock basis: 1/dim fully ergodic,
# O(1) for a state pinned to a few occupation patterns
ipr = np.sum(np.abs(states) ** 4, axis=0)
top = states[:, -1]
peak = fock.fockArray[int(np.argmax(np.abs(top)))]
print(
    f"top eigenstate E = {energy[-1]:.4f}, IPR = {ipr[-1]:.3f}, "
    f"dominant occupation pattern: {tuple(int(x) for x in peak)}"
)
print(f"spectral IPR: min {ipr.min():.4f}, mean {ipr.mean():.4f}, max {ipr.max():.4f}")

assert np.all(ipr > 0.0) and np.all(ipr <= 1.0 + 1e-12)
assert np.isclose(np.sum(energy.imag), 0.0)  # hermitian input
assert ipr.max() > 1.0 / fock.dim * 5, "expected at least one localized eigenstate"

fig, (ax_spec, ax_ipr) = plt.subplots(1, 2, figsize=(11, 4))
ax_spec.plot(energy, ".", ms=3)
ax_spec.set_xlabel("eigenvalue index")
ax_spec.set_ylabel("energy")
ax_spec.set_title("spectrum")
ax_ipr.hist(ipr, bins=25)
ax_ipr.set_xlabel("IPR")
ax_ipr.set_title("participation of eigenstates in Fock space")
ax_ipr.axvline(1.0 / fock.dim, ls="--", label=r"ergodic value $1/d$")
ax_ipr.legend()
fig.tight_layout()
plt.show()
