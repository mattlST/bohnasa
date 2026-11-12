"""Example 06: momentum-resolved spectra - who carries which eigenvalues.

Question: what do the individual symmetry blocks actually look like?

Compares block eigenvalues one-by-one against closed-form physics, not just
against the full space (example 05 did the reassembly check). On the
non-interacting ring, the exact levels are sums of single-particle energies
over bosonic mode occupations with fixed total momentum - the blocks of each
momentum class must reproduce that list exactly. Then interactions turn on:
the sector ladders shift, while the k and -k blocks stay identical, as
mirror conjugacy demands.
"""

from itertools import combinations_with_replacement

import matplotlib.pyplot as plt
import numpy as np

from bohnasa import FixedNFockSpace, Hopping, Interaction, SymmetrySectors

N, L = 4, 6
fock = FixedNFockSpace(N, L)
sectors = SymmetrySectors(fock, connections="pbc")

# single-particle ring dispersion eps_m = -2 cos(2 pi m / L); the free-boson
# N-particle levels are sums over mode multisets, classified by total momentum
eps = -2.0 * np.cos(2 * np.pi * np.arange(L) / L)
hopping = Hopping(fock, connections="pbc") * (-1.0)

free_blocks, free_analytic = {}, {}
print(f"ring N={N}, L={L}: dim {fock.dim}")
print("free bosons, block levels vs analytic multisets per momentum class:")
for k in range(L):
    labels = [lab for lab in sectors.labels if lab.k % L == k]
    free_blocks[k] = np.sort(
        np.concatenate(
            [np.linalg.eigvalsh(sectors.restrict(hopping, lab).toarray()) for lab in labels]
        )
    )
    free_analytic[k] = np.sort(
        [
            eps[list(modes)].sum()
            for modes in combinations_with_replacement(range(L), N)
            if sum(modes) % L == k
        ]
    )
    assert len(free_blocks[k]) == len(free_analytic[k])
    dev = np.abs(free_blocks[k] - free_analytic[k]).max()
    print(f"  k={k}: {len(free_blocks[k]):2d} levels, max deviation {dev:.1e}")
    assert dev < 1e-12

# interactions: same blocks, now the Hamiltonian has no closed form
hamiltonian = hopping + Interaction(fock) * 1.0
print("\ninteracting (t=1, U=1), lowest levels per sector:")
spectra = {}
for lab in sectors.labels:
    spectra[str(lab)] = np.linalg.eigvalsh(sectors.restrict(hamiltonian, lab).toarray())
    print(f"  {lab}: {np.round(spectra[str(lab)][:3], 3)}")

# mirror conjugacy prediction: the k and -k blocks are isospectral
for m in range(1, L // 2):
    assert np.allclose(spectra[f"(k={m})"], spectra[f"(k={-m})"], atol=1e-12)
print("k and -k blocks isospectral to machine precision, as mirror symmetry demands")

fig, (ax_free, ax_ladder) = plt.subplots(1, 2, figsize=(11.5, 4.5))
for k in range(L):
    ax_free.plot(
        [k] * len(free_blocks[k]),
        free_blocks[k],
        "o",
        ms=5,
        alpha=0.55,
        label="sector blocks" if k == 0 else None,
    )
    ax_free.plot(
        [k + 0.25] * len(free_analytic[k]),
        free_analytic[k],
        "|",
        ms=12,
        color="crimson",
        label="analytic" if k == 0 else None,
    )
ax_free.set_xticks(range(L))
ax_free.set_xlabel("total momentum k")
ax_free.set_ylabel("energy")
ax_free.legend(loc="upper left")
ax_free.set_title("free bosons: blocks vs closed form")

for i, lab in enumerate(sectors.labels):
    ax_ladder.plot([i] * len(spectra[str(lab)]), spectra[str(lab)], ".", ms=6)
ax_ladder.set_xticks(
    range(len(sectors.labels)), [str(lab) for lab in sectors.labels], rotation=45, ha="right"
)
ax_ladder.axhline(min(v[0] for v in spectra.values()), color="grey", lw=0.8, ls="--")
ax_ladder.set_ylabel("energy")
ax_ladder.set_title(r"interacting ladders ($t=1$, $U=1$)")
fig.tight_layout()
plt.show()
