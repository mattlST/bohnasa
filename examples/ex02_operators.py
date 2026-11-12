"""Example 02: operators, their structure, and a ground state.

Question: how do the operator classes assemble a many-body Hamiltonian and
what does its ground state look like?

Builds the chain Hamiltonian H = -t T + U sum n(n-1) by hand and from the
generic Hamiltonian builder (and shows they agree exactly; the hopping
adjacency plot is the direct descendant of t02_operator.ipynb), finds the
ground state with Lanczos, and compares against exact diagonalization.
"""

import matplotlib.pyplot as plt
import numpy as np

from bohnasa import FixedNFockSpace, Hamiltonian, Hopping, Interaction, Number, lanczos

N, L = 6, 4
t, U = 1.0, 0.5
fock = FixedNFockSpace(N, L)
print(f"dim of fixed-N sector: {fock.dim}")

hopping = Hopping(fock, connections="obc")
H = hopping * -t + Interaction(fock) * U

# the same Hamiltonian from the generic builder: coefficient matrix in,
# operator out (open-chain adjacency with amplitude -t, on-site interaction U)
adjacency = np.zeros((L, L))
for site in range(L - 1):
    adjacency[site, site + 1] = adjacency[site + 1, site] = -t
H_built = Hamiltonian(fock, one_body=adjacency, onsite_u=U)
assert np.allclose(H_built.matrix.toarray(), H.matrix.toarray())
print("builder reproduces the hand-assembled -t T + U n(n-1) exactly")

# --- structure: what the hopping matrix looks like in this basis ---
fig, (ax_adj, ax_gs) = plt.subplots(1, 2, figsize=(11, 4.5))
im = ax_adj.imshow(hopping.matrix.toarray(), aspect="auto", cmap="viridis")
ax_adj.set_title(r"hopping matrix $c_j^\dagger c_i + \mathrm{h.c.}$")
fig.colorbar(im, ax=ax_adj)

# --- ground state via Lanczos from a random Krylov seed ---
rng = np.random.default_rng(42)
seed = rng.normal(size=fock.dim) + 1j * rng.normal(size=fock.dim)
seed /= np.linalg.norm(seed)
v, T_m = lanczos(seed, H, min(40, fock.dim - 1))
m = int(np.count_nonzero(np.linalg.norm(v, axis=1) > 1e-12))
krylov_energy = np.linalg.eigvalsh(T_m[:m, :m].real)[0]
gs = np.linalg.eigh(T_m[:m, :m])[1][:, 0] @ v[:m]

exact_energy = np.linalg.eigvalsh(H.matrix.toarray())[0]
print(f"ground state energy, Lanczos ({m} Krylov vectors): {krylov_energy:.10f}")
print(f"ground state energy, exact:                        {exact_energy:.10f}")
assert np.isclose(krylov_energy, exact_energy, atol=1e-8)

density = np.array([Number(fock, i).eval_mean(gs).real for i in range(L)])
ax_gs.plot(np.arange(L), density, "o-", label=r"$\langle n_i \rangle$")
ax_gs.set_xlabel("site")
ax_gs.set_title(f"ground state density (E0 = {exact_energy:.3f})")
ax_gs.legend()
fig.tight_layout()
plt.show()

assert np.isclose(density.sum(), N)
print(f"density sums to N = {N}: {np.round(density, 3)}")
