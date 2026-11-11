"""Example 01: the fixed-N Fock space.

Question: what does the basis of N bosons on L sites look like?

Enumerates all occupation patterns with total particle number N, prints the
binomial dimension and a window of states, and shows the index lookup that
every operator builder relies on.
"""

from math import comb

import numpy as np

from bohnasa import FixedNFockSpace

N, L = 5, 3
fock = FixedNFockSpace(N, L)

print(f"N = {N} bosons on L = {L} sites")
print(f"dim = comb(N+L-1, L-1) = {fock.dim}")
print("first five basis states (site occupations -> basis index):")
for state in fock.fockSpace[:5]:
    print(f"  {state} -> {fock.dict[state]}")

# every row is a valid occupation pattern and every pattern appears once
assert fock.dim == comb(N + L - 1, L - 1)
assert len(set(fock.fockSpace)) == fock.dim
assert np.all(fock.fockArray.sum(axis=1) == N)
print("sanity checks passed")
