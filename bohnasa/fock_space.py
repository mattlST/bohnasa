"""Fock-space basis for fixed total particle number."""

# code from Jona Naegerl
from math import comb

import numpy as np


class FixedNFockSpace:
    """Fock space with a fixed total particle number ``N`` over ``L`` sites.

    Enumerates all occupation vectors ``(n_1, ..., n_L)`` with ``sum(n) = N``
    in the order produced by the recursive generator, and builds the lookup
    ``dict`` mapping each occupation tuple to its basis index.

    Attributes
    ----------
    N : int
        Total particle number.
    L : int
        Number of sites.
    dim : int
        Dimension ``comb(N + L - 1, L - 1)``.
    fockSpace : list of tuple of int
        Occupation vectors; ``fockSpace[i]`` is the ``i``-th basis state.
    fockArray : numpy.ndarray of int, shape (dim, L)
        Same states as a dense integer array.
    dict : dict
        ``{occupation tuple: basis index}``, the inverse of ``fockSpace``.

    Examples
    --------
    >>> fock = FixedNFockSpace(2, 2)
    >>> fock.dim
    3
    >>> fock.dict[(1, 1)]
    1
    """

    def __fockgenerator(self, n, level, L, focklist, fockSpaces):
        if level > 1:
            for m in range(0, n + 1):
                focklist += (m,)
                focklist = self.__fockgenerator(n - m, level - 1, L, focklist, fockSpaces)
                if m != n and level != L:
                    focklist += focklist[-L:-level]
        else:
            focklist += (n,)
            fockSpaces += focklist[-L:]
        return focklist

    def __wrapperFockGenerator(self, N, L):
        return [
            tuple(int(x) for x in state)
            for state in np.array(self.__fockgenerator(N, L, L, (), ())).reshape(-1, L)
        ]

    def __init__(self, N, L):
        self.N = N
        self.L = L
        self.dim = comb(N + L - 1, L - 1)
        self.fockSpace = self.__wrapperFockGenerator(N, L)
        self.fockArray = np.array(self.fockSpace, dtype=int)
        self.dict = dict(zip(self.fockSpace, range(self.dim), strict=True))

    def __call__(self):
        pass

    def __str__(self):
        return f"particle number: {self.N} \n site number: {self.L}\n dimension: {self.dim} "


# Legacy alias for existing notebooks/scripts; to be dropped in Phase 5.
fockStates_fixed_particle_number = FixedNFockSpace
