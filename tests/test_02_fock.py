import unittest

import numpy as np

import bohnasa


class TestOperator(unittest.TestCase):
    def test_hopping(self):
        N = 6
        L = 3
        fock = bohnasa.fockStates_fixed_particle_number(N, L)
        hopping = bohnasa.hopping(fock, connections=[(1, 2), (2, 1)]).matrix
        self.assertTrue(hopping.shape == (fock.dim, fock.dim))
        self.assertTrue(np.allclose(hopping.todense(), hopping.todense().T))


if __name__ == "__main__":
    unittest.main()
