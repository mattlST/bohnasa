import unittest

import bohnasa


class TestFock(unittest.TestCase):
    def test_array(self):
        N = 10
        L = 3
        fock = bohnasa.fockStates_fixed_particle_number(N, L)
        self.assertTrue(len(fock.fockArray) == fock.dim)


if __name__ == "__main__":
    unittest.main()
