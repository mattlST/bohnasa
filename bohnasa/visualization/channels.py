from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .geometry import _validate_bonds


def bond_energy(gamma: ArrayLike, bonds: ArrayLike) -> np.ndarray:
    r"""Symmetric bond energies ``e_ij = 2 Re gamma_ij = gamma_ij + gamma_ji``.

    One real value per bond, in units of the hopping amplitude: under
    ``H = -t sum (c_j^dag c_i + h.c.)`` the *kinetic*-energy contribution
    of that bond is ``-e_ij``, so summing ``e`` over the ring or chain
    bond list reproduces ``<Hopping> = -<H_kin>`` exactly (pinned in the
    tests). Being invariant under ``i <-> j`` it is well defined on the
    undirected edge, hence the orientation-free base channel (edge width
    and color) of the visual grammar; the directed flow is carried
    separately by :func:`bond_current` (arrowheads).

    Parameters
    ----------
    gamma : array-like of complex, shape (L, L)
        One-body density matrix ``gamma_ij = <c_i^dag c_j>`` from
        :func:`bohnasa.one_body_density_matrix` (assumed normalized).
    bonds : array-like of int, shape (E, 2)
        Directed bond list, e.g. from :func:`lattice_bonds`.

    Returns
    -------
    numpy.ndarray of float, shape (E,)
        Bond energy per bond (same value for ``(i, j)`` and ``(j, i)``).

    Examples
    --------
    >>> gamma = np.array([[0.5, 0.25 + 0.25j], [0.25 - 0.25j, 0.5]])
    >>> bond_energy(gamma, [[0, 1], [1, 0]])
    array([0.5, 0.5])
    """
    gamma = np.asarray(gamma, dtype=complex)
    if gamma.ndim != 2 or gamma.shape[0] != gamma.shape[1]:
        raise ValueError(f"bond_energy: gamma must be square 2d, got shape {gamma.shape}")
    bonds = _validate_bonds(bonds, gamma.shape[0])
    return 2.0 * gamma[bonds[:, 0], bonds[:, 1]].real


def bond_current(gamma: ArrayLike, bonds: ArrayLike) -> np.ndarray:
    r"""Bond currents ``j_ij = i (gamma_ji - gamma_ij) = 2 Im gamma_ij``.

    One real value per directed bond of ``bonds``, in units of the hopping
    amplitude; positive means flow from the bond's first site to its
    second (see the module docstring for the convention and the continuity
    derivation test that pins it). In the visual grammar this quantity
    drives the edge arrowhead (direction and size), not the edge base
    width or color.

    Parameters
    ----------
    gamma : array-like of complex, shape (L, L)
        One-body density matrix ``gamma_ij = <c_i^dag c_j>`` from
        :func:`bohnasa.one_body_density_matrix` (assumed normalized).
    bonds : array-like of int, shape (E, 2)
        Directed bond list, e.g. from :func:`lattice_bonds`.

    Returns
    -------
    numpy.ndarray of float, shape (E,)
        Signed current per bond.

    Examples
    --------
    >>> gamma = np.array([[0.5, 0.25j], [-0.25j, 0.5]])
    >>> bond_current(gamma, [[0, 1], [1, 0]])
    array([ 0.5, -0.5])
    """
    gamma = np.asarray(gamma, dtype=complex)
    if gamma.ndim != 2 or gamma.shape[0] != gamma.shape[1]:
        raise ValueError(f"bond_current: gamma must be square 2d, got shape {gamma.shape}")
    bonds = _validate_bonds(bonds, gamma.shape[0])
    return 2.0 * gamma[bonds[:, 0], bonds[:, 1]].imag


def bond_coherence(gamma: ArrayLike, bonds: ArrayLike) -> np.ndarray:
    r"""Bond coherence ``|gamma_ij|`` per directed bond.

    The alternative edge encoding of :func:`animate_lattice`
    (``edge_mode="coherence"``): the magnitude of the one-body density
    matrix across each bond, a non-negative quantity.

    Parameters
    ----------
    gamma : array-like of complex, shape (L, L)
        One-body density matrix ``gamma_ij = <c_i^dag c_j>``.
    bonds : array-like of int, shape (E, 2)
        Directed bond list, e.g. from :func:`lattice_bonds`.

    Returns
    -------
    numpy.ndarray of float, shape (E,)
        ``|gamma_ij|`` per bond.

    Examples
    --------
    >>> gamma = np.array([[0.5, 0.25j], [-0.25j, 0.5]])
    >>> bond_coherence(gamma, [[0, 1], [1, 0]])
    array([0.25, 0.25])
    """
    gamma = np.asarray(gamma, dtype=complex)
    if gamma.ndim != 2 or gamma.shape[0] != gamma.shape[1]:
        raise ValueError(f"bond_coherence: gamma must be square 2d, got shape {gamma.shape}")
    bonds = _validate_bonds(bonds, gamma.shape[0])
    return np.abs(gamma[bonds[:, 0], bonds[:, 1]])
