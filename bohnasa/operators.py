"""Second-quantized operators in the fixed-N Fock basis."""

import copy
import numbers
from collections.abc import Sequence

import numpy as np
import scipy.sparse as sp


class Operator:
    """A quantum operator as a sparse matrix in a ``FixedNFockSpace`` basis.

    Subclasses fill ``matrix`` with the operator's CSR representation
    (rows = output states, cols = input states, acting on column vectors).

    Parameters
    ----------
    fock : FixedNFockSpace
        Basis the operator is represented in.

    Attributes
    ----------
    matrix : scipy.sparse.csr_matrix or None
        Operator matrix of shape ``(dim, dim)``; ``None`` until assigned.
    N : int
        Total particle number of the underlying basis.
    L : int
        Number of sites/modes of the underlying basis.
    dim : int
        Hilbert-space dimension of the underlying basis.

    Examples
    --------
    >>> from bohnasa import FixedNFockSpace, Number
    >>> fock = FixedNFockSpace(2, 3)
    >>> n0, n1 = Number(fock, 0), Number(fock, 1)
    >>> total = n0 + n1
    >>> np.allclose(total.matrix.diagonal(),
    ...             fock.fockArray[:, 0] + fock.fockArray[:, 1])
    True
    """

    def __init__(self, fock):
        self.L = fock.L
        self.N = fock.N
        self.dim = fock.dim
        self.matrix = None

    def __call__(self, state):
        """Apply the operator to a state vector: ``op(state) == matrix @ state``."""
        return self.matrix @ state

    def eval_mean(self, state):
        """Expectation value ``<state|matrix|state>`` (complex scalar)."""
        return (self.matrix @ state) @ state.conj()

    def eval_mean_var(self, state):
        """Expectation value and variance ``(<A>, <A^2> - <A>^2)`` of the operator."""
        ms = self.matrix @ state
        mean = ms @ state.conj()
        var = (self.matrix @ ms) @ state.conj() - mean**2
        return mean, var

    def to_vector(self):
        """Flatten ``matrix`` into a dense vector (e.g. for Krylov-basis analysis).

        Raises
        ------
        ValueError
            If no matrix has been assigned yet.
        """
        if self.matrix is None:
            raise ValueError("operator has no matrix assigned yet")
        return np.asarray(self.matrix.todense()).reshape(-1)

    def __add__(self, other):
        """Sum of two operators on the same basis."""
        if not isinstance(other, Operator):
            return NotImplemented
        newOp = copy.deepcopy(self)
        newOp.matrix = self.matrix + other.matrix
        return newOp

    def __sub__(self, other):
        """Difference of two operators on the same basis."""
        if not isinstance(other, Operator):
            return NotImplemented
        newOp = copy.deepcopy(self)
        newOp.matrix = self.matrix - other.matrix
        return newOp

    def __mul__(self, scalar):
        """Scalar multiple of the operator."""
        if not isinstance(scalar, numbers.Number):
            return NotImplemented
        newOp = copy.deepcopy(self)
        newOp.matrix = self.matrix * scalar
        return newOp

    def __rmul__(self, scalar):
        """Scalar multiple, reflected form (``scalar * op``)."""
        return self.__mul__(scalar)

    def adjoint(self):
        """Hermitian conjugate as a new operator (``matrix.T.conj()``).

        Note: metadata attributes (``index1``, ``index2``, ...) still describe
        the original operator.

        Examples
        --------
        >>> from bohnasa import FixedNFockSpace, Correlator
        >>> fock = FixedNFockSpace(2, 3)
        >>> c01 = Correlator(fock, 0, 1)
        >>> np.allclose(c01.adjoint().matrix.toarray(),
        ...             Correlator(fock, 1, 0).matrix.toarray())
        True
        """
        newOp = copy.deepcopy(self)
        newOp.matrix = self.matrix.T.conj().tocsr()
        return newOp


class Number(Operator):
    """Particle number operator ``n_index`` for one site (diagonal).

    Examples
    --------
    >>> from bohnasa import FixedNFockSpace
    >>> fock = FixedNFockSpace(2, 3)
    >>> Number(fock, 0).matrix.diagonal()
    array([0., 0., 0., 1., 1., 2.])
    """

    def __init__(self, fock, index):
        super().__init__(fock)
        self.index = index
        self.matrix = sp.diags(fock.fockArray[:, self.index], dtype=float)


def _pair_matrix(fock, src, dst):
    """Sparse matrix of the transition operator ``c^dag_dst c_src``.

    Built in the fixed-N Fock basis with rows = output states,
    cols = input states, amplitudes ``sqrt(n_src * (n_dst + 1))``.
    """
    rows = []
    cols = []
    vals = []
    for x in fock.fockSpace:
        n_src = x[src]
        if n_src == 0:
            continue
        occ = list(x)
        occ[src] -= 1
        occ[dst] += 1
        vals.append(np.sqrt(n_src * occ[dst]))
        rows.append(fock.dict[tuple(occ)])
        cols.append(fock.dict[x])
    if not rows:
        return sp.csr_matrix((fock.dim, fock.dim), dtype=float)
    return sp.coo_matrix(
        (np.array(vals), (np.array(rows), np.array(cols))),
        shape=(fock.dim, fock.dim),
    ).tocsr()


class Correlator(Operator):
    """Two-site correlator ``c^dag_index2 c_index1``: annihilate at ``index1``,
    create at ``index2`` (a directed hop ``index1 -> index2``).

    The adjoint hop is ``Correlator(fock, index1, index2).adjoint()``.
    """

    def __init__(self, fock, index1, index2):
        super().__init__(fock)
        self.index1 = index1
        self.index2 = index2
        self.matrix = _pair_matrix(fock, index1, index2)


def bond_pairs(
    L: int,
    connections: str | np.ndarray | Sequence[Sequence[int]] = "pbc",
) -> np.ndarray:
    r"""Directed bond list ``indexPairs[e] = (i, j)`` for an ``L``-site geometry.

    Single source of truth for the bond convention, shared by :class:`Hopping`
    and :func:`bohnasa.visualization.lattice_bonds` so the Hamiltonian and the
    drawn lattice can never drift apart. ``"pbc"`` pairs site ``i`` with its
    left neighbour ``(i - 1) mod L`` and ``"obc"`` with ``i + 1``; both hop
    directions are added by ``Hopping`` itself (``T + T.T``), so each bond of
    the geometric ring or chain enters the sum exactly once - except at
    ``L = 2`` with ``"pbc"``, where the ring's single bond is deliberately
    counted from both ends (see :class:`Hopping`).

    Parameters
    ----------
    L : int
        Number of lattice sites.
    connections : {'pbc', 'periodic', 'obc', 'open'} or array-like of shape (*, 2)
        Bond definition: periodic ring, open chain, or an explicit list of
        directed site pairs ``(i, j)`` passed through as an integer array.

    Returns
    -------
    numpy.ndarray of int, shape (*, 2)
        The ordered bond list.

    Raises
    ------
    ValueError
        If an explicit bond list is not 2-dimensional with shape ``(*, 2)``.

    Examples
    --------
    >>> bond_pairs(3, "pbc")
    array([[0, 2],
           [1, 0],
           [2, 1]])
    >>> bond_pairs(3, "obc")
    array([[0, 1],
           [1, 2]])
    >>> bond_pairs(2, "pbc").shape
    (2, 2)
    """
    if isinstance(connections, str) and connections in ("pbc", "periodic"):
        return np.array(list(zip(np.arange(L), np.roll(np.arange(L), 1), strict=True)))
    if isinstance(connections, str) and connections in ("obc", "open"):
        return np.array(list(zip(np.arange(L - 1), np.arange(1, L), strict=True)))
    indexPairs = np.array(connections)
    if indexPairs.ndim != 2 or indexPairs.shape[1] != 2:
        raise ValueError(
            "connections must be 'pbc' (periodic), 'obc' (open) or a 2d array "
            f"with the shape (*,2), but the shape is {indexPairs.shape}"
        )
    return indexPairs


class Hopping(Operator):
    """Hopping operator ``sum_(i,j) (c^dag_j c_i + h.c.)`` over connections.

    Parameters
    ----------
    fock : FixedNFockSpace
        Basis of the system.
    connections : {'pbc', 'periodic', 'obc', 'open'} or array-like of shape (*, 2)
        Bond definition: periodic ring, open chain, or an explicit list of
        directed site pairs ``(i, j)``.

    Raises
    ------
    ValueError
        If an explicit bond list is not 2-dimensional with shape ``(*, 2)``.

    Notes
    -----
    The bond list comes from :func:`bond_pairs`, the single builder shared
    with :mod:`bohnasa.visualization` so geometry and Hamiltonian cannot
    drift apart. For ``connections="pbc"`` at ``L = 2`` the ring's single
    bond is counted from both ends, giving hopping amplitude ``2J`` across
    the dimer. This is the deliberate convention (it keeps the L = 2 ring
    topologically consistent with the general periodic chain), not a defect -
    use the :class:`Hamiltonian` builder with an explicit ``one_body``
    matrix for a deduplicated bond.

    Examples
    --------
    >>> from bohnasa import FixedNFockSpace
    >>> fock = FixedNFockSpace(1, 6)
    >>> H = Hopping(fock, connections="pbc")
    >>> np.round(np.sort(np.linalg.eigvalsh(H.matrix.toarray())), 6)
    array([-2., -1., -1.,  1.,  1.,  2.])
    """

    def __init__(self, fock, connections="pbc"):
        super().__init__(fock)
        L = self.L
        indexPairs = bond_pairs(L, connections)
        self.connections = indexPairs

        T = sp.csr_matrix((self.dim, self.dim), dtype=float)
        for index1, index2 in indexPairs:
            T = T + _pair_matrix(fock, index1, index2)
        self.matrix = T + T.T


class Interaction(Operator):
    """On-site interaction operator ``sum_i n_i (n_i - 1)`` (diagonal, unit U)."""

    def __init__(self, fock):
        super().__init__(fock)
        self.matrix = sp.diags(np.sum(fock.fockArray * (fock.fockArray - 1), axis=1), dtype=float)


class ChemicalPotential(Operator):
    """Chemical potential operator ``sum_i mu_i n_i`` (diagonal).

    Parameters
    ----------
    fock : FixedNFockSpace
        Basis of the system.
    mu : array-like of shape (L,)
        Chemical potential per site.

    Raises
    ------
    ValueError
        If ``mu`` does not have shape ``(L,)``.
    """

    def __init__(self, fock, mu):
        super().__init__(fock)
        mu = np.asarray(mu, dtype=float)
        if mu.shape != (self.L,):
            raise ValueError(f"mu must have shape ({self.L},) for {self.L} sites, got {mu.shape}")
        self.matrix = sp.diags(np.sum(mu[None, :] * fock.fockArray, axis=1))


def _quartic_matrix(fock, i, j, k, ell):
    """Sparse matrix of the normal-ordered monomial ``c^dag_i c^dag_j c_k c_l``.

    The rightmost operator acts first: annihilate at ``l``, then ``k``, then
    create at ``j``, then ``i`` - occupation bookkeeping is exact for
    coinciding indices (``k == l`` needs ``n_l >= 2`` etc. automatically).
    """
    rows = []
    cols = []
    vals = []
    for col, x in enumerate(fock.fockSpace):
        occ = list(x)
        amp = 1.0
        for site in (ell, k):
            if occ[site] == 0:
                amp = 0.0
                break
            amp *= np.sqrt(occ[site])
            occ[site] -= 1
        if amp == 0.0:
            continue
        for site in (j, i):
            amp *= np.sqrt(occ[site] + 1)
            occ[site] += 1
        rows.append(fock.dict[tuple(occ)])
        cols.append(col)
        vals.append(amp)
    if not rows:
        return sp.csr_matrix((fock.dim, fock.dim), dtype=float)
    return sp.coo_matrix(
        (np.array(vals), (np.array(rows), np.array(cols))),
        shape=(fock.dim, fock.dim),
    ).tocsr()


class Hamiltonian(Operator):
    r"""Generic normal-ordered one- and two-body bosonic Hamiltonian.

    Assembles ``H = sum_ij t_ij c^dag_i c_j + sum_ijkl V_ijkl c^dag_i c^dag_j
    c_k c_l`` from coefficient arrays on a ``FixedNFockSpace`` basis.
    Coefficients are taken literally: no Hermiticity or index symmetry is
    imposed or assumed (a Hermitian ``H`` needs Hermitian input, e.g. the
    full set of symmetry-related ``V`` entries). Only nonzero entries are
    materialized, so dense-but-sparse-valued inputs stay cheap.

    Parameters
    ----------
    fock : FixedNFockSpace
        Basis of the system.
    one_body : array-like of shape (L, L), optional
        ``t_ij`` = coefficient of ``c^dag_i c_j`` (rightmost operator acts
        first). A diagonal entry ``t_ii`` is a chemical potential ``mu_i n_i``;
        ``-t`` on a symmetric bond adjacency reproduces ``-t`` times
        ``Hopping``.
    two_body : array-like of shape (L, L, L, L), optional
        ``V_ijkl`` = coefficient of ``c^dag_i c^dag_j c_k c_l``. Mutually
        exclusive with ``onsite_u``.
    onsite_u : float or array-like of shape (L,), optional
        Convenience for the on-site term ``sum_i u_i n_i (n_i - 1)``, i.e.
        ``V_iiii = u_i``. Mutually exclusive with ``two_body``.

    Raises
    ------
    ValueError
        If a coefficient array has the wrong shape, or ``two_body`` and
        ``onsite_u`` are both given.

    Notes
    -----
    With all inputs omitted the result is the zero operator. The term
    convention carries no 1/2 factors: the Bose-Hubbard model
    ``-t sum (c^dag_j c_i + h.c.) + U/2 sum n_i (n_i - 1)`` is
    ``Hamiltonian(fock, one_body=-t * adjacency, onsite_u=U / 2)``.

    Examples
    --------
    >>> from bohnasa import FixedNFockSpace, Hopping, Interaction
    >>> fock = FixedNFockSpace(2, 3)
    >>> adjacency = np.ones((3, 3)) - np.eye(3)  # ring graph for L = 3
    >>> H = Hamiltonian(fock, one_body=-adjacency, onsite_u=0.5)
    >>> Href = Hopping(fock, connections="pbc") * (-1.0) + Interaction(fock) * 0.5
    >>> np.allclose(H.matrix.toarray(), Href.matrix.toarray())
    True
    """

    def __init__(self, fock, one_body=None, two_body=None, onsite_u=None):
        super().__init__(fock)
        L = self.L
        if two_body is not None and onsite_u is not None:
            raise ValueError("Hamiltonian: give two_body or onsite_u, not both")

        matrix = sp.csr_matrix((self.dim, self.dim), dtype=float)

        if one_body is not None:
            t = np.asarray(one_body)
            if t.shape != (L, L):
                raise ValueError(f"Hamiltonian: one_body must have shape ({L}, {L}), got {t.shape}")
            for i, j in zip(*np.nonzero(t), strict=False):
                matrix = matrix + t[i, j] * _pair_matrix(fock, int(j), int(i))

        if onsite_u is not None:
            u = np.atleast_1d(np.asarray(onsite_u, dtype=float))
            if u.size == 1:
                u = np.full(L, float(u[0]))
            if u.shape != (L,):
                raise ValueError(
                    f"Hamiltonian: onsite_u must be a scalar or shape ({L},), got {u.shape}"
                )
            occ = fock.fockArray
            matrix = matrix + sp.diags(np.sum(u[None, :] * occ * (occ - 1), axis=1))

        if two_body is not None:
            V = np.asarray(two_body)
            if V.shape != (L, L, L, L):
                raise ValueError(
                    f"Hamiltonian: two_body must have shape ({L}, {L}, {L}, {L}), got {V.shape}"
                )
            for i, j, k, ell in zip(*np.nonzero(V), strict=False):
                matrix = matrix + V[i, j, k, ell] * _quartic_matrix(
                    fock, int(i), int(j), int(k), int(ell)
                )

        self.matrix = matrix
