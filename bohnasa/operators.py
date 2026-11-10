"""Second-quantized operators in the fixed-N Fock basis."""

import copy
import numbers
import warnings

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

        if connections == "pbc" or connections == "periodic":
            indexPairs = np.array(list(zip(np.arange(L), np.roll(np.arange(L), 1))))
        elif connections == "obc" or connections == "open":
            indexPairs = np.array(list(zip(np.arange(L - 1), np.arange(1, L))))
        else:
            indexPairs = np.array(connections)
            if indexPairs.ndim != 2 or indexPairs.shape[1] != 2:
                raise ValueError(
                    "connections must be 'pbc' (periodic), 'obc' (open) or a 2d array "
                    f"with the shape (*,2), but the shape is {indexPairs.shape}"
                )

        self.connections = indexPairs

        T = sp.csr_matrix((self.dim, self.dim), dtype=float)
        for index1, index2 in indexPairs:
            T = T + _pair_matrix(fock, index1, index2)
        self.matrix = T + T.T


class Interaction(Operator):
    """On-site interaction operator ``sum_i n_i (n_i - 1)`` (diagonal, unit U)."""

    def __init__(self, fock):
        super().__init__(fock)
        self.matrix = sp.diags(
            np.sum(fock.fockArray * (fock.fockArray - 1), axis=1), dtype=float
        )


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
            raise ValueError(
                f"mu must have shape ({self.L},) for {self.L} sites, got {mu.shape}"
            )
        self.matrix = sp.diags(np.sum(mu[None, :] * fock.fockArray, axis=1))


def correlator_m(fock, index1, index2):
    """Deprecated shim: use ``Correlator(fock, index1, index2).adjoint()``."""
    warnings.warn(
        "correlator_m is deprecated; use Correlator(fock, index1, index2).adjoint()",
        DeprecationWarning,
        stacklevel=2,
    )
    return Correlator(fock, index1, index2).adjoint()


# Legacy aliases for existing notebooks/scripts; to be dropped in Phase 5.
operator = Operator
number = Number
correlator = Correlator
hopping = Hopping
interaction = Interaction
chemical_mu = ChemicalPotential
