import numpy as np 
import scipy.sparse as sp 
import copy
import numbers

class Operator:
    """A quantum operator represented as a sparse matrix in a FixedNFockSpace basis."""
    def __init__(self,fock):
        self.L = fock.L
        self.N = fock.N
        self.dim = fock.dim
        self.matrix = None
    
    def __call__(self,state):
        return self.matrix@state
    
    def eval_mean(self,state):
        return (self.matrix @ state) @ state.conj()
    
    def eval_mean_var(self,state):
        ms = self.matrix @ state
        mean = ms @ state.conj()
        var = (self.matrix@ms)@state.conj() -mean**2
        return mean,var
    
    def to_vector(self):
        if self.matrix is None:
            raise ValueError("operator has no matrix assigned yet")
        return np.asarray(self.matrix.todense()).reshape(-1)

    def __add__(self, other):
        if not isinstance(other, Operator):
            return NotImplemented
        newOp = copy.deepcopy(self)
        newOp.matrix = self.matrix + other.matrix
        return newOp

    def __sub__(self, other):
        if not isinstance(other, Operator):
            return NotImplemented
        newOp = copy.deepcopy(self)
        newOp.matrix = self.matrix - other.matrix
        return newOp

    def __mul__(self, scalar):
        if not isinstance(scalar, numbers.Number):
            return NotImplemented
        newOp = copy.deepcopy(self)
        newOp.matrix = self.matrix * scalar
        return newOp

    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
class Number(Operator):
    """Particle number operator n_i for site i."""
    def __init__(self,fock,index):
        super().__init__(fock)
        self.index = index
        self.matrix = sp.diags(fock.fockArray[:,self.index], dtype=float)

def _pair_matrix(fock, src, dst):
    """Sparse matrix of the transition operator c^dag_dst c_src.

    Built in the fixed-N Fock basis with rows = output states,
    cols = input states, amplitudes sqrt(n_src * (n_dst + 1)).
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
    """c^dag_index2 c_index1: annihilate a particle at index1, create at index2."""

    def __init__(self, fock, index1, index2):
        super().__init__(fock)
        self.index1 = index1
        self.index2 = index2
        self.matrix = _pair_matrix(fock, index1, index2)


class Hopping(Operator):
    """Sum of (c^dag_j c_i + h.c.) over the connection pairs (i, j)."""

    def __init__(self,fock,connections = 'pbc'):
        super().__init__(fock)
        L = self.L
        
        if ((connections == 'pbc') or (connections == 'periodic')):
            indexPairs = np.array(list(zip(np.arange(L),np.roll(np.arange(L),1))))
        elif ((connections == 'obc') or (connections == 'open')):
            indexPairs = np.array(list(zip(np.arange(L-1),np.arange(1,L))))
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
    """On-site interaction operator sum_i n_i (n_i - 1)."""

    def __init__(self, fock):
        super().__init__(fock)
        self.matrix = sp.diags(np.sum(fock.fockArray * (fock.fockArray - 1), axis=1), dtype=float)


class ChemicalPotential(Operator):
    """Chemical potential operator sum_i mu_i n_i."""

    def __init__(self, fock, mu):
        super().__init__(fock)
        self.matrix = sp.diags(np.sum(mu[None, :] * fock.fockArray, axis=1))



class correlator_m(Operator):
    """c^dag_index1 c_index2: annihilate a particle at index2, create at index1."""

    def __init__(self, fock, index1, index2):
        super().__init__(fock)
        self.index1 = index1
        self.index2 = index2
        self.matrix = _pair_matrix(fock, index2, index1)


# Legacy aliases for existing notebooks/scripts; to be dropped in Phase 5.
operator = Operator
number = Number
correlator = Correlator
hopping = Hopping
interaction = Interaction
chemical_mu = ChemicalPotential
