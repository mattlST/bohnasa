import numpy as np 
import scipy.sparse.linalg as spl
import scipy.sparse as sp 
import typing

class operator:
    def __init__(self,fock):
        self.L = fock.L
        self.N = fock.N
        self.dim = fock.dim
        self.matrix = None #sp.zeros((self.fock_class.dim,self.fock_class),dtype=float)
    
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
        if self.matrix is not None:
            return np.asarray(self.matrix.todense()).reshape(-1)
        
    def __add__(self,A):
        newOp = self
        newOp.matrix += A.matrix
        return newOp
    
class number(operator):
    def __init__(self,fock,index):
        super().__init__(fock)
        self.index = index
        self.matrix = sp.diags(fock.fockArray[:,self.index])
    
class hopping(operator):
    def __init__(self,fock,connections = 'pbc'):
        super().__init__(fock)
        L = self.L
        dim = self.dim
        
        if (connections == 'pbc'):
            indexPairs = np.array(list(zip(np.arange(L),np.roll(np.arange(L),1))))
        elif (connections == 'obc'):
            indexPairs = np.array(list(zip(np.arange(L-1),np.arange(1,L))))
        else: 
            indexPairs = np.array(connections)
            
            assert (indexPairs.shape[1] ==2), f"connections must 'pbc', 'obc' or be 2d array with the shape (*,2), but the shape is {indexPairs.shape}"
            
        self.connections = indexPairs

        i_index = []
        j_index = []
        val = []
        for x in fock.fockSpace:
            ix = fock.dict[x]

            for index1,index2 in indexPairs:
                ay = np.array(x)
                if ay[index1] ==0:
                    continue
                ay[index1] -= 1
                ay[index2] += 1
                y = tuple(ay) 
                iy = fock.dict[y]
                i_index += [ix]
                j_index += [iy]
                val += [np.sqrt(x[index1]*y[index2])]
        i_index = np.array(i_index)
        j_index = np.array(j_index)
        val = np.array(val)

        hopping = sp.coo_matrix((val,(i_index,j_index)),shape=(dim,dim)).tocsr()
        self.matrix = hopping+hopping.T


class interaction(operator):
     def __init__(self,fock):
        super().__init__(fock)
        self.matrix = sp.diags(np.sum(fock.fockArray*(fock.fockArray-1),axis=1))


class chemical_mu(operator):
     def __init__(self,fock,mu):
        super().__init__(fock)
        self.matrix = sp.diags(np.sum(mu[None,:]*fock.fockArray,axis=1))