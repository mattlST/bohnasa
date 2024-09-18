import numpy as np 



#def multinomial(lst):
#    res, i = 1, sum(lst)
#    i0 = lst.index(max(lst))
#    for a in lst[:i0] + lst[i0+1:]:
#        for j in range(1,a+1):
#            res *= i
#            res //= j
#            i -= 1
#    return res
def multinomial(lst):
    res, i = 1, np.sum(lst)
    lst_sorted = np.sort(lst)[::-1]
    for a in lst_sorted:
        for j in range(1,a+1):
            res *= i
            res //= j
            i -= 1
    return res

def multinomial_sqrt(lst):
    res, i = 1, np.sum(lst)
    lst_sorted = np.sort(lst)[::-1]
    for a in lst_sorted:
        for j in range(1,a+1):
            res *= np.sqrt(i)
            res /= np.sqrt(j)
            i -= 1
    return res

def single_fockstate(fock,vec):
    N = fock.N
    L = fock.L
    assert ((np.sum(vec)==N) and (vec.shape==(L,))), "wrong input vec"
    
    index = fock.dict[tuple(vec)]

    sta_vec = np.zeros(fock.dim,dtype=complex)
    sta_vec[index] = 1.
    return sta_vec

def coherent_state_number_projected(fock,phi):
    L = fock.L
    assert ((L == len(phi)) or (2*L == len(phi))), "wrong input sizes"
    
    if (len(phi) == 2*L):
        phi = phi[:L] + 1j * phi[L:]
    phi *= 1./np.linalg.norm(phi)

    #preCalMulti = np.array([np.sqrt(multinomial(x)) for x in fock.fockArray])

    preCalMulti = np.array([(multinomial_sqrt(x)) for x in fock.fockArray])
    return np.prod(phi[None,:]**fock.fockArray,axis=1) * preCalMulti