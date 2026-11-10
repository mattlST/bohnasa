import numpy as np 

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
    vec = np.asarray(vec)
    if vec.shape != (L,) or np.sum(vec) != N:
        raise ValueError(
            f"state vector must have shape ({L},) and particle number {N}, "
            f"got shape {vec.shape} and sum {int(np.sum(vec))}"
        )
    
    index = fock.dict[tuple(vec)]

    sta_vec = np.zeros(fock.dim,dtype=complex)
    sta_vec[index] = 1.
    return sta_vec

def coherent_state_number_projected(fock,phi):
    L = fock.L
    phi = np.asarray(phi, dtype=complex)
    if len(phi) == 2*L:
        phi = phi[:L] + 1j * phi[L:]
    elif len(phi) != L:
        raise ValueError(f"phi must have length {L} or {2*L}, got {len(phi)}")
    phi = phi / np.linalg.norm(phi)

    #preCalMulti = np.array([np.sqrt(multinomial(x)) for x in fock.fockArray])

    preCalMulti = np.array([(multinomial_sqrt(x)) for x in fock.fockArray])
    return np.prod(phi[None,:]**fock.fockArray,axis=1) * preCalMulti