import numpy as np
import scipy.sparse as sp
from . import operators



def lanczos(sta,op,krylov_dim,beta_tol=1e-8):

    v = np.zeros((krylov_dim + 1, sta.shape[0]), dtype=complex)
    T_m = np.zeros((krylov_dim+1,krylov_dim+1),dtype=complex)
    v[0] = sta/np.linalg.norm(sta)

    w_prime = op(v[0])

    alpha = np.vdot(w_prime, v[0, :])

    w = w_prime - alpha * v[0, :]

    T_m[0, 0] = alpha

    for j in range(1, krylov_dim + 1):

        beta = np.linalg.norm(w)
        if beta < beta_tol:

            # Happy breakdown
            v_happy = v[0:j, :]
            T_m_happy = T_m[0:j, 0:j]

            return v_happy, T_m_happy

        v[j, :] = w / beta
        w_prime = op(v[j, :])
        alpha = np.vdot(w_prime, v[j, :])

        w = w_prime - alpha * v[j, :] - beta * v[j - 1, :] 
        w = w - v.T@(v.conj()@w)        
        while True:
           wn = w - v.T@(v.conj()@w)
           if (np.linalg.norm(wn-w))<1e-12:
               w = wn
               break
           print("index", j,"norm of difference",np.linalg.norm(wn-w))
           w = wn
           print('repeat')

        T_m[j, j] = alpha
        T_m[j, j - 1] = beta
        T_m[j - 1, j] = beta

    return v, T_m

def Krylov_complexity(fock,op,sta,numVec):
    v, T_m = lanczos(sta,op,numVec)

    krylov_op = operators.operator(fock)
    krylov_op.matrix = sp.csr_matrix(v.conj().T@np.diag(np.arange(v.shape[0]))@v)
    return krylov_op

def time_evolution_exact_ind(H,state,time,observables):
    if not hasattr(H,"eigval"):
        eigval, eigvec = np.linalg.eigh(H.matrix.todense())
        H.eigval = np.asarray(eigval)
        H.eigvec = np.asarray(eigvec)
    state = np.asarray(state,dtype=complex)
    state_eval = H.eigvec.T.conj() @ state
    print(state.shape)

    print(state_eval.shape)
    dt = time[1]-time[0]
    dE = np.exp(-1j *dt*H.eigval)
    results_observables = np.zeros((len(observables),2,len(time)),dtype=float)
    for iob,ob in enumerate(observables):
        results_observables[iob,:,0] = ob.eval_mean_var(state)
    for it,t in enumerate(time[:-1]):
        state_eval *= dE
        ret = H.eigvec@state_eval
        for iob,ob in enumerate(observables):
            results_observables[iob,:,it+1] = ob.eval_mean_var(ret)
    return results_observables
