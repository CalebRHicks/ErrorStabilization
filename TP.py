import numpy as np
from scipy import linalg

def TP(N=100,order=4,seed=1,eN=0.002,eH=0.002):
    np.random.seed(seed)
    H0 = 2*np.random.rand(N,N)-1
    H0 = (H0+H0.T)/2
    
    v = np.zeros((N,order))
    
    psi = np.random.rand(N)
    psi = psi/np.linalg.norm(psi)
    t = 0.1


    for nc in range(order):
        vv = np.dot(psi,linalg.expm(-H0*t*nc))
        v[:,nc] = vv

    nmat_exact = np.dot(v.T,v)
    hmat_exact = np.dot(v.T,np.dot(H0,v))

    #nmat_err = 2*np.random.rand(order,order)-1
    nmat_err = np.random.normal(0,eN,(order,order))
    nmat_err = (nmat_err + nmat_err.T)/2*np.sqrt(2)
    nmat_start = nmat_exact+nmat_err

    #hmat_err = 2*np.random.rand(order,order)-1
    hmat_err = np.random.normal(0,eH,(order,order))
    hmat_err = (hmat_err + hmat_err.T)/2*np.sqrt(2)
    hmat_start = hmat_exact+hmat_err

    return hmat_start,nmat_start,hmat_exact,nmat_exact
