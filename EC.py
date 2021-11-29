import numpy as np



def EC(N=100,order=4,seed=1,eN = 0.002,eH = 0.002):
    EC_step_coupling = 0.4/order
    target_coupling = 1
    errsize_nmat = eN
    errsize_hmat = eH
    np.random.seed(seed)
    

    H0 = 2*np.random.rand(N,N)-1
    H1 = 2*np.random.rand(N,N)-1
    H0 = (H0+H0.T)/2
    H1 = (H1+H1.T)/2

#    np.random.seed(seed)
    # Extract ground states for small coupling, put them in matrix vv
    v = np.zeros((N,order))

    for nc in range(order):
        dd,vv = np.linalg.eig(H0+nc*EC_step_coupling*H1)
        v[:,nc] = vv[:,0]

    # Target Hamiltonian and desired ground state
    Ht = H0 + target_coupling*H1
    E0 = np.min(np.linalg.eigvals(Ht))

    print('True Energy:',E0,flush=True)


    # Exact N matrix and Hamiltonian in EC subspace
    nmat_exact = np.dot(v.T,v)
    hmat_exact = np.dot(v.T,np.dot(Ht,v))

    # Generate noisy N matrix
    #nmat_err = 2*np.random.rand(order,order)-1
    nmat_err = np.random.normal(0,eN,(order,order))
    nmat_err = (nmat_err + nmat_err.T)/2*np.sqrt(2)
    nmat_start = nmat_exact+nmat_err

    # Generate noisy H matrix
    #hmat_err = 2*np.random.rand(order,order)-1
    hmat_err = np.random.normal(0,eH,(order,order))
    hmat_err = (hmat_err + hmat_err.T)/2*np.sqrt(2)
    hmat_start = hmat_exact + hmat_err


    condH = np.linalg.cond(hmat_start)
    condN = np.linalg.cond(nmat_start)

    #hmat_start = hmat_start + np.diag(np.ones(order))*1e-8
    #nmat_start = nmat_start + np.diag(np.ones(order))*1e-8


    # Solve eigenvalues/vectors of H exactly, in subspace
    vsmall_exact = np.zeros((order,order))
    for k in range(1,order+1):
        dtemp,vtemp = np.linalg.eig(np.dot(np.linalg.inv(nmat_exact[:k,:k]),hmat_exact[:k,:k]))
        vsmall_exact[:k,k-1]=vtemp[:k,np.argmin(dtemp)]

    # List of ground states at each order k of EC, exact and with noise
    e_exact = np.array([min(np.linalg.eigvals(np.dot(np.linalg.inv(nmat_exact[:k,:k]),hmat_exact[:k,:k]))) for k in range(1,order+1)])
    e_start = np.array([min(np.linalg.eigvals(np.dot(np.linalg.inv(nmat_start[:k,:k]),hmat_start[:k,:k]))) for k in range(1,order+1)])
    
 #   print(np.linalg.cond(nmat_start))

    return hmat_start,nmat_start,hmat_exact,nmat_exact
