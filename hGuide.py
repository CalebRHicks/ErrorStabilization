import numpy as np
#from scipy.linalg import eigh,eigvals
from numba import jit

@jit(nopython=True,nogil=True)
def hConcavity(h,N,lowestOrderRatio,convergenceRatio,meanConv = False):
    #h = h.copy()
    #N = N.copy()
    order = h.shape[0]
    e = np.zeros(order)
    for k in range(1,order+1):
        M = (np.dot(np.linalg.inv(N[:k,:k]),h[:k,:k])).astype(np.complex128)
        eigVals = np.real(np.linalg.eigvals(M))
        #eigVals = np.linalg.eigvals(np.dot(np.linalg.inv(N[:k,:k]),h[:k,:k]))
        #eigV = scipy.linalg.eigvals(h[:k,:k],N[:k,:k])
        e[k-1] = np.min(eigVals)
   # concavity = np.max([np.abs(e[k-1]-e[k])/np.abs(e[k-2]-e[k-1]) for k in range(lowestOrderRatio,order)])
    
    if e[k-2]-e[k-1] == 0:
        print("Found Equal Energies in Denominator -------------------------------------------------------")
        return np.inf

    c = [np.abs(e[k-1]-e[k])/np.abs(e[k-2]-e[k-1]) for k in range(lowestOrderRatio,order)]
    if meanConv==True:
        concavity = 1/np.mean(1/np.array(c))
    else:
        concavity = np.max(np.array(c))
    #concavity = -np.inf
    #for k in range(lowestOrderRatio,order):
    #    c = np.abs(e[k-1]-e[k])/np.abs(e[k-2]-e[k-1])
    #    if c > concavity:
    #        concavity = c
    return concavity

@jit(nopython=True,nogil=True)
def hGuide(h,H,N,deltaH,errsizeH,lowestOrderRatio,convergenceRatio,meanConv):
    #h = h.copy()
    #N = N.copy()
    #H = H.copy()
    concavity = hConcavity(h,N,lowestOrderRatio,convergenceRatio,meanConv)
#    sigmoid = 1/(np.exp((concavity-convergenceRatio)/deltaH)+1)
    #sigmoid = s1(h,concavity,convergenceRatio,deltaH)
    #gauss = np.exp(-np.sum((h-H)**2)/(2*errsizeH**2))
    #print(errsizeH)
    #print('sigmoid:',sigmoid)
    #print('gauss:',gauss)
    #print('hconv:',concavity)
    
    #Calculating in logarithms now

    order = h.shape[0]

    gauss = (-np.sum((h-H)**2)/(2*errsizeH**2))
    if (concavity-convergenceRatio)/deltaH > 10:
        sigmoid = (convergenceRatio-concavity)/deltaH
    else:
        sigmoid = np.log(s1(h,concavity,convergenceRatio,deltaH))
    return sigmoid+gauss,concavity

    #return sigmoid*gauss,concavity
@jit(nopython=True,nogil=True)
def s1(h,convergence,convergenceRatio,deltaH):
    return 1/(np.exp((convergence-convergenceRatio)/deltaH)+1)

