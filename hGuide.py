import numpy as np
#from scipy.linalg import eigh,eigvals
from numba import jit

#@jit(nopython=True,nogil=True)
#def hConcavity(h,N,lowestOrderRatio,convergenceRatio,meanConv = False):
#    #h = h.copy()
#    #N = N.copy()
#    order = h.shape[0]
#    e = np.zeros(order)
#    for k in range(1,order+1):
#        M = (np.dot(np.linalg.inv(N[:k,:k]),h[:k,:k])).astype(np.complex128)
#        eigVals = np.real(np.linalg.eigvals(M))
#        #eigVals = np.linalg.eigvals(np.dot(np.linalg.inv(N[:k,:k]),h[:k,:k]))
#        #eigV = scipy.linalg.eigvals(h[:k,:k],N[:k,:k])
#        e[k-1] = np.min(eigVals)
#   # concavity = np.max([np.abs(e[k-1]-e[k])/np.abs(e[k-2]-e[k-1]) for k in range(lowestOrderRatio,order)])
#    
#    if e[k-2]-e[k-1] == 0:
#        print("Found Equal Energies in Denominator -------------------------------------------------------")
#        return np.inf
#
#    c = [np.abs(e[k-1]-e[k])/np.abs(e[k-2]-e[k-1]) for k in range(lowestOrderRatio,order)]
#    if meanConv==True:
#        concavity = 1/np.mean(1/np.array(c))
#    else:
#        concavity = np.max(np.array(c))
#    #concavity = -np.inf
#    #for k in range(lowestOrderRatio,order):
#    #    c = np.abs(e[k-1]-e[k])/np.abs(e[k-2]-e[k-1])
#    #    if c > concavity:
#    #        concavity = c
#    return concavity
#





@jit(nopython=True,nogil=True)
def hVolume(h,N,cutoff):
    order = h.shape[0]

    sgnMaxOrder,volMaxOrder,volFlag = vol(h,N)
    sgnNextOrder,volNextOrder,volFlag2 = vol(h[:-1,:-1],N[:-1,:-1])
    
    #print(sgnMaxOrder,sgnNextOrder)


    if volFlag == False or volFlag2 == False:
        return 9999.,False
    elif sgnMaxOrder > 0 and sgnNextOrder > 0:
        return volMaxOrder-volNextOrder,True
    else:
        return 9999.,False



@jit(nopython=True,nogil=True)
def hGuide(h,H,N,deltaH,errsizeH,cutoff):
    
    volRatio,volumeFlag = hVolume(h,N,cutoff)
    
    if volumeFlag == False:
        return 0.,9999.,False

    #Calculating in logarithms now

    order = h.shape[0]

    gauss = (-np.sum((h-H)**2)/(2*errsizeH**2))
    if (volRatio-cutoff)/deltaH > 10:
        sigmoid = (cutoff-volRatio)/deltaH
    else:
        sigmoid = np.log(s1(volRatio,cutoff,deltaH))
    return sigmoid+gauss,volRatio,True

    #return sigmoid*gauss,concavity
@jit(nopython=True,nogil=True)
def s1(volRatio,cutoff,deltaH):
    return 1/(np.exp((volRatio-cutoff)/deltaH)+1)

@jit(nopython=True,nogil=True)
def vol(H,N):
    order = len(N)
    groundStates = np.zeros((order,order),dtype=np.complex128)

    #At first order, the gs is 1
    
    if N[0,0] == 0:
        return 0,9999,False
    groundStates[0,0] = 1/np.sqrt(N[0,0])
    
    for i in range(1,order):
        gs,gsFlag = getGS(H[:i+1,:i+1],N[:i+1,:i+1])
        if gsFlag:
            groundStates[:i+1,i] = gs
        else:
            return 0.,9999.,False
    m = np.zeros((order,order),dtype=np.complex128)

    for i in range(order):
        for j in range(i,order):
            m[j,i] = m[i,j] = inner(groundStates[:i+1,i],N[:i+1,:j+1],groundStates[:j+1,j])
    sign,logDet = np.linalg.slogdet(m)

    return np.real(sign),logDet,True

@jit(nopython=True,nogil=True)
def inner(a,N,b):
    a = a.copy().astype(np.complex128)
    b = b.copy().astype(np.complex128)
    N = N.copy().astype(np.complex128)
    #return np.linalg.multi_dot([np.conj(a),N,b])
    return np.dot(np.dot(np.conj(a),N),b)
@jit(nopython=True,nogil=True)
def getGS(H,N):
    #print('test')
    #print(np.linalg.eig(N)[0])
    #print(np.linalg.det(np.linalg.inv(N)))
    M = (np.dot(np.linalg.inv(N),H)).astype(np.complex128)
    dd,vv = np.linalg.eig(M)
    #print('test2')

    i = np.argsort(np.real(dd))[0]
    v = vv[:,i]
    
    norm = np.abs(inner(v,N,v))
    if norm == 0:
        return v, False
    if np.imag(dd[i]) < 1e-9:
        return v/np.sqrt(norm),True
    else:
        return np.ones(len(N),dtype=np.complex128),False












