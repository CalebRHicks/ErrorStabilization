import numpy as np
from hGuide import *
from numba import jit

#@jit
@jit(nopython=True,nogil=True)
def calcWeightInt(hList,H,n,stepH,deltaH,errsizeH,lowestOrderRatio,convergenceRatio,meanConv):
    N = n.copy()
    hAvg = np.sum(hList,0)/len(hList)
    order = hAvg.shape[0]
    L = int(1/2*(order+1)*order)
    cList = np.zeros((len(hList),L,L))

    mu = np.array([hAvg[i,j] for i in range(order) for j in range(i,order)])
    for ch in range(len(hList)):
        hVec = np.array([hList[ch,i,j] for i in range(order) for j in range(i,order)])
        for i in range(L):
            for j in range(L):
                cList[ch,i,j] = (hVec[i]-mu[i])*(hVec[j]-mu[j])
    cAvg = np.sum(cList,0)/len(hList)

    #print(np.linalg.eigvals(cAvg))
    epsilon = 1e-6
    cAvg = cAvg + np.eye(L)*epsilon


    gMax = gaussian(mu,mu,cAvg,L)
    
    #print(np.min(np.linalg.eigvals(cAvg)))

    A = np.linalg.cholesky(cAvg)

    wInt = 0
    terms = np.zeros(len(hList))
    for i in range(len(hList)):
        h = reshapeMatrix(mu+np.dot(A,np.random.normal(0,1,len(mu))))

        guide,convergence = hGuide(h,H,N,deltaH,errsizeH,lowestOrderRatio,convergenceRatio,meanConv)
        #guide = np.exp(guide)
        x = np.array([h[j,k] for j in range(order) for k in range(j,order)])
        g = gaussian(x,mu,cAvg,L)
        #print('g',g)
        #print('guide',guide)
        #term = np.exp(guide-g)/len(hList)
        term = guide-g
        terms[i] = term
        #wInt = wInt + guide/g*1/len(hList)
        #wInt = wInt + term
        #print(wInt)
    K = np.max(terms)
    #print(K)
    terms = terms - K
    #print(max(terms))
    S = np.sum(np.exp(terms))
    #print(S)
    wInt = np.log(S) + K #Now log of V
    #print('log(wInt):',wInt)
    #wInt = np.exp(wInt)
    return wInt,hList[-1]

#@jit(cache=True,nopython=True)
@jit(nopython=True,nogil=True)
def reshapeMatrix(v):
    order = int(np.rint(-0.5+np.sqrt(1/4+2*len(v))))
    res = np.zeros((order,order))
    k = 0
    for i in range(order):
        for j in range(i,order):
            res[i,j] = v[k]
            res[j,i] = v[k]
            k += 1
    return res

#@jit(cache=True,nopython=True)
@jit(nopython=True,nogil=True)
def gaussian(x,mu,C,L): #Log of Gaussian
    cInv = np.linalg.inv(C)
    cDet = np.abs(np.linalg.det(C))
    sign,cDet = np.linalg.slogdet(C)
    #print("det: ",cDet)

   

    g= -1/2*np.dot(x-mu,np.dot(cInv,(x-mu)))-L/2*np.log(2*np.pi)-1/2*cDet
    #print(g)
    return g
    #return -1/2*np.dot(x-mu,np.dot(cInv,(x-mu)))-L/2*np.log(2*np.pi)-1/2*cDet
    #return np.exp(-1/2*np.dot(x-mu,np.dot(cInv,(x-mu))))/(np.sqrt((2*np.pi)**(L)*cDet))


