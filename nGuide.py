import numpy as np
from numba import jit

#@jit

@jit(nopython=True,nogil=True)
def s0(n,deltaN):
    length = np.shape(n)[0]
    minEig = [min(np.linalg.eigvals(n[:k,:k])) for k in range(2,length+1)]
    minEig = min(minEig)
    return 1/(np.exp(-minEig/deltaN)+1)


@jit(nopython=True,nogil=True)
def nGuide(n,nOld,deltaN,errsizeN):
    minEig = np.min(np.linalg.eigvals(n))
    #sigmoid = 1/(np.exp(-minEig/deltaN)+1)
    
    #sigmoid = s0(n,deltaN)
    #gauss = np.exp(-np.sum((n-nOld)**2)/(2*errsizeN**2))
    
    gauss = -np.sum((n-nOld)**2)/(2*errsizeN**2)
    if minEig/deltaN < -100:
        sigmoid = minEig/deltaN
    else:
        sigmoid = np.log(s0(n,deltaN))

    return sigmoid+gauss

