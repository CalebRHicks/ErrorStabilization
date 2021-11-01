import numpy as np
from nGuide import *
from numba import jit


#@jit
@jit(nopython=True,nogil=True)
def nStep(n,N,guide,stepN,deltaN,errsizeN):
    order = N.shape[0]
    eps = stepN*errsizeN*2*(np.random.rand(order,order)-0.5)
    eps = (eps+eps.T)/2*np.sqrt(2)

    nTrial = n + eps

    guideNew = nGuide(nTrial,N,deltaN,errsizeN)

    if np.log(np.random.rand()) < (guideNew-guide):
        return nTrial,guideNew,True
    else:
        return n,guide,False

