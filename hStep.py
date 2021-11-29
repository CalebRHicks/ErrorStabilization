import numpy as np
from hGuide import *
from numba import jit

#@jit

@jit(nopython=True,nogil=True)
def hStep(h,H,N,guide,stepH,deltaH,errsizeH,cutoff):
    order = H.shape[0]
    eps = stepH*errsizeH*2*(np.random.rand(order,order)-0.5)
    eps = (eps+eps.T)/2*np.sqrt(2)

    hTrial = h+eps

    guideNew,volRatio,volFlag = hGuide(hTrial,H,N,deltaH,errsizeH,cutoff)
    #print(volRatio,volFlag)
    
    if np.log(np.random.rand()) < guideNew-guide and volFlag:
        return hTrial,guideNew,True,volRatio
    else:
        return h,guide,False,volRatio


#    if (guide == 0 and guideNew == 0):
#        return h,guide,False,convergence
#    else:
#        if (guide == 0 and guideNew !=0) or np.random.rand() < guideNew/guide:
#            return hTrial,guideNew,True,convergence
#        else:
#            return h,guide,False,convergence


