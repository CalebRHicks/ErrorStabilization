import numpy as np
from hGuide import *
from numba import jit

#@jit

@jit(nopython=True,nogil=True)
def hStep(h,H,N,guide,stepH,deltaH,errsizeH,lowestOrderRatio,convergenceRatio,meanConv):
    order = H.shape[0]
    eps = stepH*errsizeH*2*(np.random.rand(order,order)-0.5)
    eps = (eps+eps.T)/2*np.sqrt(2)

    hTrial = h+eps

    guideNew,convergence = hGuide(hTrial,H,N,deltaH,errsizeH,lowestOrderRatio,convergenceRatio,meanConv)
    
    if np.log(np.random.rand()) < guideNew-guide:
        return hTrial,guideNew,True,convergence
    else:
        return h,guide,False,convergence


#    if (guide == 0 and guideNew == 0):
#        return h,guide,False,convergence
#    else:
#        if (guide == 0 and guideNew !=0) or np.random.rand() < guideNew/guide:
#            return hTrial,guideNew,True,convergence
#        else:
#            return h,guide,False,convergence


