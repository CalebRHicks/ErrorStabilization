import numpy as np
from nGuide import *
from hGuide import *
from nStep import *
from hStep import *


@jit(nopython=True)
def getNAcceptRate(N,errsizeN,deltaN,stepN):
    n = N.copy()
    guideN = nGuide(n,N,deltaN,errsizeN)
    maxTrials = int(1e3)

    nTrials = 0
    nAccepted = 0
    nSuccess = 0
    stepAcc = 0

    for i in range(maxTrials):
        nTrials += 1
        n,guideN,accepted = nStep(n,N,guideN,stepN,deltaN,errsizeN)
        if accepted:
            nAccepted += 1
            stepAcc += 1

            ev = np.sort(np.linalg.eigvals(n))[0]

            if nAccepted%(100) == 0:
                accRate = stepAcc/100
                stepAcc = 0
                stepN = stepN*2**(2*(accRate-0.5))
            if ev > 0:
                nSuccess += 1

    successRate = nSuccess/maxTrials

    return successRate
@jit(nopython=True)
def tuneN(n,errsizeN,deltaN,stepN):
    n = n.copy()

    sOld = getNAcceptRate(n,errsizeN,deltaN,stepN)

    numSteps = int(1e3)

    stepSize = 0.01

    dStart = np.log10(deltaN)
    dOld = dStart

    #x = np.logspace(dStart-5,dStart+5,numSteps)
    x = np.linspace(dStart-5,dStart+5,numSteps)


    for i in x:
        sRate = getNAcceptRate(n,errsizeN,10**i,stepN)
        if sRate > sOld:
            sOld = sRate
            dOld = i

    #for i in range(numSteps):
    #    step = 2*(np.random.rand()-0.5)*stepSize
    #    dTrial = dOld + step
#
#        sRate = getNAcceptRate(n,errsizeN,10**dTrial,stepN)
#        T = 0.1
#        if np.random.rand() < np.exp((sRate-sOld)/T):
#            dOld = dTrial
#            sOld = sRate
    return 10**dOld,sOld
@jit(nopython=True)
def getHAcceptRate(H,N,errsizeH,deltaH,stepH,lowestOrderRatio,convergenceRatio,meanConv):
    h = H.copy()
    n = N.copy()
    guideH,convergence = hGuide(h,H,n,deltaH,errsizeH,lowestOrderRatio,convergenceRatio,meanConv)

    nTrials = 0
    nAccepted = 0
    nSuccess = 0
    stepAcc = 0

    maxTrials = int(1e3)


    for i in range(maxTrials):
        nTrials += 1
        h,guideH,accepted,convergence = hStep(h,H,n,guideH,stepH,deltaH,errsizeH,lowestOrderRatio,convergenceRatio,meanConv)
        if accepted:
            nAccepted += 1
            stepAcc += 1

            if nAccepted%(100) == 0:
                accRate = stepAcc/100
                stepAcc = 0
                stepH = stepH*2**(2*(accRate-0.5))
            if convergence < convergenceRatio:
                nSuccess += 1
    return nSuccess/maxTrials
@jit(nopython=True)
def tuneH(H,N,errsizeH,deltaH,stepH,lowestOrderRatio,convergenceRatio,meanConv):
    h = H.copy()
    n = N.copy()
    
    sOld = getHAcceptRate(h,n,errsizeH,deltaH,stepH,lowestOrderRatio,convergenceRatio,meanConv)
    numSteps = int(1e3)

    stepSize = 0.1

    dStart = np.log10(deltaH)
    dOld = dStart
    x = np.linspace(dStart-5,dStart+5,numSteps)
    
    for i in x:
        sRate = getHAcceptRate(h,n,errsizeH,10**i,stepH,lowestOrderRatio,convergenceRatio,meanConv)
        if sRate > sOld:
            sOld = sRate
            dOld = i


    #for i in range(numSteps):
    #    step = 2*(np.random.rand()-0.5)*stepSize
    #    dTrial = dOld+step
#
#        sRate = getHAcceptRate(h,n,errsizeH,10**dTrial,stepH,lowestOrderRatio,convergenceRatio)
#        T = 0.01
#        if np.random.rand() < np.exp((sRate-sOld)/T):
#            dOld = dTrial
#            sOld = sRate
    return 10**dOld,sOld
