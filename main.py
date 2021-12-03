import numpy as np

from scipy.linalg import eigh

from nGuide import *
from hGuide import *
from nStep import *
from hStep import *
from calcWeightInt import *
from autotune import *
import multiprocessing as mp
import os
import time
import sys
from scipy.special import logsumexp
#Function to return a single n matrix after moving through the chain for a bit
@jit(nopython=True,nogil=True)
def sampleN(nStart,N,errsizeN,deltaN,stepN,verbose):

    n = nStart.copy()
    guideN = nGuide(nStart,N,deltaN,errsizeN)
    
    nAccepted = 0
    trials = 0

    while nAccepted < 500:
        #step N and calculate its guide function
        n,guideN,accepted = nStep(n,N,guideN,stepN,deltaN,errsizeN)
        trials+= 1
        if accepted:
            #see if it meets the cutoff. If not, reject it. Otherwise, add to list.
            length = np.shape(n)[0]
            minEigList = [min(np.linalg.eigvals(n[:k,:k])) for k  in range(2,length+1)]
            minEig = min(minEigList)
            if minEig > 0 and n[0,0] > 0:
                nAccepted += 1
                #if verbose and nAccepted%(500/10) == 0: print('N matrix Acceptance Ratio: ',nAccepted/trials)
    return n
#Function to sample a list of H matrices, with option to take only those that meet the hard cutoff.
#This function has the most room for improvement. When the code gets stuck, it is almost always here.
#The cutoff condition is sometimes very difficult. The code can get stuck when it finds an early h matrix or two
#that meet the cutoff condition, but it does not find any others. The code can run indefinitely in this case.
#One possible solution to this is to implement minimum progress for given numbers of accepted matrices.
#This would prevent the code from getting stuck, but it can still loop indefinitely trying to find a good n matrix
#A better solution is to implement smart autotuning so that we can find better h matrices more quickly,
#but that is messy and hard to implement. I will do so later.
@jit(nopython=True,nogil=True)
def sampleH(H,n,hSamples,errsizeH,deltaH,stepH,cutoff,verbose,applyCutoff):
    
    maxChainBase = stepH*1*hSamples

    h = H.copy()
    n = n.copy()
    guideH,volRatio,_ = hGuide(h,H,n,deltaH,errsizeH,cutoff)
    order = np.shape(h)[0]
    trials = 0
    nAccepted = 0
    hList = np.zeros((hSamples,order,order))
    k = 0
    nStepAccepted =0

    chainLength = 0

    while k < hSamples:
        trials += 1
        h,guideH,accepted,volRatio = hStep(h,H,n,guideH,stepH,deltaH,errsizeH,cutoff)
        if accepted:
            nStepAccepted +=1
            nAccepted += 1
            printOption=False
            chainLength += max([stepH,maxChainBase/1e5])
           # print('convergence:',convergence)
           # print('length',chainLength,'/',maxChainBase)
           # print(volRatio)

            if applyCutoff==True:
                if volRatio < cutoff:
                    hList[k,:,:] = h
                    k += 1
                    printOption=True
            else:
                if nAccepted%10 == 0:
                    hList[k,:,:] = h
                    k+=1
                    printOption=True
            #if verbose and k%(hSamples/10) == 0 and printOption==True: print('Acceptance Ratio: ',nAccepted/trials)
            #if verbose and k%(hSamples/10) == 0 and printOption ==True and applyCutoff==True: print('Valid H to Accepted Ratio: ',k/nAccepted)
            if applyCutoff and k == 0 and chainLength > maxChainBase:
                if verbose: print('no valid H matrices found, abandoning N matrix')
                return None
            if applyCutoff and k < hSamples/2 and chainLength > 10*maxChainBase:
                if verbose: print('not enough valid H matrices found, abandoning N matrix')
                return None
            if applyCutoff and chainLength > 25*maxChainBase:
                if verbose: print('not enough valid H matrices found, abandoning N matrix')
                return None
        if chainLength > 50*maxChainBase:
            if verbose: print('Not enough valid H matrices found, abandoning N matrix')
            return None
        if nAccepted%100 == 0:
            accRate = nStepAccepted/100
            stepH = stepH * 2**(2*(accRate-0.5))
            nStepAccepted = 0
        if trials > hSamples*10000:
            if verbose: print('Not enough H matrices found, abandoning N matrix, found ',k)
            return None
    return hList

#Default operator to calculate ground state energy. May be substituted with any function that takes to matrices H and N as inputs and returns either a scalar or numpy array.
def defaultOperator(H,N):
    return np.sort(eigh(H,N,eigvals_only=True))[0]

def calcOnePair(H,nStart,N,O,errsizeN,errsizeH,stepN,stepH,deltaN,deltaH,cutoff,hSamples,verbose,seed):
    np.random.seed(seed)
    try:
        #Find one n matrix for this pair
        if verbose: print('starting n calculation',flush=True)
        n = sampleN(nStart,N,errsizeN,deltaN,stepN,verbose)

        #Now calculate a distribution of H from G1*S1 that does not have the cutoff applied
        if verbose: print('found n. Beginning V calculation',flush=True)
        hList = sampleH(H,n,hSamples,errsizeH,deltaH,stepH,cutoff,verbose,False)

        if hList is None:
            return 0,0,0,0,n

        #Using that list of H matrices, compute the integral for V(N)
        if verbose: print('hlist calculated for V. Beginning integral calculation',flush=True)
        V = calcWeightInt(np.array(hList),H,n,stepH,deltaH,errsizeH,cutoff)[0]

        #Now resample H matrices according to G1*S1 but now only take those that meet the cutoff
        if verbose: print('Calculated V. Beginning cutoff hList calculation,log(V)=',V,flush=True)
    #if np.exp(V) == 0.0:
    #    if verbose: print('V=0. Moving on.', flush=True)
    #    return 0,0,0,n
        hList = sampleH(H,n,hSamples,errsizeH,deltaH,stepH,cutoff,verbose,True)
        if verbose and hList is not None: print('Completed Cutoff list of H, V=',V,flush=True)
    #hList will be None iff sampleH() gave up on the N matrix, in which case we need to continue trying until we get a better N matrix.
    #The return code for this case is sd=0, but we still return n so that we continue the chain from this point.
        if hList is None:
            return 0,0,0,0,n
        sn = 0
        sno2 = 0
        sd = 0
    #Now compute the inner integral over H for the numerator and denominator
        for h in hList:
            volRatio,volFlag = hVolume(h.copy(),n.copy(),cutoff)
            if volRatio < cutoff and volFlag:
            #print('s1',s1(h,convergence,convergenceRatio,deltaH))
            #print('v',V)
                sn += 1/s1(volRatio,cutoff,deltaH)*O(h,n)
                sno2 += 1/s1(volRatio,cutoff,deltaH)*(O(h,n))**2
                sd += 1/s1(volRatio,cutoff,deltaH)
                #print(sd)
        #V = np.exp(V)
        if any(sn==0) or sd==0 or any(sno2==0):
            return 0,0,0,0,n


        logs0 = np.log(s0(n,deltaN))
        snSign = np.sign(sn)
    #sno2 and sd should always be positive, so only the sign of sn matters
        sn = np.log(sn*snSign)+V-logs0
        sd = np.log(sd)+V-logs0
        sno2 = np.log(sno2)+V-logs0
        #print(sd,'------------------------------------------------------')
        #print(V)
        #print(1/s0(n,deltaN))
        #sn = sn*V*1/s0(n,deltaN)
        #sd = sd*V*1/s0(n,deltaN)
        #sno2 = sno2*V*1/s0(n,deltaN)
        
    #return the value for this term in the numerator, denominator. Then return the n matrix so we continue the chain from here.
        #return sn,sd,sno2,n
        return sn,snSign,sd,sno2,n
    except:
        if verbose: print('Continuing from error:',sys.exc_info())
        return 0,0,0,0,n

def run(taskQueue,retQueue):
    t = taskQueue.get()
    while t is not None:
        H,n,N,O,errsizeN,errsizeH,stepN,stepH,deltaN,deltaH,cutoff,hSamples,verbose,seed,index = t
        sd = 0
        trials = 0
        while sd == 0:
            sn,snSign,sd,sno2,n = calcOnePair(H,n,N,O,errsizeN,errsizeH,stepN,stepH,deltaN,deltaH,cutoff,hSamples,verbose,seed+trials)
            trials = trials+1
        if verbose: print('completed task ',index,flush=True)
        retQueue.put((sn,snSign,sd,sno2))
        t = taskQueue.get()
    retQueue.put(None)



def stabilize(N,H,O=defaultOperator,numPairs=100,errsizeN = 0.01,errsizeH = 0.01,stepN=0.1,stepH=0.1,deltaN=1e-5,deltaH=1e-10,cutoff=-13,hSamples=100,autotune=True,verbose=False,seed=1):
    #Copy input matrices and lose the references for saftey
    N = N.copy()
    H = H.copy()

    order = H.shape[0]



    np.random.seed(seed)

    SN=0
    SD = 0
    SNO2 = 0
    n = N.copy()
        #Each n matrix term wants a random seed, and we'll pick those randomly. If we need to increment seeds for any reason, this will prevent overlap.
        #This ensures that when this code is multithreaded the walks will not repeat the same work.
    seeds = np.random.randint(0,2**31-1,size=numPairs)
        #loop over all the pairs we want
    
    if autotune==True:
        t1 = time.time()
        if verbose: print('Beginning Autotune Routine')
        deltaN,nSuccessRate = tuneN(N,errsizeN,deltaN,stepN)
        if verbose: print('Tuned deltaN value: ',deltaN)
        if verbose: print('Tuned Success Rate: ',nSuccessRate)
        try:
            deltaH,hSuccessRate = tuneH(H,N,errsizeH,deltaH,stepH,cutoff)

            if verbose: print('Tuned deltaH value: ',deltaH)
            if verbose: print('Tuned Success Rate: ',hSuccessRate,flush=True)
        except:
            if verbose: print('Error in autotune, reverting to default value,',sys.exc_info())
            
        t2 = time.time()
        if verbose: print('Autotune Time: ',t2-t1,flush=True)
    cores = mp.cpu_count()-2

    if verbose: print('running on '+str(cores)+' cores.')
    taskQueue = mp.Queue()
    retQueue = mp.Queue()


    for i in range(numPairs):
        taskQueue.put((H,n,N,O,errsizeN,errsizeH,stepN,stepH,deltaN,deltaH,cutoff,hSamples,verbose,seeds[i],i))
    for i in range(cores):
        taskQueue.put(None)
        
    processList = [mp.Process(target=run,args=(taskQueue,retQueue,)) for i in range(cores)]

    for p in processList:
        p.start()
    finProcess = 0
    SN = []
    SNO2 = []
    SD = []
    SNSign = []
    while finProcess < cores:
        result = retQueue.get()
        if result is not None:
            SN.append(result[0])
            SNSign.append(result[1])
            SD.append(result[2])
            SNO2.append(result[3])
        else:
            finProcess += 1
    
    for p in processList:
        p.join()

#    print('------------------------------')
#    print(SN)
#    print('------------------------------')



    SN,sign = logsumexp(SN,axis=0,b=SNSign,return_sign=True)

    SNO2 = logsumexp(SNO2,axis=0)
    SD = logsumexp(SD,axis=0)


    #K = np.max(SN)
    #SN = np.array(SN)-K
    #SN = np.sum(np.array(SNSign)*np.exp(SN))
    #sign = np.sign(SN)
    #SN = np.log(SN*sign)+K

    #SD = np.array(SD)
    #K = np.max(SD)
    #SD = SD-K
    #SD = np.sum(np.exp(SD))
    #SD = np.log(SD)+K

    #SNO2 = np.array(SNO2)
    #K = np.max(SNO2)
    #SNO2 = SNO2-K
    #SNO2 = np.sum(np.exp(SNO2))
    #SNO2 = np.log(SNO2)+K

    #SN = sum(SN)
    #SD = sum(SD)
    #SNO2=sum(SNO2)
    #print(SD)

    logO = SN-SD
    retO = sign*np.exp(logO)

    retO2 = np.exp(SNO2-SD)-retO**2
    
    return retO,retO2
    return np.log(np.abs(retO)),np.log(retO2)
    #return SN/SD,SNO2/SD-(SN/SD)**2


