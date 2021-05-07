"""
Module containing basic functions for OU generation, binning, estimating MAPs, spike generation methods, etc.
"""

import numpy as np
from scipy import stats


def OU_gen(tau, D, deltaT, T, numTrials):
    """Generate an OU process with a single timescale, zero mean and unit variance.

    Parameters
    -----------
    tau : float
        timescale.
    D : float
        diffusion parameter.
    deltaT : float
        temporal resolution for the OU process generation.    
    T : float
        duration of trials.
    numTrials : float
        number of trials.
    
    
    Returns
    -------
    ou : nd array
        array of generated OU process (numTrials * (T/deltaT)).
    """
    
    numBin = int(T/deltaT)
    noise =  np.random.normal(loc=0,scale=1, size=(numTrials,numBin))
    ou = np.zeros((numTrials,numBin))
    ou[:,0] = noise[:,0]
    for iBin in range(1,numBin):
        ou[:,iBin]  = ou[:,iBin-1] - (ou[:,iBin-1]/tau) * deltaT + np.sqrt(2*D*deltaT) * noise[:,iBin-1]
        
    return ou




def gamma_sp(rate, disp):
    """Generate spike counts from a gamma distribution.

    Parameters
    -----------
    rate : nd array
        instantaneous rate (numTrials * numBin).
    disp : float
        disperssion parameter (fano factor) of spike generation function.
    
    
    Returns
    -------
    spCounts : nd array
        spike counts (numTrials * numBin).
    """
    m = rate
    v = disp * rate
    theta = v/m
    k = m**2/v
    where_are_NaNs = np.isnan(k)
    k[where_are_NaNs] = 1
    theta[where_are_NaNs] = 1
    spCounts = np.random.gamma(shape = k, scale=theta)
    spCounts[where_are_NaNs] = 0
    return spCounts


def gaussian_sp(rate, disp):
    """Generate spike counts from a Gaussian distribution.

    Parameters
    -----------
    rate : nd array
        instantaneous rate (numTrials * numBin).
    disp : float
        disperssion parameter (fano factor) of spike generation function.
    
    
    Returns
    -------
    spCounts : nd array
        spike counts (numTrials * numBin).
    """
    
    spCounts = np.zeros(rate.shape)
    for tr in range(len(rate)):
        spCounts[tr] = np.random.normal(loc = rate[tr],scale = np.sqrt(disp * rate[tr]))
    return spCounts



def binData(data, new_shape):
    """bin time-series.

    Parameters
    -----------
    data : nd array
        time-series (numTrials * time-points).
    new_shape : 1d array
        [numTrials, numBin]
    
    Returns
    -------
    binned_data : nd array
        binned time-series (numTrials * numBin).
    """
    
    shape = (new_shape[0], data.shape[0] // new_shape[0],
             new_shape[1], data.shape[1] // new_shape[1])
    binned_data = data.reshape(shape).sum(-1).sum(1)
    return binned_data






def generateInhomPoisson_Thinning(rate, deltaT, T):
    """Generate Inhomogeous Poisson process using thinning method.

    Parameters
    -----------
    rate : nd array
        time-series for instanteneous rate (numTrials * numBin).
    deltaT : float
        temporal resolution for the OU process generation.    
    T : float
        duration of trials.
    
    
    Returns
    -------
    spikeTrain_inhom : nd array
        spike trians.
    """
    # generate homPois with rate rmax for each trial ( we used bernoulli approximation of Pois)
    r_max = np.max(rate, axis=1)
    SF = 1/deltaT
    numSamples = np.shape(rate)[1]
    numTrials = np.shape(rate)[0]
    repeated_rmax = np.transpose(npmt.repmat(r_max, numSamples, 1))
    probThrslds = repeated_rmax/SF
    spikeTrain_hom = (np.random.rand(numTrials,numSamples)<probThrslds).astype(int)
    
    # create rejection matrix
    rejectMat = ((rate/repeated_rmax) > np.random.rand(numTrials,numSamples)).astype(int)
    
    #create inhom pois
    spikeTrain_inhom = rejectMat * spikeTrain_hom 
    return spikeTrain_inhom 


def find_MAP(theta_accepted, N):
    """Find the MAP estimates from posteriors with grid search.

    Parameters
    -----------
    theta_accepted : nd array
        array of accepted samples from the final step of the ABC: pmc_posterior[final_step - 1]['theta accepted']
    N : float
        number samples for grid search.    
    
    
    Returns
    -------
    theta_map : 1d array
        MAP estimates of the parameters.
    """
    
    numParams = len(theta_accepted)
    kernel = stats.gaussian_kde(theta_accepted)

    positions = []
    for i in range(numParams):
        param = theta_accepted[i]
        positions.append(np.random.uniform(np.min(param),np.max(param),N))

    positions = np.array(positions)
    probs = kernel(positions)
    theta_map = positions[:,np.where(probs == np.max(probs))[0][0]] 
    return theta_map


def double_exp(time, a, tau1, tau2, coeff):
    """a double expoenetial decay function.

    Parameters
    -----------
    time : 1d array
        time points.
    a : float
        amplitude of autocorrelation at lag 0. 
    tau1 : float
       first timescale.
    tau2 : float
       second timescale.
    coeff: float
        weight of the first timescale between [0,1]
    
    
    Returns
    -------
    exp_func : 1d array
        double expoenetial decay function.
    """
    exp_func = a * (coeff) * np.exp(-time/tau1) + a * (1-coeff) * np.exp(-time/tau2)
    return  exp_func

def single_exp(time, a, tau):
    """a single expoenetial decay function.

    Parameters
    -----------
    time : 1d array
        time points.
    a : float
        amplitude of autocorrelation at lag 0. 
    tau : float
       timescale.
    
    
    Returns
    -------
    exp_func : 1d array
        single expoenetial decay function.
    """
    exp_func = a * np.exp(-time/tau) 
    return exp_func

