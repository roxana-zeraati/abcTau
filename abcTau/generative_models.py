"""
Module containing different generative models 
"""

import numpy as np
from scipy import stats
from basic_functions import *



def oneTauOU(theta, deltaT, binSize, T, numTrials, data_mean, data_var):
    """Generate an OU process with a single timescale.

    Parameters
    -----------
    theta : 1d array
        [timescale].
    deltaT : float
        temporal resolution for OU process generation.
    binSize : float
        bin-size for binning data and computing the autocorrelation.
    T : float
        duration of trials.
    numTrials : float
        number of trials.
    data_mean : float
        mean value of the OU process (average of firing rate). 
    data_var : float
        variance of the OU process (variance of firing rate).

    Returns
    -------
    syn_data : nd array
        array of generated OU process (numTrials * int(T/binSize)).
    numBinData : int
        number of bins/samples per trial (required for computing autocorrelation).

    """
    
    # load params
    tau = np.array(theta[0])
    
    # setting params for OU
    v = 1
    D = v/tau
    binsData =  np.arange(0, T + binSize, binSize)
    numBinData = len(binsData)-1
    
    # generate OU
    ou_all = OU_gen(tau,D,deltaT,T,numTrials)
    ou_check = np.max(ou_all)
    if not np.isfinite(ou_check) or ou_check>10**10: # check for nan values
        return np.zeros((numTrials,numBinData)) , numBinData
    
    # fit mean and var
    ou_std = np.sqrt(data_var)
    ou_all = ou_std * ou_all + data_mean
    
    # bin rate 
    syn_data = binData(ou_all, [numTrials,numBinData]) * deltaT
    return syn_data, numBinData


    

def twoTauOU(theta, deltaT, binSize, T, numTrials, data_mean, data_var):
    """Generate a two-timescales OU process.
    
    Parameters
    -----------
    theta : 1d array
        [timescale1, timescale2, coefficient for timescale1].
    deltaT : float
        temporal resolution for the OU process generation.
    binSize : float
        bin-size for binning data and computing the autocorrelation.
    T : float
        duration of trials.
    numTrials : float
        number of trials.
    data_mean : float
        mean value of the OU process (average of firing rate). 
    data_var : float
        variance of the OU process (variance of firing rate). 

    Returns
    -------
    syn_data : nd array
        array of binned spike-counts (numTrials * int(T/binSize)).
    numBinData : int
        number of bins/samples per trial (required for computing autocorrelation).

    """
    # load params
    tau1 = np.array(theta[0])
    tau2 = np.array(theta[1])
    coeff = np.array(theta[2])
    
    # setting the params of OU
    v = 1
    D1 = v/tau1
    D2 = v/tau2     
    binsData =  np.arange(0, T + binSize, binSize)
    numBinData = len(binsData)-1
    
    # generate OU
    ou_all1 = OU_gen(tau1,D1,deltaT,T,numTrials)
    ou_all2 = OU_gen(tau2,D2,deltaT,T,numTrials)
    ou_all = np.sqrt(coeff) * ou_all1 + np.sqrt(1 - coeff) * ou_all2
    ou_check = np.max(ou_all)
    if not np.isfinite(ou_check) or ou_check>10**10: # check for all-nan values
        return np.zeros((numTrials,numBinData)), numBinData
    
    # fit mean and var
    ou_std = np.sqrt(data_var)
    ou_all = ou_std * ou_all + data_mean
    
    # bin rate 
    syn_data = binData(ou_all, [numTrials,numBinData]) * deltaT
    return syn_data, numBinData


def oneTauOU_oscil(theta, deltaT, binSize, T, numTrials, data_mean, data_var):
    """Generate a one-timescale OU process with an additive oscillation.
    
    Parameters
    -----------
    theta : 1d array
        [timescale of OU, frequency of oscillation, coefficient for OU].
    deltaT : float
        temporal resolution for the OU process generation.
    binSize : float
        bin-size for binning data and computing the autocorrelation.
    T : float
        duration of trials.
    numTrials : float
        number of trials.
    data_mean : float
        mean value of the OU process (average of firing rate). 
    data_var : float
        variance of the OU process (variance of firing rate). 

    Returns
    -------
    syn_data : nd array
        array of binned spike-counts (numTrials * int(T/binSize)).
    numBinData : int
        number of bins/samples per trial (required for computing autocorrelation).

    """
    
    # load params
    tau = np.array(theta[0])
    f = np.array(theta[1])
    coeff = np.array(theta[2])
    
    # setting params for OU
    v = 1
    D = v/tau
    binsData =  np.arange(0, T + binSize, binSize)
    binsData_sin = np.arange(0, T, deltaT )
    numBinData = len(binsData)-1

    # generate OU + oscil
    ou_all = OU_gen(tau, D, deltaT, T, numTrials)
    time_mat = np.tile(binsData_sin, (numTrials,1))
    phases = np.random.rand(numTrials,1)* 2 * np.pi
    oscil = np.sqrt(2)*np.sin(phases + 2*np.pi*0.001*f* time_mat)
    data = np.sqrt(1 - coeff) * oscil + np.sqrt(coeff) * ou_all
    
    # fit mean and var
    ou_std = np.sqrt(data_var)
    data_meanVar = ou_std * data + data_mean
    
    # bin rate 
    syn_data = binData(data_meanVar, [numTrials,numBinData]) * deltaT
    return syn_data, numBinData



def oneTauOU_poissonSpikes(theta, deltaT, binSize, T, numTrials, data_mean, data_var):
    """Generate a one-timescale process with spike counts sampled from a Gaussian distribution.
    Assuming that disperssion parameter (fano factor) of spike generation function is known.

    Parameters
    -----------
    theta : 1d array
        [timescale].
    deltaT : float
        temporal resolution for the OU process generation.
    binSize : float
        bin-size for binning data and computing the autocorrelation.
    T : float
        duration of trials.
    numTrials : float
        number of trials.
    data_mean : float
        mean of the spike counts. 
    data_var : float
        variance of the spike counts.
    

    Returns
    -------
    syn_data : nd array
        array of binned spike-counts (numTrials * int(T/binSize)).
    numBinData : int
        number of bins/samples per trial (required for computing autocorrelation).

    """
    # load params
    tau = np.array(theta[0])    
    
    # setting the params of OU
    v = 1
    D = v/tau
    
    ou_std =  np.sqrt(data_var - data_mean)# law of total variance      
    binsData =  np.arange(0, T + binSize, binSize)
    numBinData = len(binsData)-1
    
    # generate OU
    ou_all = OU_gen(tau, D, deltaT, T, numTrials)
   
    ou_check = np.max(ou_all)
    if not np.isfinite(ou_check) or ou_check>10**10: # check for all-nan values
        return np.zeros((numTrials,numBinData)), numBinData
    
    # fit mean and var
    ou_all = ou_std * ou_all + data_mean
    ou_all[ou_all < 0] = 0
    
    # bin rate and generate spikes
    rate_sum = binData(ou_all, [numTrials,numBinData]) * deltaT
    syn_data = np.random.poisson(rate_sum)
    return syn_data, numBinData



def oneTauOU_gammaSpikes(theta, deltaT, binSize, T, numTrials, data_mean, data_var, disp):
    """Generate a one-timescale process with spike counts sampled from a Gamma distribution.
    Assuming that disperssion parameter (fano factor) of spike generation function is known.

    Parameters
    -----------
    theta : 1d array
        [timescale].
    deltaT : float
        temporal resolution for the OU process generation.
    binSize : float
        bin-size for binning data and computing the autocorrelation.
    T : float
        duration of trials.
    numTrials : float
        number of trials.
    data_mean : float
        mean of the spike counts. 
    data_var : float
        variance of the spike counts.
    disp : float
        disperssion parameter (fano factor) of spike generation function.

    Returns
    -------
    syn_data : nd array
        array of binned spike-counts (numTrials * int(T/binSize)).
    numBinData : int
        number of bins/samples per trial (required for computing autocorrelation).

    """
    # load params
    tau = np.array(theta[0])    
    
    # setting the params of OU
    v = 1
    D = v/tau
    
    ou_std =  np.sqrt(data_var - disp*data_mean)# law of total variance      
    binsData =  np.arange(0, T + binSize, binSize)
    numBinData = len(binsData)-1
    
    # generate OU
    ou_all = OU_gen(tau, D, deltaT, T, numTrials)
   
    ou_check = np.max(ou_all)
    if not np.isfinite(ou_check) or ou_check>10**10: # check for all-nan values
        return np.zeros((numTrials,numBinData)), numBinData
    
    # fit mean and var
    ou_all = ou_std * ou_all + data_mean
    ou_all[ou_all < 0] = 0
    
    # bin rate and generate spikes
    rate_sum = binData(ou_all, [numTrials,numBinData]) * deltaT
    syn_data = gamma_sp(rate_sum,disp)
    return syn_data, numBinData


def oneTauOU_gaussianSpikes(theta, deltaT, binSize, T, numTrials, data_mean, data_var, disp):
    """Generate a one-timescale process with spike counts sampled from a Gaussian distribution.
    Assuming that disperssion parameter (fano factor) of spike generation function is known.

    Parameters
    -----------
    theta : 1d array
        [timescale].
    deltaT : float
        temporal resolution for the OU process generation.
    binSize : float
        bin-size for binning data and computing the autocorrelation.
    T : float
        duration of trials.
    numTrials : float
        number of trials.
    data_mean : float
        mean of the spike counts. 
    data_var : float
        variance of the spike counts.
    disp : float
        disperssion parameter (fano factor) of spike generation function.

    Returns
    -------
    syn_data : nd array
        array of binned spike-counts (numTrials * int(T/binSize)).
    numBinData : int
        number of bins/samples per trial (required for computing autocorrelation).

    """
    # load params
    tau = np.array(theta[0])    
    
    # setting the params of OU
    v = 1
    D = v/tau
    
    ou_std =  np.sqrt(data_var - disp*data_mean)# law of total variance      
    binsData =  np.arange(0, T + binSize, binSize)
    numBinData = len(binsData)-1
    
    # generate OU
    ou_all = OU_gen(tau, D, deltaT, T, numTrials)
   
    ou_check = np.max(ou_all)
    if not np.isfinite(ou_check) or ou_check>10**10: # check for all-nan values
        return np.zeros((numTrials,numBinData)), numBinData
    
    # fit mean and var
    ou_all = ou_std * ou_all + data_mean
    ou_all[ou_all < 0] = 0
    
    # bin rate and generate spikes
    rate_sum = binData(ou_all, [numTrials,numBinData]) * deltaT
    syn_data = gaussian_sp(rate_sum, disp)
    return syn_data, numBinData


def twoTauOU_poissonSpikes(theta, deltaT, binSize, T, numTrials, data_mean, data_var):
    """Generate a two-timescales process with spike counts sampled from a Poisson distribution.
    
    Parameters
    -----------
    theta : 1d array
        [timescale1, timescale2, coefficient for timescale1].
    deltaT : float
        temporal resolution for the OU process generation.
    binSize : float
        bin-size for binning data and computing the autocorrelation.
    T : float
        duration of trials.
    numTrials : float
        number of trials.
    data_mean : float
        mean of the spike counts. 
    data_var : float
        variance of the spike counts. 

    Returns
    -------
    syn_data : nd array
        array of binned spike-counts (numTrials * int(T/binSize)).
    numBinData : int
        number of bins/samples per trial (required for computing autocorrelation).

    """
    # load params
    tau1 = np.array(theta[0])
    tau2 = np.array(theta[1])
    coeff = np.array(theta[2])
    
    # setting the params of OU
    v = 1
    D1 = v/tau1
    D2 = v/tau2
    ou_std =  np.sqrt(data_var - data_mean)# law of total variance      
    binsData =  np.arange(0, T + binSize, binSize)
    numBinData = len(binsData)-1
    
    # generate OU
    ou_all1 = OU_gen(tau1,D1,deltaT,T,numTrials)
    ou_all2 = OU_gen(tau2,D2,deltaT,T,numTrials)
    ou_all = np.sqrt(coeff) * ou_all1 + np.sqrt(1 - coeff) * ou_all2
    ou_check = np.max(ou_all)
    if not np.isfinite(ou_check) or ou_check>10**10: # check for all-nan values
        return np.zeros((numTrials,numBinData)), numBinData
    
    # fit mean and var
    ou_all = ou_std * ou_all + data_mean
    ou_all[ou_all < 0] = 0
    
    # bin rate and generate spikes
    rate_sum = binData(ou_all, [numTrials,numBinData]) * deltaT
    syn_data = np.random.poisson(rate_sum)
    return syn_data, numBinData
    

def twoTauOU_gammaSpikes(theta, deltaT, binSize, T, numTrials, data_mean, data_var, disp):
    """Generate a two-timescales process with spike counts sampled from a Gamma distribution.
    Assuming that disperssion parameter (fano factor) of spike generation function is known.

    Parameters
    -----------
    theta : 1d array
        [timescale1, timescale2, coefficient for timescale1].
    deltaT : float
        temporal resolution for the OU process generation.
    binSize : float
        bin-size for binning data and computing the autocorrelation.
    T : float
        duration of trials.
    numTrials : float
        number of trials.
    data_mean : float
        mean of the spike counts. 
    data_var : float
        variance of the spike counts.
    disp : float
        disperssion parameter (fano factor) of spike generation function.

    Returns
    -------
    syn_data : nd array
        array of binned spike-counts (numTrials * int(T/binSize)).
    numBinData : int
        number of bins/samples per trial (required for computing autocorrelation).

    """
    # load params
    tau1 = np.array(theta[0])
    tau2 = np.array(theta[1])
    coeff = np.array(theta[2])
    
    # setting the params of OU
    v = 1
    D1 = v/tau1
    D2 = v/tau2
    ou_std =  np.sqrt(data_var - disp*data_mean)# law of total variance      
    binsData =  np.arange(0, T + binSize, binSize)
    numBinData = len(binsData)-1
    
    # generate OU
    ou_all1 = OU_gen(tau1,D1,deltaT,T,numTrials)
    ou_all2 = OU_gen(tau2,D2,deltaT,T,numTrials)
    ou_all = np.sqrt(coeff) * ou_all1 + np.sqrt(1 - coeff) * ou_all2
    ou_check = np.max(ou_all)
    if not np.isfinite(ou_check) or ou_check>10**10: # check for all-nan values
        return np.zeros((numTrials,numBinData)), numBinData
    
    # fit mean and var
    ou_all = ou_std * ou_all + data_mean
    ou_all[ou_all < 0] = 0
    
    # bin rate and generate spikes
    rate_sum = binData(ou_all, [numTrials,numBinData]) * deltaT
    syn_data = gamma_sp(rate_sum, disp)
    return syn_data, numBinData


def twoTauOU_gaussianSpikes(theta, deltaT, binSize, T, numTrials, data_mean, data_var, disp):
    """Generate a two-timescales process with spike counts sampled from a Guassion distribution.
    Assuming that disperssion parameter (fano factor) of spike generation function is known.

    Parameters
    -----------
    theta : 1d array
        [timescale1, timescale2, coefficient for timescale1].
    deltaT : float
        temporal resolution for the OU process generation.
    binSize : float
        bin-size for binning data and computing the autocorrelation.
    T : float
        duration of trials.
    numTrials : float
        number of trials.
    data_mean : float
        mean of the spike counts. 
    data_var : float
        variance of the spike counts.
    disp : float
        disperssion parameter (fano factor) of spike generation function.

    Returns
    -------
    syn_data : nd array
        array of binned spike-counts (numTrials * int(T/binSize)).
    numBinData : int
        number of bins/samples per trial (required for computing autocorrelation).

    """
    # load params
    tau1 = np.array(theta[0])
    tau2 = np.array(theta[1])
    coeff = np.array(theta[2])
    
    # setting the params of OU
    v = 1
    D1 = v/tau1
    D2 = v/tau2
    ou_std =  np.sqrt(data_var - disp*data_mean)# law of total variance      
    binsData =  np.arange(0, T + binSize, binSize)
    numBinData = len(binsData)-1
    
    # generate OU
    ou_all1 = OU_gen(tau1,D1,deltaT,T,numTrials)
    ou_all2 = OU_gen(tau2,D2,deltaT,T,numTrials)
    ou_all = np.sqrt(coeff) * ou_all1 + np.sqrt(1 - coeff) * ou_all2
    ou_check = np.max(ou_all)
    if not np.isfinite(ou_check) or ou_check>10**10: # check for all-nan values
        return np.zeros((numTrials,numBinData)), numBinData
    
    # fit mean and var
    ou_all = ou_std * ou_all + data_mean
    ou_all[ou_all < 0] = 0
    
    # bin rate and generate spikes
    rate_sum = binData(ou_all, [numTrials,numBinData]) * deltaT
    syn_data = gaussian_sp(rate_sum,disp)
    return syn_data, numBinData


    
def twoTauOU_gammaSpikes_withDispersion(theta, deltaT, binSize, T, numTrials, data_mean, data_var): 
    """Generate a two-timescales process with spike counts sampled from a Gamma distribution.
    disperssion parameter (fano factor) of spike generation function is fitted with ABC.

    Parameters
    -----------
    theta : 1d array
        [timescale1, timescale2, coefficient for timescale1, disperssion_parameter].
    deltaT : float
        temporal resolution for the OU process generation.
    binSize : float
        bin-size for binning data and computing the autocorrelation.
    T : float
        duration of trials.
    numTrials : float
        number of trials.
    data_mean : float
        mean of the spike counts. 
    data_var : float
        variance of the spike counts.
    
    Returns
    -------
    syn_data : nd array
        array of binned spike-counts (numTrials * int(T/binSize)).
    numBinData : int
        number of bins/samples per trial (required for computing autocorrelation).

    """
    
    # load params
    tau1 = np.array(theta[0])
    tau2 = np.array(theta[1])
    coeff = np.array(theta[2])
    disp = np.array(theta[3])
    
    # setting the params of OU
    v = 1
    D1 = v/tau1
    D2 = v/tau2
    ou_std =  np.sqrt(data_var - disp*data_mean) # law of total variance       
    binsData =  np.arange(0, T + binSize, binSize)
    numBinData = len(binsData)-1
    
    # generate OU
    ou_all1 = OU_gen(tau1,D1,deltaT,T,numTrials)
    ou_all2 = OU_gen(tau2,D2,deltaT,T,numTrials)
    ou_all = np.sqrt(coeff) * ou_all1 + np.sqrt(1 - coeff) * ou_all2
    ou_check = np.max(ou_all)
    if not np.isfinite(ou_check) or ou_check>10**10: # check for all-nan values
        return np.zeros((numTrials,numBinData)), numBinData
    
    # fit mean and var
    ou_all = ou_std * ou_all + data_mean
    ou_all[ou_all < 0] = 0
    
    # bin rate and generate spikes
    rate_sum = binData(ou_all, [numTrials,numBinData]) * deltaT
    syn_data = gamma_sp(rate_sum,disp)
    return syn_data, numBinData
    
    
def twoTauOU_gaussianSpikes_withDispersion(theta, deltaT, binSize, T, numTrials, data_mean, data_var): 
    """Generate a two-timescales process with spike counts sampled from a Gamma distribution.
    disperssion parameter (fano factor) of spike generation function is fitted with ABC.

    Parameters
    -----------
    theta : 1d array
        [timescale1, timescale2, coefficient for timescale1, disperssion_parameter].
    deltaT : float
        temporal resolution for the OU process generation.
    binSize : float
        bin-size for binning data and computing the autocorrelation.
    T : float
        duration of trials.
    numTrials : float
        number of trials.
    data_mean : float
        mean of the spike counts. 
    data_var : float
        variance of the spike counts.
    
    Returns
    -------
    syn_data : nd array
        array of binned spike-counts (numTrials * int(T/binSize)).
    numBinData : int
        number of bins/samples per trial (required for computing autocorrelation).

    """
    
    # load params
    tau1 = np.array(theta[0])
    tau2 = np.array(theta[1])
    coeff = np.array(theta[2])
    disp = np.array(theta[3])
    
    # setting the params of OU
    v = 1
    D1 = v/tau1
    D2 = v/tau2
    ou_std =  np.sqrt(data_var - disp*data_mean) # law of total variance       
    binsData =  np.arange(0, T + binSize, binSize)
    numBinData = len(binsData)-1
    
    # generate OU
    ou_all1 = OU_gen(tau1,D1,deltaT,T,numTrials)
    ou_all2 = OU_gen(tau2,D2,deltaT,T,numTrials)
    ou_all = np.sqrt(coeff) * ou_all1 + np.sqrt(1 - coeff) * ou_all2
    ou_check = np.max(ou_all)
    if not np.isfinite(ou_check) or ou_check>10**10: # check for all-nan values
        return np.zeros((numTrials,numBinData)), numBinData
    
    # fit mean and var
    ou_all = ou_std * ou_all + data_mean
    ou_all[ou_all < 0] = 0
    
    # bin rate and generate spikes
    rate_sum = binData(ou_all, [numTrials,numBinData]) * deltaT
    syn_data = gaussian_sp(rate_sum,disp)
    return syn_data, numBinData
 