"""
Module containing different metric for computing summay statistics (autocrrelation or power spectrum).
"""

import numpy as np
from scipy import stats


def comp_sumStat(data, summStat_metric, ifNorm, deltaT, binSize, T, numBin, maxTimeLag = None):
    """Compute summary statistics for a given metric.

    Parameters
    -----------
    data : nd array
        time-series from binned data (numTrials * numBin).
    summStat_metric : string
        metric for computing summay statistics ('comp_cc', 'comp_ac_fft', 'comp_psd').
    ifNorm : string
        if normalize the autocorrelation to the zero time-lag or PSD to have integral equal to 1.
    deltaT : float
        temporal resolution of data (or binSize of spike counts).    
    maxTimeLag : float
        maximum time-lag for computing cross- or auto-correlation.    
    binSize : float
        bin-size used for binning data.
    T : float
        duration of each trial/time-series.
    numBin : int
        number of time-bins in each trial of x1 and x2.
    maxTimeLag : float, default None
        maximum time-lag for computing autocorrelation. 
    
    
    Returns
    -------
    ac : 1d array
        average non-normalized cross- or auto-correlation across all trials.
    """
    if summStat_metric == 'comp_cc':
        # compute ac time domain
        sumStat = comp_cc(data, data, maxTimeLag, binSize, numBin)            
        if ifNorm:
            # normalize autocorrelation (optional depending on the application)
            sumStat = sumStat/sumStat[0]
    elif summStat_metric == 'comp_ac_fft': 
        # compute ac fft
        sumStat = comp_ac_fft(data)
        lm = round(maxTimeLag/binSize)
        sumStat = sumStat[0:lm]
        if ifNorm:
            # normalize autocorrelation (optional depending on the application)
            sumStat = sumStat/sumStat[0]
    elif summStat_metric == 'comp_psd': 
        # compute psd
        sumStat = comp_psd(data, T, deltaT)
        if ifNorm:
            # normalize the psd to not to deal with large numbers
            sumStat = sumStat/np.sum(sumStat)
    else:
        raise Exception("unknown summary statistics")
        sumStat = None 
    
    return sumStat

          


def comp_cc_2(x1, x2, maxTimeLag, binSize, numBin):
    """Compute cross- or auto-correlation from binned data (without normalization).

    Parameters
    -----------
    x1, x2 : nd array
        time-series from binned data (numTrials * numBin).
    D : float
        diffusion parameter.
    maxTimeLag : float
        maximum time-lag for computing cross- or auto-correlation.    
    binSize : float
        bin-size used for binning x1 and x2.
    numBin : int
        number of time-bins in each trial of x1 and x2.
    
    
    Returns
    -------
    ac : 1d array
        average non-normalized cross- or auto-correlation across all trials.
    """
    
    numTr1 = np.shape(x1)[0]
    numTr2 = np.shape(x2)[0]
    
    if numTr1 != numTr2:
        raise Exception('numTr1 != numTr2')
    
    numBinLag = int(np.ceil( (maxTimeLag)/binSize )+1)
    ac = np.zeros((numBinLag))
    for tr in range(numTr1):
        xt1 = x1[tr]
        xt2 = x2[tr]
        for iLag in  range(0,numBinLag):            
            ind1 = np.arange(np.max([0,-iLag]),np.min([numBin-iLag,numBin]))  # index to take this part of the array 1
            ind2 = np.arange(np.max([0,iLag]),np.min([numBin+iLag,numBin]))  # index to take this part of the array 2
            ac[iLag] = ac[iLag]+(np.dot(xt1[ind1],xt2[ind2])/(len(ind1))-np.mean(xt1[ind1])*np.mean(xt2[ind2]))
        
    return ac/numTr1


def comp_cc(x1, x2, maxTimeLag, binSize, numBin):
    """Compute cross- or auto-correlation from binned data (without normalization).
    Uses matrix computations to speed up, preferred when multiple processors are available.

    Parameters
    -----------
    x1, x2 : nd array
        time-series from binned data (numTrials * numBin).
    D : float
        diffusion parameter.
    maxTimeLag : float
        maximum time-lag for computing cross- or auto-correlation.    
    binSize : float
        bin-size used for binning x1 and x2.
    numBin : int
        number of time-bins in each trial of x1 and x2.
    
    
    Returns
    -------
    ac : 1d array
        average non-normalized cross- or auto-correlation across all trials.
    """
    
    numBinLag = int(np.ceil( (maxTimeLag)/binSize )+1)-1
    ac = np.zeros((numBinLag))
    for iLag in  range(0,numBinLag):            
        ind1 = np.arange(np.max([0,-iLag]),np.min([numBin-iLag,numBin]))  # index to take this part of the array 1
        ind2 = np.arange(np.max([0,iLag]),np.min([numBin+iLag,numBin]))  # index to take this part of the array 2

        cov_trs = np.sum((x1[:, ind1] * x2[:, ind2]),axis = 1)/len(ind1)
        ac[iLag] = np.mean(cov_trs - np.mean(x1[:, ind1] , axis =1) * np.mean(x2[:, ind2] , axis =1)) 
        
    return ac

def comp_ac_fft_middlepad(data):
    """Compute auto-correlations from binned data (without normalization).
    Uses FFT after zero-padding the time-series in the middle.

    Parameters
    -----------
    data : nd array
        time-series from binned data (numTrials * numBin).
        
    Returns
    -------
    ac : 1d array
        average non-normalized auto-correlation across all trials.
    """
    numTrials = np.shape(data)[0]
    ac_sum = 0
    for tr in range(numTrials):
        x = data[tr]
        xp = np.fft.ifftshift((x - np.average(x)))
        n,  = xp.shape
        xp = np.r_[xp[:n//2], np.zeros_like(xp), xp[n//2:]]
        f = np.fft.fft(xp)
        p = np.absolute(f)**2
        pi = np.fft.ifft(p)
        ac = np.real(pi)[:n-1]/np.arange(1,n)[::-1]
        ac_sum = ac_sum + ac
    ac = ac_sum/numTrials
    return ac

def comp_ac_fft_middlepad_zscore(data):
    """Compute auto-correlations from binned data (without normalization).
    Uses FFT after z-scoring and zero-padding the time-series in the middle.

    Parameters
    -----------
    data : nd array
        time-series from binned data (numTrials * numBin).
        
    Returns
    -------
    ac : 1d array
        average non-normalized auto-correlation across all trials.
    """
    numTrials = np.shape(data)[0]
    ac_sum = 0
    for tr in range(numTrials):
        x = data[tr]
        xp = np.fft.ifftshift((x - np.average(x))/np.std(x))
        n,  = xp.shape
        xp = np.r_[xp[:n//2], np.zeros_like(xp), xp[n//2:]]
        f = np.fft.fft(xp)
        p = np.absolute(f)**2
        pi = np.fft.ifft(p)
        ac = np.real(pi)[:n-1]/np.arange(1,n)[::-1]
        ac_sum = ac_sum + ac
    ac = ac_sum/numTrials
    return ac



def comp_ac_fft(data):
    """Compute auto-correlations from binned data (without normalization).
    Uses FFT after zero-padding the time-series in the right side.

    Parameters
    -----------
    data : nd array
        time-series from binned data (numTrials * numBin).
        
    Returns
    -------
    ac : 1d array
        average non-normalized auto-correlation across all trials.
    """
    n = np.shape(data)[1]
    xp = data - data.mean(1)[:,None]
    xp = np.concatenate((xp,  np.zeros_like(xp)), axis = 1)
    f = np.fft.fft(xp)
    p = np.absolute(f)**2
    pi = np.fft.ifft(p)
    ac_all = np.real(pi)[:, :n-1]/np.arange(1,n)[::-1]
    ac = np.mean(ac_all, axis = 0)  
    return ac


def comp_psd(x, T, deltaT):
    """Compute the power spectrum density (PSD) using a Hamming window and direct fft.

    Parameters
    -----------
    x1 : nd array
        time-series from binned data (numTrials * numBin).
    T : float
        duration of each trial/time-series.
    deltaT : float
        temporal resolution of data (or binSize of spike counts).    
    
    
    Returns
    -------
    psd : 1d array
        average  power spectrum density (PSD) across all trials.
    """
    fs = T/deltaT
    n_points = len(x[0])
    x_windowed = (x - x.mean(1)[:,None])*np.hamming(n_points)
    PSD = np.mean(np.abs(np.fft.rfft(x_windowed))**2, axis = 0)[1:-1]
    
    return PSD
