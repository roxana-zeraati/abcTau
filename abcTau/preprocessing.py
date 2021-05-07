"""
Module for extracting statistics from data and running the preprocessing function.
"""


import numpy as np
from scipy.optimize import curve_fit
import scipy.stats as stats
from generative_models import *
from basic_functions import *
from distance_functions import *
from summary_stats import *




def extract_stats(data, deltaT, binSize, summStat_metric, ifNorm, maxTimeLag = None):
    """Extract required statistics from data for ABC fitting.
    
    Parameters
    -----------
    data : nd array
        time-series of continous data, e.g., OU process, (numTrials * numTimePoints)
        or binned spike counts (numTrials * numBin).
    deltaT : float
        temporal resolution of data (or binSize of spike counts).
    binSize : float
        bin-size for computing the autocorrelation.    
    summStat : string
        metric for computing summay statistics ('comp_cc', 'comp_ac_fft', 'comp_psd').
    ifNorm : string
        if normalize the autocorrelation or PSD.
    maxTimeLag : float, default None
        maximum time-lag when using 'comp_cc' summary statistics.
    

    Returns
    -------
    sumStat_nonNorm : 1d array
        non normalized autocorrelation or PSD depending on chosen summary statistics.
    data_mean : float
        mean of data (for one unit of time)
    data_mean : float
        variance of data (for one unit of time)
    T : float
        duration of each trial in data
    numTrials : float
        number of trials in data
    binSize : float
        bin-size used for computing the autocorrlation.
    maxTimeLag : float
        maximum time-lag used for computing the autocorrelation.

    """
    
    # extract duration and number of trials
    numTrials, numTimePoints = np.shape(data)
    T = numTimePoints* deltaT

    # compute mean and variance for one unit of time (1 s or 1 ms)
    bin_var = 1
    binsData_var =  np.arange(0, T + bin_var, bin_var)
    numBinData_var = len(binsData_var)-1
    binned_data_var = binData(data, [numTrials,numBinData_var]) * deltaT
    data_mean = np.mean(binned_data_var)/bin_var
    data_var = comp_cc(binned_data_var, binned_data_var, 1, bin_var, numBinData_var)[0]
    
    # bin data
    binsData =  np.arange(0, T + binSize, binSize)
    numBinData = len(binsData)-1
    binned_data = binData(data, [numTrials,numBinData]) * deltaT
    
    sumStat = comp_sumStat(binned_data, summStat_metric, ifNorm, deltaT, binSize, T, numBinData, maxTimeLag)
    
    return sumStat, data_mean, data_var, T, numTrials

    
    
def compute_spikesDisperssion_twoTau(ac_data, theta, deltaT, binSize, T, numTrials, data_mean, data_var,\
                             min_disp, max_disp, jump_disp, borderline_jump, numIter):
    """Compute the disperssion parameter of spike counts from the autocorrelation of 
    a doubly stochastic process with two timescale using grid search method.
    
    Parameters
    -----------
    ac_data : 1d array
        autocorrelation of data.
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
    min_disp : float
        minimum value of disperssion for grid search.
    max_disp : float
        maximum value of disperssion for grid search.
    jump_disp : float
        resolution of grid search.
    borderline_jump : int
        is used when the intial range of dispersion grid search is initially small, 
        defines number of "jump_disps" to be searched below or above the initial range.

    Returns
    -------
    disp_estim : float
        estimated value of disperssion.

    """
    
    disp_range = np.arange(min_disp, max_disp, jump_disp)    
    maxTimeLag = 2
   
    min_border = 0
    max_border = 0
    border_line = 1

    while border_line == 1:
        error_all = []
        if min_border or max_border:
            disp_range = np.arange(min_disp, max_disp, jump_disp)        

        for disp in disp_range:
            print(disp)
            error_sum = 0
            for i in range(numIter):
                data_syn, numBinData = twoTauOU_gammaSpikes(theta, deltaT, binSize, T, numTrials, data_mean,\
                                                            data_var, disp)
                ac_syn = comp_cc(data_syn, data_syn, maxTimeLag, binSize, numBinData)
                error_sum = error_sum + abs(ac_syn[1] - ac_data[1])

            error_all.append(error_sum)
        error_all = np.array(error_all)
        disp_estim = disp_range[np.argmin((error_all))]
   

        if disp_estim == disp_range[0]:
                min_border = 1
                min_disp, max_disp = min_disp - borderline_jump * jump_disp, min_disp + jump_disp

        elif  disp_estim == disp_range[-1]:
                max_border = 1
                min_disp, max_disp = max_disp - jump_disp, max_disp + borderline_jump * jump_disp

        else:
            border_line = 0
        
    return disp_estim


def compute_spikesDisperssion_oneTau(ac_data, theta, deltaT, binSize, T, numTrials, data_mean, data_var,\
                             min_disp, max_disp, jump_disp, borderline_jump, numIter = 200):
    """Compute the disperssion parameter of spike counts from the autocorrelation of 
    a doubly stochastic process with one timescale using grid search method.
    
    Parameters
    -----------
    ac_data : 1d array
        autocorrelation of data.
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
        mean value of the OU process (average of firing rate). 
    data_var : float
        variance of the OU process (variance of firing rate). 
    min_disp : float
        minimum value of disperssion for grid search.
    max_disp : float
        maximum value of disperssion for grid search.
    jump_disp : float
        resolution of grid search.
    borderline_jump : int
        is used when the intial range of dispersion grid search is initially small, 
        defines number of "jump_disps" to be searched below or above the initial range.

    Returns
    -------
    disp_estim : float
        estimated value of disperssion.

    """
    
    disp_range = np.arange(min_disp, max_disp, jump_disp)    
    maxTimeLag = 2
  
    min_border = 0
    max_border = 0
    border_line = 1

    while border_line == 1:
        error_all = []
        if min_border or max_border:
            disp_range = np.arange(min_disp, max_disp, jump_disp)        

        for disp in disp_range:
            print(disp)
            error_sum = 0
            for i in range(numIter):
                data_syn, numBinData = oneTauOU_gammaSpikes(theta, deltaT, binSize, T, numTrials, data_mean,\
                                                            data_var, disp)
                ac_syn = comp_cc(data_syn, data_syn, maxTimeLag, binSize, numBinData)
                error_sum = error_sum + abs(ac_syn[1] - ac_data[1])

            error_all.append(error_sum)
        error_all = np.array(error_all)
        disp_estim = disp_range[np.argmin((error_all))]
   

        if disp_estim == disp_range[0]:
                min_border = 1
                min_disp, max_disp = min_disp - borderline_jump * jump_disp, min_disp + jump_disp

        elif  disp_estim == disp_range[-1]:
                max_border = 1
                min_disp, max_disp = max_disp - jump_disp, max_disp + borderline_jump * jump_disp

        else:
            border_line = 0
        
    return disp_estim


def check_expEstimates(theta, deltaT, binSize, T, numTrials, data_mean, data_var,\
                              maxTimeLag, numTimescales, numIter = 500, confidence_level = 0.05):
    """Preprocessing function to check if timescales from exponential fits are reliable for the given data.
    
    Parameters
    -----------
    theta : 1d array
        timescales fitted by exponential functions on real data autocorrelations
        [timescale1, timescale2, coefficient for timescale1] or [timescale]
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
    maxTimeLag : float
        maximum time-lag for computing and fitting autocorrelations
    numTimescales: 1 or 2
        number of timescales to fit
    numIter: float, default 500
        number iterations to generate synthetic data and fit with exponentials
    confidence_level: float, default 0.05
        confidence level to check for the significant differences between fitted and original timescales

    Returns
    -------
    taus_all : nd array
        array of estimated timescales from exponential fits.
    
    """
    if numTimescales > 2:
        raise ValueError('Function is not designed for more than two timescales.')
        
    if numTimescales == 2:
        tau1_exp = []
        tau2_exp = []
        for i in range(numIter):
            data_syn, numBinData = twoTauOU(theta, deltaT, binSize, T, numTrials, data_mean, data_var)
            ac_syn = comp_ac_fft(data_syn)
            lm = round(maxTimeLag/binSize)
            ac_syn = ac_syn[0:lm]

            # fit exponentials
            xdata = np.arange(0, maxTimeLag , binSize)
            ydata = ac_syn
            popt, pcov = curve_fit(double_exp, xdata, ydata, maxfev = 2000)
            taus = np.sort(popt[1:3])
            tau1_exp.append(taus[0])
            tau2_exp.append(taus[1])
        taus_all = np.array([tau1_exp, tau2_exp])

        # check if timescales are within confidence range
        percent = confidence_level *100
        lower_bound = np.percentile(tau2_exp, percent)
        upper_bound = np.percentile(tau2_exp, 100- percent)
        good_exp_tau2 = (theta[1] < lower_bound) & (theta[1] > upper_bound)
        if not good_exp_tau2:
            print('Estimates for tau2 are significantly biased!')

        lower_bound = np.percentile(tau1_exp, percent)
        upper_bound = np.percentile(tau1_exp, 100- percent)
        good_exp_tau1 = (theta[0] < lower_bound) & (theta[0] > upper_bound)
        if not good_exp_tau1:
            print('Estimates for tau1 are significantly biased!')


    if numTimescales == 1:
        tau_exp = []
        for i in range(numIter):
            data_syn, numBinData = oneTauOU(theta, deltaT, binSize, T, numTrials, data_mean, data_var)
            ac_syn = comp_ac_fft(data_syn)
            lm = round(maxTimeLag/binSize)
            ac_syn = ac_syn[0:lm]

            # fit exponentials
            xdata = np.arange(0, maxTimeLag , binSize)
            ydata = ac_syn
            popt, pcov = curve_fit(single_exp, xdata, ydata, maxfev = 2000)
            tau_exp.append(popt[1])
        taus_all = np.array(tau_exp)
        # check if timescales are within confidence range
        percent = confidence_level *100
        lower_bound = np.percentile(tau_exp, percent)
        upper_bound = np.percentile(tau_exp, 100- percent)
        good_exp_tau = (theta[0] < lower_bound) & (theta[0] > upper_bound)
        if not good_exp_tau:
            print('Estimates for tau are significantly biased!')

    return taus_all



def fit_twoTauExponential(ac, binSize, maxTimeLag):
    """Fit the autocorrelation with a double exponential using non-linear least squares method.
    
    Parameters
    -----------
    ac : 1d array
        autocorrelation.
    binSize : float
        bin-size used for computing the autocorrelation.
    maxTimeLag : float
        maximum time-lag  used for computing the autocorrelation.
  
    Returns
    -------
    popt : 1d array
        optimal values for the parameters: [amplitude, timescale1, timescale2, weight of the timescale1].
    pcov: 2d array
        estimated covariance of popt. The diagonals provide the variance of the parameter estimate. 
    
    """
    

    # fit exponentials
    xdata = np.arange(0, maxTimeLag, binSize)
    ydata = ac
    popt, pcov = curve_fit(double_exp, xdata, ydata, maxfev = 2000)
    return popt, pcov


def fit_oneTauExponential(ac, binSize, maxTimeLag):
    """Fit the autocorrelation with a single exponential using non-linear least squares method.
    
    Parameters
    -----------
    ac : 1d array
        autocorrelation.
    binSize : float
        bin-size used for computing the autocorrelation.
    maxTimeLag : float
        maximum time-lag  used for computing the autocorrelation.
  
    Returns
    -------
    popt : 1d array
        optimal values for the parameters: [amplitude, timescale].
    pcov: 2d array
        estimated covariance of popt. The diagonals provide the variance of the parameter estimate. 
    
    """
    

    # fit exponentials
    xdata = np.arange(0, maxTimeLag, binSize)
    ydata = ac
    popt, pcov = curve_fit(single_exp, xdata, ydata, maxfev = 2000)
    return popt, pcov