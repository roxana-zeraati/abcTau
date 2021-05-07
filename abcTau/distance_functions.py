"""
Module containing different distance functions. 
"""
import numpy as np
from scipy import stats


def linear_distance(data, synth_data):
    """ compute linear distance between autocorrelations.

    Parameters
    -----------
    data : 1d array
        autocorrelation of real data.
    synth_data : 1d array
        autocorrelation of synthetic data.
    
    Returns
    -------
    d : float
        linear ditance between autocorrelations.
    """
        
    d = np.nanmean(np.power(((data) - (synth_data)),2))
    return d


def logarithmic_distance(data, synth_data):
    """ compute logarithmic distance between autocorrelations.

    Parameters
    -----------
    data : 1d array
        autocorrelation of real data.
    synth_data : 1d array
        autocorrelation of synthetic data.
    
    Returns
    -------
    d : float
        logarithmic ditance between autocorrelations.
    """
    d = np.nanmean(np.power((np.log(data) - np.log(synth_data)),2))
    return d