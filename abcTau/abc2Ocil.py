"""
Module for fitting the aABC algorithm with two-timescales generative models
"""

from simple_abc_onlyOU2oscil import Model, basic_abc, pmc_abc
from generative_models import *
from basic_functions import *
from distance_functions import *
from summary_stats import *
import numpy as np
from scipy import stats







def fit_withABC_2oscil(MyModel, data_ac, priorDist, inter_save_direc, inter_filename, datasave_path, filenameSave,\
                     epsilon_0, min_samples, steps, minAccRate, parallel = False, n_procs = None, disp = None,\
                     resume = None):
    """Fits data autocorrelation with a given two-timescales generative model and saves the results.
    
    Parameters
    -----------
    MyModel : object
        Model object containing the generative model and distance functions (check example scripts or tutorials).
    data_ac : 1d array
        Prior distributions for aABC fitting (check example scripts or tutorials).
    priorDist : list object
        bin-size for binning data and computing the autocorrelation.
    inter_save_direc : string
        directory for saving intermediate results after running each step.
    inter_filename : string
        filename for saving intermediate results after running each step.
    filenameSave : string
        filename for saving final results, number of steps and maximumTimeLag will be attached to it.
    epsilon_0 : float
        initial error threshold
    min_samples : int
        number of accepted samples in postrior distribution for each step of the aABC.
    steps : int
        maximum number of steps (iterations) for running the aABC algorithm.
    minAccRate : float
        minimum proportion of samples accepted in each step, between 0 and 1.
    parallel : boolean, default False
        if run parallel processing.
    n_procs : int, optional, default None
        number of cores used for parallel processing.
    disp : float, default None
        The value of dispersion parameter if computed with the grid search method.
    resume : numpy record array, optional
        A record array of a previous pmc sequence to continue the sequence on.
        
        

    Returns
    -------
    abc_results : object
        A record containing all aABC output from all steps, including 'theta accepted', 'epsilon'.
    final_step : int
        Last step of running the aABC algorithm.
    
    """
    
     
    #Initialize model object
    model = MyModel()
    model.set_prior(priorDist)


    # give the model our observed data 
    model.set_data(data_ac)
    data = data_ac
    np.warnings.filterwarnings('ignore')

    # fit the model
    abc_results = pmc_abc(model, data, inter_save_direc, inter_filename, epsilon_0 = epsilon_0,\
                            min_samples = min_samples, steps = steps, parallel = parallel, n_procs = n_procs,\
                            minAccRate = minAccRate, resume = resume)
    # finding the final step and save the results  
    final_step = steps
    for i in range(len(abc_results)):
        if abc_results[i][-1] == None:
                final_step = i
                break
    filenameSave = filenameSave  + '_steps' + str(final_step)
    np.save(datasave_path + filenameSave, abc_results)

    print('END OF FITTING!!!')
    print('***********************')
    
    return abc_results, final_step