## abcTau

abcTau is a Python package for unbiased estimation of timescales from autocorrelations or power spectrums using adaptive Approximate Bayesian Computations (aABC). Timescales estimated from data by fitting the autocorrelation of sample time-series with exponential decay functions are systematically biased. abcTau overcomes this bias by fitting the sample autocorrelation or power spectrum with a generative model based on a mixture of Ornstein-Uhlenbeck (OU) processes. This way it accounts for finite sample size and noise in data and returns a posterior distribution of timescales, that quantifies the uncertainty of estimates and can be used to compare alternative hypotheses about the dynamics of the underlying process. 

Details of the method are explained in:   
Zeraati, R., Engel, T. A. & Levina, A. Estimation of autocorrelation timescales with Approximate Bayesian Computations. bioRxiv 2020.08.11.245944 (2020). https://www.biorxiv.org/content/10.1101/2020.08.11.245944v1
The basis of aABC algorithm in this package is adopted from a previous implementation (originally developed in Python 2):
Robert C. Morehead and Alex Hagen. A Python package for Approximate Bayesian Computation (2014). https://github.com/rcmorehead/simpleabc. 

Check Jupyter Notebook Tutorials 1 and 2 to learn how to use abcTau for estimating timescales and performing Bayesian model comparison. You can use "check_expEstimates" function from the the "preprocessing" module to check the bias in timescales estimated from exponential fits on your data.


## Dependencies
- Python >= 3.7.1
- Numpy >= 1.15.4 
- Scipy >= 1.1.0 


## Installation
For the current version you need to add the package path manually:
```
import sys
sys.path.append('./abcTau')
```
For the final version we will provide the "pip install" option for the package.


## List of available summary statistics computations (check "summary_stats.py" for details):
Ordered from fastest to slowest fitting:
- comp_psd: computing the power spectral density
- comp_ac_fft: computing autocorrelation using FFT to speed up (more biased for direct fitting) 
- comp_cc: computing autocorrelation in time domain based on paper 


## List of available generative models (check "generative_models.py" for details):
- oneTauOU 
- twoTauOU
- oneTauOU_oscil
- oneTauOU_poissonSpikes
- oneTauOU_gammaSpikes
- oneTauOU_gaussianSpikes
- twoTauOU_poissonSpikes
- twoTauOU_gammaSpikes
- twoTauOU_gaussianSpikes
- twoTauOU_gammaSpikes_withDispersion
- twoTauOU_gaussianSpikes_withDispersion


## List of available distance functions (check "distance_functions.py" for details):
- linear_distance
- logarithmic_distance
