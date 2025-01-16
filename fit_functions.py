import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import estimator as est
from typing import List
from scipy.optimize import curve_fit

"""
ID list:
0: MaxMinEstimator
1: MaxBaselineEstimator
2: SumAllEstimator
3: SumAllBaselineEstimator
4: SumPulseEstimator
"""
ESTIMATOR_ID = 1
DATA_FILE = "calibration.pkl"
PLOT_BEST_FIT = False

def gaussian(x, A, mean, width, base):
    """
    Model function for a Gaussian with a uniform background.
    
    Parameters:
        A: Constant amplitude of the Gaussian
        mean: Mean of the Gaussian
        width: Width of the Gaussian
        base: Constant background level

    Returns:
        y: The value of the Gaussian at x
    """
    return A * np.exp(-(x-mean)**2/(2*width**2)) + base

def plot_bestfit():
    """
    Helper function used to plot the Gaussian fit
    """
    # Optimize values for the Gaussian fit with curve_fit
    popt, pcov = curve_fit(gaussian, bin_centres, bin_counts,
                           sigma=uncertainty, absolute_sigma=True)

    # Calculate the Chi squared and degrees of freedom
    bin_counts_fit = gaussian(bin_centres, *popt)
    chi_squared = np.sum( ((bin_counts - bin_counts_fit)/uncertainty )**2)
    dof = bin_counts - len(popt)

    # Plot the Gaussian fit
    x_bestfit = np.linspace(bin_edges[0], bin_edges[-1], 1000)
    y_bestfit = gaussian(x_bestfit, *popt)
    plt.plot(x_bestfit, y_bestfit, label='Fit')

if __name__ == "__main__":
    # TODO: Fix the sum all estimators
    # TODO: Apply estimators to noise and signal data

    # Initialize font and font size
    font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 22}
    rc('font', **font)

    # Initialize the estimators   
    estimators: List[est.AbstractEstimator] = [
        est.MaxMinEstimator(DATA_FILE),
        est.MaxBaselineEstimator(DATA_FILE),
        est.SumAllEstimator(DATA_FILE),
        est.SumAllBaselineEstimator(DATA_FILE),
        est.SumPulseEstimator(DATA_FILE)
    ]

    # Calculate estimations
    for estimator in estimators:
        estimator.calculate_estimation()
        
    # Define the number of bins, and the bin range for the histogram
    hist_settings = {
        'num_bins': [40, 40, 40, 40, 40],
        'bin_range': [(0.2, 0.4), (0.2, 0.4), (0.2, 0.4), (0.2, 0.4), (0.2, 0.4)]
    }

    # Plot the histogram of the calibration data, save the bin counts and bin edges
    chosen_estimator = estimators[ESTIMATOR_ID]
    chosen_num_bins = hist_settings['num_bins'][ESTIMATOR_ID]
    chosen_bin_range = hist_settings['bin_range'][ESTIMATOR_ID]

    bin_counts, bin_edges, _ = plt.hist(chosen_estimator.get_estimation(), bins=chosen_num_bins, range=chosen_bin_range,
                                        color='k', histtype='step', label='Data')
    
    # Calculate the x-location for the bin centres
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate uncertainty for histogram
    uncertainty = np.sqrt(bin_counts)
    uncertainty = np.where(uncertainty == 0, 1, uncertainty)

    # Plot errorbars
    plt.errorbar(bin_centres, bin_counts, yerr=uncertainty,
                 fmt='none', c='k')
    
    # Determine Gaussian fit, and plot it
    if PLOT_BEST_FIT:
        plot_bestfit()

    # Set plot labels and formatting
    plt.xlabel('Amplitude (mV)')
    plt.ylabel('Number of Events per 5 microvolts')
    plt.xlim(chosen_bin_range)
    plt.tight_layout()
    plt.legend(loc=1)

    plt.show()