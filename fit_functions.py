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
5: PulseFitEstimator
"""
ESTIMATOR_ID = 5
DATA_FILE = "calibration.pkl"
PLOT_BEST_FIT = True

"""
To avoid using calibration constants, set to 0
To use calibration constants, set to 1
"""
USE_CALIBRATION_CONSTANTS = 0
CALIBRATION_CONSTANTS = [
        [1, 28.96],
        [1, 35.5],
        [1, 0.26],
        [1, 0.30],
        [1, 0.31],
        [1, 38.7]
    ]

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
    use_conversion = np.absolute(USE_CALIBRATION_CONSTANTS - 1)
    adjust_factor = CALIBRATION_CONSTANTS[ESTIMATOR_ID][use_conversion]
    popt, pcov = curve_fit(gaussian, bin_centres, bin_counts,
                           sigma=uncertainty, p0=(150, 10 / adjust_factor, 1 / adjust_factor, 3), absolute_sigma=True)

    # Calculate the Chi squared and degrees of freedom
    bin_counts_fit = gaussian(bin_centres, *popt)
    chi_squared = np.sum( ((bin_counts - bin_counts_fit)/uncertainty )**2)
    dof = bin_counts - len(popt)

    # Plot the Gaussian fit
    x_bestfit = np.linspace(bin_edges[0], bin_edges[-1], 1000)
    y_bestfit = gaussian(x_bestfit, *popt)
    plt.plot(x_bestfit, y_bestfit, label='Fit')

def get_estimator(choice: int):
    """
    Creates an estimator based on use choice.
    
    Parameters:
        choice: The chosen estimator, as dictated by the value from the table above
    
    Returns:
        The chosen estimator
    """
    applied_constant = CALIBRATION_CONSTANTS[choice][USE_CALIBRATION_CONSTANTS]
    
    match choice:
        case 0:
            return est.MaxMinEstimator(DATA_FILE, applied_constant)
        case 1:
            return est.MaxBaselineEstimator(DATA_FILE, applied_constant)
        case 2:
            return est.SumAllEstimator(DATA_FILE, applied_constant)
        case 3:
            return est.SumAllBaselineEstimator(DATA_FILE, applied_constant)
        case 4:
            return est.SumPulseEstimator(DATA_FILE, applied_constant)
        case 5:
            return est.PulseFitEstimator(DATA_FILE, applied_constant)

def get_hist_settings():
    # Define the number of bins, and the bin range for the histogram
    if USE_CALIBRATION_CONSTANTS == 1 and DATA_FILE == "calibration.pkl":
        hist_settings = {
            'num_bins': [30, 30, 40, 40, 40, 30],
            'bin_range': [(7, 13), (7, 13), (-20, 40), (0, 20), (0, 20), (7, 13)]
        }
    elif USE_CALIBRATION_CONSTANTS == 1 and DATA_FILE == "noise.pkl":
        hist_settings = {
            'num_bins': [20, 20, 40, 40, 40, 20],
            'bin_range': [(3, 5), (1.7, 3), (-20, 40), (0, 20), (0, 20), (-0.5, 0.5)]
        }
    elif USE_CALIBRATION_CONSTANTS == 1 and DATA_FILE == "signal.pkl":
        hist_settings = {
            'num_bins': [20, 20, 40, 40, 40, 20],
            'bin_range': [(3, 8), (1.7, 8), (-20, 40), (0, 20), (0, 20), (-1, 8)]
        }
    else:
        # Default settings. Mainly used for no calibration constants, on the calibration data
        hist_settings = {
            'num_bins': [40, 30, 40, 40, 40, 40],
            'bin_range': [(0.2, 0.5), (0.2, 0.4), (-100, 130), (-5, 70), (10, 50), (0.1, 0.4)]
        }
    return hist_settings


if __name__ == "__main__":
    # TODO: Apply estimators to noise and signal data

    # Initialize font and font size
    font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 22}
    rc('font', **font)

    # Initialize the estimator
    chosen_estimator = get_estimator(ESTIMATOR_ID)

    # Calculate estimations
    chosen_estimator.calculate_estimation()
        
    # Collect histogram settings
    hist_settings = get_hist_settings()

    # Plot the histogram of the calibration data, save the bin counts and bin edges
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
    individual_bin_range = (chosen_bin_range[1] - chosen_bin_range[0]) / chosen_num_bins
    units = ["mV", "keV"]
    plt.xlabel(f"Amplitude ({units[USE_CALIBRATION_CONSTANTS]})")
    plt.ylabel(f"Number of Events per {round(individual_bin_range * 1e3, 2)} μV")
    plt.xlim(chosen_bin_range)
    plt.tight_layout()
    plt.legend(loc=1)

    plt.show()