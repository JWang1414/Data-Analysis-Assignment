import pickle
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import estimator as est
from typing import List

"""
ID list:
0: MaxMinEstimator
1: MaxBaselineEstimator
2: SumAllEstimator
3: SumAllBaselineEstimator
4: SumPulseEstimator
"""
ESTIMATOR_ID = 2
DATA_FILE = "calibration.pkl"

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

if __name__ == "__main__":
    # TODO: Create Gaussian fit for the distribution of the chosen estimator
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
                                        color='k', histtype='step', label='Data'
                                        )
    
    # Calculate the x-location for the bin centres
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Set plot labels and formatting
    plt.xlabel('Amplitude (mV)')
    plt.ylabel('Number of Events per 5 microvolts')
    plt.tight_layout()
    plt.legend(loc=1)

    plt.show()