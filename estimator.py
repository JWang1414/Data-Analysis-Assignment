from abc import abstractmethod
import pickle
import numpy as np
from scipy.optimize import curve_fit
from pulse_plotting import *

class AbstractEstimator:
    def __init__(self, data, constant):
        """
        data: str
            The path to the data file
        """
        self.constant = constant
        self.estimation = np.zeros(1000)
        with open(data,"rb") as file:
            self.data=pickle.load(file)
        
    def calculate_estimation(self):
        """
        Calculate the estimation of the data as dictated by
        the estimator procedure
        """
        for i in range(1000):
            current_data = np.multiply(self.data['evt_%i'%i], self.constant)
            self.estimation[i] = self._calculate_estimation_helper(current_data) * 1e3

    @abstractmethod
    def _calculate_estimation_helper(self, current_data):
        """
        Calculate the associated estiamtion procedure for the given data
        """
        pass

    def get_estimation(self):
        return self.estimation


class MaxMinEstimator(AbstractEstimator):
    def _calculate_estimation_helper(self, current_data):
        return np.max(current_data) - np.min(current_data)
    
class MaxBaselineEstimator(AbstractEstimator):
    def _calculate_estimation_helper(self, current_data):
        baseline = np.mean(current_data[:999])
        return np.max(current_data) - baseline
    
class SumAllEstimator(AbstractEstimator):
    def _calculate_estimation_helper(self, current_data):
        return np.sum(current_data)
    
class SumAllBaselineEstimator(AbstractEstimator):
    def _calculate_estimation_helper(self, current_data):
        baseline = np.mean(current_data[:999])
        current_data -= baseline
        return np.sum(current_data)
    
class SumPulseEstimator(AbstractEstimator):
    def _calculate_estimation_helper(self, current_data):
        return np.sum(current_data[1000:1300])
    
class PulseFitEstimator(AbstractEstimator):
    def __init__(self, data, constant):
        super().__init__(data, constant)
        self.uncertainty = pulse_uncertainty()

    def _calculate_estimation_helper(self, current_data):
        xx=np.linspace(0, 4095, 4096)
        return curve_fit(fit_pulse, xx, current_data,
                         sigma=self.uncertainty, absolute_sigma=True)[0][0]