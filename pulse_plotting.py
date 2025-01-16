import pickle
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.optimize import curve_fit

def pulse_shape(t_rise, t_fall):
    xx=np.linspace(0, 4095, 4096)
    yy = -(np.exp(-(xx-1000)/t_rise)-np.exp(-(xx-1000)/t_fall))
    yy[:1000]=0
    yy /= np.max(yy)
    return yy

def fit_pulse(x, A):
    _pulse_template = pulse_shape(20,80)
    xx=np.linspace(0, 4095, 4096)
    return A*np.interp(x, xx, _pulse_template)

def pulse_uncertainty():
    # Determine the uncertainty in the calibration using the noise data
    with open("noise.pkl","rb") as file:
        noise_data=pickle.load(file)
    
    std_list = []
    for i in range(1000):
        current_data = noise_data['evt_%i'%i]
        std_list.append(np.std(current_data))

    return np.mean(std_list)


if __name__ == "__main__":
    # Initialize font and font size
    font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 22}
    rc('font', **font)

    uncertainty = pulse_uncertainty()
    
    # Load calibration data
    with open("calibration.pkl","rb") as file:
        calibration_data=pickle.load(file)

    # Define linspace for plotting
    xx = np.linspace(0, 4095, 4096)
    
    # Determine optimized parameters for the pulse fit
    popt, pcov = curve_fit(fit_pulse, xx, calibration_data['evt_2'],
                           sigma=uncertainty, p0=(0.5), absolute_sigma=True
                           )
      
    # Plot pulse fit
    x_bestfit = np.linspace(0, 4095, 10000)
    y_bestfit = fit_pulse(x_bestfit, popt[0])
    plt.plot(x_bestfit, y_bestfit * 1e3, label='Pulse Fit', color='b')
    
    # Plot data
    plt.plot(xx, calibration_data['evt_2'] * 1e3, color = 'k', lw = 0.5)

    # Set plot labels and formatting
    plt.xlabel('Time (ms)')
    plt.ylabel('Readout Voltage (mV)')
    plt.tight_layout()
    plt.legend(loc=1)

    plt.show()