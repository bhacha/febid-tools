import numpy as np
from scipy.optimize import curve_fit



def calibration_fit_function(t, gr, k):
        """Function for fitting the calibration."""
        return 1 / k * np.log(k * gr * t + 1)


def fit_calibration(dwell_times, lengths, gr0=0.1, k0=1):
        """Fits the calibration and returns optimal parameters and the
        fit function.

        Args:
            dwell_times (array): Array of measured dwell times.
            lengths (array): Array of measured lengths.
            gr0 (float, optional): Initial guess for GR. Defaults to 0.1.
            k0 (float, optional): Initial guess for k. Defaults to 1.

        Returns:
            [type]: [description]
        """
        fn = calibration_fit_function

        popt, pcov = curve_fit(
            fn, dwell_times, lengths, p0=[gr0, k0], bounds=(0, np.inf)
        )
        print("GR: ", popt[0])
        print("k: ", popt[1])
        return fn, popt, pcov


def get_resistance(structure, single_pixel_width=50):
    pass