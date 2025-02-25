import numpy as np
from scipy.optimize import curve_fit
from calibration.functions import get_fitting_function
from calibration.dose import CalibrationDose
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error 

class FilmCalibration:

    def __init__(self, groundtruth_image: np.ndarray, bits_per_channel = 8, 
                calibration_type: str = 'single-channel', fitting_function_name: str = 'polynomial  ',
                filter_type=None):
        """
        Initializes the film calibration process by defining the ground truth image
        and the calibration type.

        Parameters
        ----------
        groundtruth_image : np.ndarray
            The original ground truth image where the irradiated films are located.
            It must be a NumPy array with dimensions [height, width, channels].
        calibration_type : str, optional
            Calibration type ('single-channel' or 'multi-channel'). Default is 'single-channel'.
        """
        self.groundtruth_image = groundtruth_image
        self.bits_per_channel = bits_per_channel
        self.calibration_type = calibration_type
        self.filter_type = filter_type
        self.doses = {}           # Dictionary mapping dose value to a CalibrationDose object
        # Global PV_before per channel, defined as the PV_after of dose 0 for each channel.
        self.pixel_values_before = [None, None, None]
        #self.dose_to_netOD_by_channel = []
        self.dose_to_independent_by_channel = []  
        self.parameters = None
        self.uncertainties = None  # Will store the uncertainties (standard deviations) of the parameters
        
        # Retrieve the FittingFunction instance
        self.fitting_func_instance = get_fitting_function(fitting_function_name)

    def add_roi(self, dose: float, x: int, y: int, size: int):
        """
        Registers an ROI associated with a given dose. If the dose does not exist,
        a new CalibrationDose instance is created.

        Parameters
        ----------
        dose : float
            Dose value (in Gy).
        x : int
            X-coordinate (top-left corner) of the ROI.
        y : int
            Y-coordinate (top-left corner) of the ROI.
        size : int
            Length of the square defining the ROI.
        """
        if dose not in self.doses:
            self.doses[dose] = CalibrationDose(dose, calibration=self)
        self.doses[dose].add_roi(x, y, size)
    
    def get_total_roi_count(self) -> int:
        """
        Returns the total number of ROIs registered across all doses.

        Returns
        -------
        int
            Total number of ROIs.
        """
        return sum(calib_dose.get_roi_count() for calib_dose in self.doses.values())

    def get_total_dose_count(self) -> int:
        """
        Returns the total number of doses registered in the calibration.

        Returns
        -------
        int
            Total number of doses.
        """
        return len(self.doses)

    def get_rois_by_dose(self):
        for dose, calib_dose in self.doses.items():
            print(f"Dosis: {calib_dose.value} Gy")
            print(f"ROIs: {calib_dose.rois} \n")
            
    def compute_channel_independent_values(self, channel: int = 0) -> dict:
        """
        Computes the independent variable (e.g., netOD or netT) for each dose using the global PV_before.
        """
        if 0 not in self.doses:
            raise ValueError("Dose 0 must be defined in order to obtain pixel_values_before.")
        
        # First, compute the average pixel value for all doses.
        for dose, calib_dose in self.doses.items():
            calib_dose.compute_average_pv(channel)

        # Set the global pixel value before exposure (using dose 0).
        self.pixel_values_before[channel] = self.doses[0].pixel_values_after.get(channel, None)
        if self.pixel_values_before[channel] is None:
            raise ValueError(f"PV_before (from dose 0) for channel {channel} is not defined.")
        
        dose_to_value = {}
        for dose, calib_dose in self.doses.items():
            dose_to_value[dose] = calib_dose.compute_independent_value(self.pixel_values_before, channel)

        return dict(sorted(dose_to_value.items()))

    def calibrate(self, calibration_type: str = 'single-channel'):
        """
        Calibrates the film based on the specified calibration type.
        For now, only the single-channel calibration is implemented, which includes:
        
          1. Computing channel averages.
          2. Computing netOD for each dose.
          3. Fitting the calibration function using curve_fit.
        
        The fitting uses the netOD values as independent variables and the dose values as the dependent ones.
        
        Parameters
        ----------
        calibration_type : str, optional
            Calibration type ('single-channel' or 'multi-channel'). Default is 'single-channel'.
        
        Returns
        -------
        tuple
            The optimal parameters (a_opt, b_opt, n_opt) obtained from curve_fit.
        
        Raises
        ------
        ValueError
            If pixel_values_before is not defined.
        """
        if calibration_type != 'single-channel':
            raise NotImplementedError("Only single-channel calibration is implemented.")
        
        # Ensure pixel_values_before is defined using the dose with value 0.
        if 0 not in self.doses:
            raise ValueError("Dose 0 must be defined to set pixel_values_before.")

        dose_to_independent_by_channel = []
        parameters = []
        uncertainties = []
        for channel in range(0, 3):
            dose_to_independent = self.compute_channel_independent_values(channel)
            dose_to_independent_by_channel.append(dose_to_independent)

            # Prepare lists for curve fitting.
            # The keys of netODs are the dose values.
            dose_list = np.array(list(dose_to_independent.keys()))
            independent_list = np.array(list(dose_to_independent.values()))
        
            # Determine initial parameter guess based on the number of parameters.
            num_params = len(self.fitting_func_instance.param_names)
            p0 = self.fitting_func_instance.initial_param_guess

            # Fit the calibration function to the data:
            popt, pcov = curve_fit(self.fitting_func_instance.func, independent_list, dose_list,
                                p0=p0, maxfev=10000)
            parameters.append(popt)
            # Calculate uncertainties as the square root of the diagonal of the covariance matrix.
            uncertainties.append(np.sqrt(np.diag(pcov)))


        self.dose_to_independent_by_channel = dose_to_independent_by_channel
        self.parameters = parameters
        self.uncertainties = uncertainties

        return parameters

    def get_metric(self, metric_name: str, channel: int = None):
        """
        Computes the specified metric for the calibration curve(s) and returns a tuple containing the metric value and its formatted name.

        Parameters
        ----------
        metric_name : str
            The metric to compute. Options are:
            - 'r2' or 'r^2' for the coefficient of determination,
            - 'rmse' for the root mean square error,
            - 'mse' for the mean square error,
            - 'chi2', 'chi-squared', or 'chi^2' for the chi-squared value.
        channel : int, optional
            The channel index for which to compute the metric. If None, returns a dictionary mapping channel indices to (value, formatted name) tuples.

        Returns
        -------
        tuple or dict
            A tuple (metric_value, formatted_metric_name) if channel is specified,
            otherwise a dictionary mapping channel indices to such tuples.
        """
        if self.parameters is None or not self.dose_to_independent_by_channel:
            raise ValueError("Calibration parameters have not been computed. Please run calibrate() first.")

        def compute_for_channel(ch):
            dose_to_x = self.dose_to_independent_by_channel[ch]
            dose_list = np.array(list(dose_to_x.keys()))
            x_list = np.array(list(dose_to_x.values()))
            popt = self.parameters[ch]
            dose_predicted = self.fitting_func_instance.func(x_list, *popt)
            
            metric = metric_name.lower()
            if metric in ['r2', 'r^2']:
                value = r2_score(dose_list, dose_predicted)
                formatted = "R²"
            elif metric in ['RMSE', 'rmse']:
                value = root_mean_squared_error(dose_list, dose_predicted)
                formatted = "RMSE"
            elif metric in ['MSE', 'mse']:
                value = mean_squared_error(dose_list, dose_predicted)
                formatted = "MSE"
            elif metric in ['chi2', 'chi-squared', 'chi squared', 'chi^2']:
                value = np.sum((dose_list - dose_predicted) ** 2)
                formatted = "χ²"
            else:
                raise ValueError("Metric not recognized. Available metrics: 'r2', 'rmse', 'mse', and 'chi2'.")
            return value, formatted

        if channel is not None:
            return compute_for_channel(channel)
        else:
            metrics = {}
            for ch in range(len(self.parameters)):
                metrics[ch] = compute_for_channel(ch)
            return metrics


    def graph_calibration_curve(self, metric_name='r2'):
        """
        Graphs the calibration curves for each channel.
        The x-axis corresponds to the independent variable (e.g., netOD, netT) and
        the y-axis corresponds to the dose.
        """
        colors = ['r', 'g', 'b']
        plt.figure(figsize=(8, 6))

        for i, dose_to_x in enumerate(self.dose_to_independent_by_channel):

            dose_list = np.array(list(dose_to_x.keys()))
            x_list = np.array(list(dose_to_x.values()))
        
            popt = self.parameters[i]
            uncert = self.uncertainties[i]

            # add small value for curve to get near the last point (needed in some cases)
            x_fit = np.linspace(min(x_list), max(x_list) + 0.015, 300)
            dose_fit = self.fitting_func_instance.func(x_fit, *popt)
            
            dose_predicted = self.fitting_func_instance.func(x_list, *popt)

            # Compute metrics using the new get_metric method.
            metric_value, metric_text = self.get_metric(metric_name, channel=i)

            # Generate label text with parameter values and uncertainties.
            label_text = "\n".join(
                f"{name}={p:.3f}±{u:.3f}" 
                for name, p, u in zip(self.fitting_func_instance.param_names, popt, uncert)
            )
            label_text += f"\n{metric_text}={metric_value:.3f}"

            plt.scatter(x_list, dose_list, color=colors[i])
            plt.plot(x_fit, dose_fit, color=colors[i], linestyle='--', label=label_text)

        plt.title(f"Calibration Curves {self.fitting_func_instance.description}")
        plt.ylabel("Dose (Gy)")
        plt.xlabel(self.fitting_func_instance.independent_variable)
        # Position the legend outside the plot to the right
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
        plt.grid(True)
        plt.show()

    
    def __repr__(self):
        return (f"FilmCalibration(NumDoses={self.get_total_dose_count()}, "
                f"NumROIs={self.get_total_roi_count()}, Type={self.calibration_type})")

