import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.optimize import curve_fit
from calibration.functions import get_fitting_function
from calibration.dose import CalibrationDose
from calibration.image_processing import read_image, filter_image
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error 
import cv2


class FilmCalibration:

    def __init__(self, groundtruth_image: np.ndarray, bits_per_channel=8, 
                 calibration_type: str = 'single-channel', fitting_function_name: str = 'polynomial',
                 filter_type='median'):
        """
        Initializes the film calibration process by setting the ground truth image 
        and the calibration type.

        Parameters
        ----------
        groundtruth_image : np.ndarray
            The original ground truth image where the irradiated films are located.
            It must be a NumPy array with dimensions [height, width, channels].
        bits_per_channel : int, optional
            Number of bits per channel (default is 8).
        calibration_type : str, optional
            Calibration type ('single-channel' or 'multi-channel'). Default is 'single-channel'.
        fitting_function_name : str, optional
            Name of the fitting function to use. Default is 'polynomial'.
        filter_type : str, optional
            Type of filter to apply to images. Default is 'median'.
        """
        if not isinstance(groundtruth_image, np.ndarray):
            if isinstance(groundtruth_image, str):
                # If a string is provided, attempt to read the image from the file
                groundtruth_image = read_image(groundtruth_image)
            else:
                raise ValueError("groundtruth_image must be a NumPy array.")
        # Store the ground truth image and basic calibration parameters
        self.groundtruth_image = groundtruth_image
        self.bits_per_channel = bits_per_channel
        self.calibration_type = calibration_type
        self.filter_type = filter_type
        # Dictionary to store CalibrationDose instances mapped by dose value
        self.doses = {}
        # List to store the pre-irradiation pixel values for each channel
        self.pixel_values_before = [None, None, None]
        # List to store the mapping between dose and independent variable per channel
        self.dose_to_independent_by_channel = []  
        self.parameters = None
        self.uncertainties = None  # Will store the standard deviations of the fitting parameters
        self.fitting_func_name = fitting_function_name
        # Retrieve the fitting function instance based on the provided function name
        self.fitting_func_instance = get_fitting_function(fitting_function_name)

    def add_roi(self, dose: float, x: int, y: int, width: int, height: int):
        """
        Registers a region of interest (ROI) associated with a given dose. If the dose 
        does not exist, a new CalibrationDose instance is created.

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
        # If the dose is not already registered, create a new CalibrationDose instance
        if dose not in self.doses:
            self.doses[dose] = CalibrationDose(dose, calibration=self)
        # Add the ROI to the corresponding dose
        self.doses[dose].add_roi(x, y, width, height)
    
    def get_total_roi_count(self) -> int:
        """
        Returns the total number of ROIs registered across all doses.

        Returns
        -------
        int
            Total number of ROIs.
        """
        # Sum the number of ROIs for each dose
        return sum(calib_dose.get_roi_count() for calib_dose in self.doses.values())

    def get_total_dose_count(self) -> int:
        """
        Returns the total number of doses registered in the calibration.

        Returns
        -------
        int
            Total number of doses.
        """
        # The count of doses is equal to the number of keys in the doses dictionary
        return len(self.doses)

    def get_rois_by_dose(self):
        """
        Prints out the ROIs for each registered dose.
        """
        # Iterate over each dose and its corresponding CalibrationDose object to print the ROIs
        for dose, calib_dose in self.doses.items():
            print(f"Dose: {calib_dose.value} Gy")
            print(f"ROIs: {calib_dose.rois} \n")
            
            
    def compute_channel_independent_values(self, channel: int = 0) -> dict:
        """
        Computes the independent variable (e.g., netOD or netT) for each dose using 
        the global pre-irradiation pixel value (PV_before).

        Parameters
        ----------
        channel : int, optional
            The channel index for which to compute the independent variable (default is 0).

        Returns
        -------
        dict
            A dictionary mapping dose values to the computed independent variable.
        """
        # Ensure that dose 0 is defined to set the pre-irradiation pixel value (PV_before)
        if 0 not in self.doses:
            raise ValueError("Dose 0 must be defined in order to obtain pixel_values_before.")
        
        # Compute the average pixel value for all doses for the specified channel
        for dose, calib_dose in self.doses.items():
            calib_dose.compute_average_pv(channel)

        # Set the global PV_before using dose 0
        self.pixel_values_before[channel] = self.doses[0].pixel_values_after.get(channel, None)
        if self.pixel_values_before[channel] is None:
            raise ValueError(f"PV_before (from dose 0) for channel {channel} is not defined.")
        
        dose_to_value = {}
        # Compute the independent variable (e.g., netOD or netT) for each dose
        for dose, calib_dose in self.doses.items():
            dose_to_value[dose] = calib_dose.compute_independent_value(self.pixel_values_before, channel)

        # Return the mapping sorted by dose value
        return dict(sorted(dose_to_value.items()))


    def calibrate(self, calibration_type: str = 'single-channel'):
        """
        Calibrates the film based on the specified calibration type.
        Currently, only single-channel calibration is implemented, which includes:

          1. Computing channel averages.
          2. Computing netOD (or another independent variable) for each dose.
          3. Fitting the calibration function using curve_fit.

        The fitting uses the independent variable as the independent value and the dose as the dependent value.

        Parameters
        ----------
        calibration_type : str, optional
            Calibration type ('single-channel' or 'multi-channel'). Default is 'single-channel'.

        Returns
        -------
        list
            A list of optimal parameters for each channel obtained from curve_fit.

        Raises
        ------
        ValueError
            If the pre-irradiation pixel value (PV_before) is not defined.
        """
        # Only single-channel calibration is currently implemented
        if calibration_type != 'single-channel':
            raise NotImplementedError("Only single-channel calibration is implemented.")
        
        # Ensure that dose 0 exists for setting PV_before
        if 0 not in self.doses:
            raise ValueError("Dose 0 must be defined to set pixel_values_before.")

        dose_to_independent_by_channel = []
        parameters = []
        uncertainties = []
        
        # Iterate over the three channels of the image
        for channel in range(0, 3):
            # Calculate the independent variable mapping for the given channel
            dose_to_independent = self.compute_channel_independent_values(channel)
            dose_to_independent_by_channel.append(dose_to_independent)

            # Prepare lists for curve fitting: dose values and corresponding independent variable values
            dose_list = np.array(list(dose_to_independent.keys()))
            independent_list = np.array(list(dose_to_independent.values()))
        
            # Get initial parameter guess for the fitting function based on expected number of parameters
            num_params = len(self.fitting_func_instance.param_names)
            p0 = self.fitting_func_instance.initial_param_guess

            # Perform the curve fitting using the fitting function and the data
            popt, pcov = curve_fit(self.fitting_func_instance.func, independent_list, dose_list,
                                   p0=p0, maxfev=10000)
            parameters.append(popt)
            # Calculate uncertainties as the square root of the diagonal elements of the covariance matrix
            uncertainties.append(np.sqrt(np.diag(pcov)))

        # Store the computed mappings, parameters, and uncertainties for later use
        self.dose_to_independent_by_channel = dose_to_independent_by_channel
        self.parameters = parameters
        self.uncertainties = uncertainties

        return parameters
    
    def get_metric(self, metric_name: str, channel: int = None):
        """
        Computes a specified metric for the calibration curve(s) and returns a tuple 
        containing the metric value and its formatted name.

        Parameters
        ----------
        metric_name : str
            The metric to compute. Options are:
            - 'r2' or 'r^2' for the coefficient of determination,
            - 'rmse' for the root mean square error,
            - 'mse' for the mean square error,
            - 'chi2', 'chi-squared', or 'chi^2' for the chi-squared value.
        channel : int, optional
            The channel index for which to compute the metric. If None, returns a
            dictionary mapping each channel index to a (value, formatted name) tuple.

        Returns
        -------
        tuple or dict
            A tuple (metric_value, formatted_metric_name) if a channel is specified,
            otherwise a dictionary mapping channel indices to such tuples.
        """
        # Ensure that calibration has been performed and the necessary parameters exist
        if self.parameters is None or not self.dose_to_independent_by_channel:
            raise ValueError("Calibration parameters have not been computed. Please run calibrate() first.")

        def compute_for_channel(ch):
            # Get dose values and the corresponding independent variable values for the channel
            dose_to_x = self.dose_to_independent_by_channel[ch]
            dose_list = np.array(list(dose_to_x.keys()))
            x_list = np.array(list(dose_to_x.values()))
            popt = self.parameters[ch]
            # Compute the predicted dose using the fitting function with the optimized parameters
            dose_predicted = self.fitting_func_instance.func(x_list, *popt)
            
            metric = metric_name.lower()
            # Calculate the requested metric
            if metric in ['r2', 'r^2']:
                value = r2_score(dose_list, dose_predicted)
                formatted = "R²"
            elif metric in ['rmse', 'RMSE']:
                value = root_mean_squared_error(dose_list, dose_predicted)
                formatted = "RMSE"
            elif metric in ['mse', 'MSE']:
                value = mean_squared_error(dose_list, dose_predicted)
                formatted = "MSE"
            elif metric in ['chi2', 'chi-squared', 'chi squared', 'chi^2']:
                value = np.sum((dose_list - dose_predicted) ** 2)
                formatted = "χ²"
            else:
                raise ValueError("Metric not recognized. Available metrics: 'r2', 'rmse', 'mse', and 'chi2'.")
            return value, formatted

        if channel is not None:
            # Return the metric for the specified channel
            return compute_for_channel(channel)
        else:
            # Otherwise, compute the metric for each channel and return in a dictionary
            metrics = {}
            for ch in range(len(self.parameters)):
                metrics[ch] = compute_for_channel(ch)
            return metrics
        
    def graph_calibration_curve(self, metric_name='r2'):
        """
        Graphs the calibration curves for each channel.
        The x-axis represents the independent variable (e.g., netOD or netT) and
        the y-axis represents the dose.
        
        Parameters
        ----------
        metric_name : str, optional
            The metric to display on the curve legend (default is 'r2').
        """
        colors = ['r', 'g', 'b']
        plt.figure(figsize=(8, 6))

        # Iterate over each channel to plot experimental data points and the fitted curve
        for i, dose_to_x in enumerate(self.dose_to_independent_by_channel):

            dose_list = np.array(list(dose_to_x.keys()))
            x_list = np.array(list(dose_to_x.values()))
        
            popt = self.parameters[i]
            uncert = self.uncertainties[i]

            # Define a smooth range for the independent variable for curve plotting
            x_fit = np.linspace(min(x_list), max(x_list) + 0.012, 300)
            dose_fit = self.fitting_func_instance.func(x_fit, *popt)
            
            # Predict the dose values at the measured independent variable values
            dose_predicted = self.fitting_func_instance.func(x_list, *popt)

            # Compute the specified metric for the current channel
            metric_value, metric_text = self.get_metric(metric_name, channel=i)

            # Create a label displaying optimized parameters with their uncertainties and the metric
            label_text = "\n".join(
                f"{name}={p:.3f}±{u:.3f}" 
                for name, p, u in zip(self.fitting_func_instance.param_names, popt, uncert)
            )
            label_text += f"\n{metric_text}={metric_value:.3f}"

            # Plot experimental data points and the fitted curve
            plt.scatter(x_list, dose_list, color=colors[i])
            plt.plot(x_fit, dose_fit, color=colors[i], linestyle='--', label=label_text)

        plt.title(f"Calibration Curves {self.fitting_func_instance.description}", fontsize=14)
        plt.ylabel("Dose (Gy)", fontsize=12)
        plt.xlabel(self.fitting_func_instance.independent_variable, fontsize=12)
        # Place the legend outside the plot area for clarity
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
        plt.grid(True)
        #plt.show()
        return plt.gcf()  

    def graph_response_curve(self, metric_name='r2'):
        """
        Graphs the response curves for each channel.
        The x-axis corresponds to the dose (Gy) and the y-axis corresponds to the film response 
        (e.g., netOD or netT), which is the independent variable.
        
        For each channel, this method numerically inverts the calibration function (which 
        returns the dose for a given response) to obtain the independent variable for a range of doses.

        Parameters
        ----------
        metric_name : str, optional
            The metric to display on the curve legend (default is 'r2').
        """
        colors = ['r', 'g', 'b']
        plt.figure(figsize=(8, 6))

        for i, dose_to_x in enumerate(self.dose_to_independent_by_channel):

            dose_list = np.array(list(dose_to_x.keys()))
            x_list = np.array(list(dose_to_x.values()))
        
            popt = self.parameters[i]
            uncert = self.uncertainties[i]

            # Define a smooth range for the independent variable inversion
            x_fit = np.linspace(min(x_list), max(x_list) + 0.015, 300)
            dose_fit = self.fitting_func_instance.func(x_fit, *popt)
            
            # Predict the dose values at the measured independent variable values
            dose_predicted = self.fitting_func_instance.func(x_list, *popt)

            # Obtain the specified metric for the current channel to include in the legend
            metric_value, metric_text = self.get_metric(metric_name, channel=i)

            # Create a label displaying optimized parameters with uncertainties and the metric
            label_text = "\n".join(
                f"{name}={p:.3f}±{u:.3f}"
                for name, p, u in zip(self.fitting_func_instance.param_names, popt, uncert)
            )
            label_text += f"\n{metric_text}={metric_value:.3f}"

            plt.scatter(dose_list, x_list, color=colors[i])
            plt.plot(dose_fit, x_fit, color=colors[i], linestyle='--', label=label_text)

        plt.title(f"Response Curves {self.fitting_func_instance.description}", fontsize=14)
        plt.xlabel("Dose (Gy)", fontsize=12)
        plt.ylabel(self.fitting_func_instance.independent_variable, fontsize=12)
        # Place the legend outside the plot area for clarity
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
        plt.grid(True)
        plt.show()

    def to_json(self, filename: str):
        """
        Exports the FilmCalibration instance to a JSON file.
        Only the following attributes are saved:
            - groundtruth_image (converted to list)
            - bits_per_channel
            - calibration_type
            - filter_type
            - pixel_values_before
            - dose_to_independent_by_channel
            - parameters (converted to lists)
            - uncertainties (converted to lists)
            - fitting_func_name
        Note: The doses, pre-irradiation pixel values (pixel_values_before), and the fitting function
              instance are not saved.

        Parameters
        ----------
        filename : str
            The path to the JSON file where the instance will be saved.
        """
        # Prepare a dictionary with the data to export, converting NumPy arrays to lists
        data = {
            "groundtruth_image": self.groundtruth_image.tolist(),
            "bits_per_channel": self.bits_per_channel,
            "calibration_type": self.calibration_type,
            "filter_type": self.filter_type,
            "pixel_values_before": [None if x is None else float(x) for x in self.pixel_values_before],
            "dose_to_independent_by_channel": self.dose_to_independent_by_channel,
            "parameters": [p.tolist() if isinstance(p, (np.ndarray, list)) else p for p in self.parameters] if self.parameters is not None else None,
            "uncertainties": [u.tolist() if isinstance(u, (np.ndarray, list)) else u for u in self.uncertainties] if self.uncertainties is not None else None,
            "fitting_func_name": self.fitting_func_name
        }
        # Write the dictionary to the specified JSON file
        with open(filename, 'w') as f:
            json.dump(data, f)

    @classmethod
    def from_json(cls, filename: str):
        """
        Loads a FilmCalibration instance from a JSON file.
        The JSON must contain the keys:
            - groundtruth_image
            - bits_per_channel
            - calibration_type
            - filter_type
            - dose_to_independent_by_channel
            - parameters
            - uncertainties
            - fitting_func_name
        Note: The doses, pre-irradiation pixel values (pixel_values_before), and the fitting function
              instance are not stored; the fitting function instance is re-initialized using the fitting_func_name.

        Parameters
        ----------
        filename : str
            The path to the JSON file to load.

        Returns
        -------
        FilmCalibration
            The reconstructed FilmCalibration instance.
        """
        # Open and read the JSON file to load the data
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Reconstruct the ground truth image from the stored list
        groundtruth_image = np.array(data["groundtruth_image"])
        # Create an instance using the basic stored parameters
        instance = cls(
            groundtruth_image=groundtruth_image,
            bits_per_channel=data.get("bits_per_channel"),
            calibration_type=data.get("calibration_type"),
            fitting_function_name=data.get("fitting_func_name"),
            filter_type=data.get("filter_type")
        )

        # Load the global pre-irradiation pixel values, converting each to np.float64 if necessary
        instance.pixel_values_before = [
            None if x is None else np.float64(x) for x in data.get("pixel_values_before", [None, None, None])
        ]
        
        # Reconstruct the mapping of dose to independent variable for each channel
        dose_to_independent_by_channel = data.get("dose_to_independent_by_channel", [])
        instance.dose_to_independent_by_channel = [
            {float(k): np.float64(v) for k, v in d.items()} for d in dose_to_independent_by_channel
        ]

        # Reconstruct the fitting parameters if available
        parameters = data.get("parameters")
        if parameters is not None:
            instance.parameters = [np.array(p) for p in parameters]
        else:
            instance.parameters = None

        # Reconstruct the uncertainties if available
        uncertainties = data.get("uncertainties")
        if uncertainties is not None:
            instance.uncertainties = [np.array(u) for u in uncertainties]
        else:
            instance.uncertainties = None
        
        return instance
    
    def compute_dose_map(self, film_file: str, channel: int = 0, new_size=512) -> np.ndarray:
        """
        Loads an irradiated film from a .tif file and computes the dose map using the calibrated model
        and the single-channel method.

        Parameters
        ----------
        film_file : str
            Path to the .tif file that contains the irradiated film.
        channel : int, optional
            The channel of the image to be used for calculating the dose map (default is 0).
        new_size : int, optional
            A new size for resizing the image (not used in the current implementation).

        Returns
        -------
        np.ndarray
            A 2D array representing the calculated dose map for the film.

        Raises
        ------
        ValueError
            If the global pre-irradiation pixel value (PV_before) for the specified channel is not defined
            or if the independent variable used is not supported.
        """
        # Load the film image from the file
        film_image = read_image(film_file)

        # Optionally resize the image using OpenCV (commented out to preserve original logic)
        # film_image = cv2.resize(film_image, (new_size, new_size), interpolation=cv2.INTER_NEAREST)

        # If the image has multiple channels, extract the specified channel; otherwise use the image itself
        if film_image.ndim == 3:
            film_channel = film_image[:, :, channel]
        else:
            film_channel = film_image

        # Apply the specified filter to reduce noise if a filter type is provided
        if self.filter_type is not None:
            film_channel = filter_image(film_channel, self.filter_type)

        # Retrieve the global pre-irradiation pixel value for the specified channel (set during calibration)
        PV_before = self.pixel_values_before[channel]
        if PV_before is None:
            raise ValueError(f"The global PV_before for channel {channel} is not defined. Please ensure dose 0 is calibrated.")

        # Calculate the independent variable (netOD or netT) for each pixel
        independent_variable = self.fitting_func_instance.independent_variable
        if independent_variable == "netOD":
            # Replace zeros to avoid division by zero
            film_channel_safe = np.where(film_channel == 0, 1e-6, film_channel)
            x_map = np.log10(PV_before / film_channel_safe)
        elif independent_variable == "netT":
            x_map = (film_channel - PV_before)
        else:
            raise ValueError(f"Unsupported independent variable type: {independent_variable}")

        # Retrieve the calibrated parameters for the specified channel
        popt = self.parameters[channel]

        # Apply the calibration function using the optimized parameters to obtain the dose map
        dose_map = self.fitting_func_instance.func(x_map, *popt)

        # Replace any NaN values with 0
        dose_map = np.nan_to_num(dose_map)
        
        return dose_map

def __repr__(self):
        # Return a string representation of the object for easier debugging and printing
        return (f"FilmCalibration(NumDoses={self.get_total_dose_count()}, "
                f"NumROIs={self.get_total_roi_count()}, Type={self.calibration_type})")
