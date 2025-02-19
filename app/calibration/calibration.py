import numpy as np
from scipy.optimize import curve_fit
from calibration.functions import get_fitting_function
from calibration.dose import CalibrationDose
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

class FilmCalibration:

    def __init__(self, groundtruth_image: np.ndarray, calibration_type: str = 'single-channel', fitting_function_name: str = 'exponential'):
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
        self.calibration_type = calibration_type
        self.doses = {}           # Dictionary mapping dose value to a CalibrationDose object
        # Global PV_before per channel, defined as the PV_after of dose 0 for each channel.
        self.pixel_values_before = [None, None, None]
        self.dose_to_netOD_by_channel = []

        self.parameters = None
        # Fitting function (to be defined according to the calibration method)
        self.fitting_function_name = fitting_function_name  
        func, text = get_fitting_function(fitting_function_name)
        self.fitting_function = func
        self.fitting_function_text = text

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
            

    def compute_channel_netODs(self, channel: int = 0) -> dict:
        """
        Computes the net optical density (netOD) for each dose using the global PV_before.
        PV_before is defined as the PV_after of the dose with value 0.
        
        Parameters
        ----------
        channel : int, optional
            The image channel to use for computation (default is 0).
        
        Returns
        -------
        dict
            A dictionary where the key is the dose value and the value is the computed netOD.
        
        Raises
        ------
        ValueError
            If PV_before is not defined.
        """
        # Ensure that the dose 0 exists and its PV_after is computed.
        if 0 not in self.doses:
            raise ValueError("Dose 0 must be defined in order to obtain pixel_values_before.")
        
        # Compute average PV for all doses first.
        for dose, calib_dose in self.doses.items():
            pv = calib_dose.compute_average_pv(channel)

        # Set the global PV_before for the channel using dose 0.
        self.pixel_values_before[channel] = self.doses[0].pixel_values_after.get(channel, None)
        if self.pixel_values_before[channel] is None:
            raise ValueError("PV_before (from dose 0) for channel {} is not defined.".format(channel))
        

        # Compute the netOD for each dose.
        dose_to_netOD = {}
        for dose, calib_dose in self.doses.items():
            dose_to_netOD[dose] = calib_dose.compute_netOD(self.pixel_values_before, channel)

        return dict(sorted(dose_to_netOD.items()))

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

        dose_to_netOD_by_channel = []
        parameters = []
        for channel in range(0, 3):
            
            dose_to_netOD = self.compute_channel_netODs(channel)
            dose_to_netOD_by_channel.append(dose_to_netOD)

            # Prepare lists for curve fitting.
            # The keys of netODs are the dose values.
            dose_list = np.array(list(dose_to_netOD.keys()))
            netOD_list = np.array(list(dose_to_netOD.values()))
        
            # Fit the calibration function to the data:
            popt, pcov = curve_fit(self.fitting_function, netOD_list, dose_list,
                                p0=[1.0, 1.0, 1.0], maxfev=5000)
            a_opt, b_opt, n_opt = popt
            parameters.append(popt)

        self.dose_to_netOD_by_channel = dose_to_netOD_by_channel
        self.parameters = parameters
        return parameters


    def graph_calibration_curve(self):

        colors = ['r', 'g', 'b']
        channels = ['R', 'G', 'B']

        plt.figure(figsize=(8, 6))

        for i, (dose_to_netOD) in enumerate(self.dose_to_netOD_by_channel):

            dose_list = np.array(list(dose_to_netOD.keys()))
            netOD_list = np.array(list(dose_to_netOD.values()))
        
            popt = self.parameters[i]
            a_opt, b_opt, n_opt = popt

            # Curva ajustada
            netOD_fit = np.linspace(min(netOD_list), max(netOD_list), 300)
            dose_fit = self.fitting_function(netOD_fit, a_opt, b_opt, n_opt)
            
            dose_predicted = self.fitting_function(netOD_list, *popt)
            r2 = r2_score(dose_list, dose_predicted)

            # TODO: cambiar
            text = f"a={a_opt:.4f}\n b={b_opt:.4f}\n n={n_opt:.4f}\n R²={r2:.4f}",

            # Graficar datos experimentales y curva de ajuste para cada canal
            plt.scatter(netOD_list, dose_list, color=colors[i])
            plt.plot(netOD_fit, dose_fit, color=colors[i], linestyle='--', label=text)


        plt.title(f"Curvas de calibración  {self.fitting_function_text}")
        plt.ylabel("Net Optical Density (netOD)")
        plt.xlabel("Dosis (Gy)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def graph_response_curve(self):

        colors = ['r', 'g', 'b']
        channels = ['R', 'G', 'B']

        plt.figure(figsize=(8, 6))

        for i, (dose_to_netOD) in enumerate(self.dose_to_netOD_by_channel):

            dose_list = np.array(list(dose_to_netOD.keys()))
            netOD_list = np.array(list(dose_to_netOD.values()))
        
            popt = self.parameters[i]
            a_opt, b_opt, n_opt = popt

            # Curva ajustada
            netOD_fit = np.linspace(min(netOD_list), max(netOD_list), 300)
            dose_fit = self.fitting_function(netOD_fit, a_opt, b_opt, n_opt)
            
            dose_predicted = self.fitting_function(netOD_list, *popt)
            r2 = r2_score(dose_list, dose_predicted)
            
            # TODO: cambiar
            text = f"a={a_opt:.4f}\n b={b_opt:.4f}\n n={n_opt:.4f}\n R²={r2:.4f}",

            # Graficar datos experimentales y curva de ajuste para cada canal
            plt.scatter(dose_list, netOD_list, color=colors[i])
            plt.plot(dose_fit, netOD_fit, color=colors[i], linestyle='--', label=text)


        plt.title(f"Curvas de respuesta  {self.fitting_function_text}")
        plt.xlabel("Net Optical Density (netOD)")
        plt.ylabel("Dosis (Gy)")
        plt.legend()
        plt.grid(True)
        plt.show()


    def __repr__(self):
        return (f"FilmCalibration(NumDoses={self.get_total_dose_count()}, "
                f"NumROIs={self.get_total_roi_count()}, Type={self.calibration_type})")

