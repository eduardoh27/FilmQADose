from calibration.image_processing import filter_image
import numpy as np

class CalibrationDose:
    def __init__(self, dose_value: float, calibration=None):
        """
        Initializes an instance for a specific dose.

        Parameters
        ----------
        dose_value : float
            Dose value (in Gy).
        """
        self.calibration = calibration

        self.value = dose_value
        self.rois = []                # List of ROIs associated with this dose, each as (x, y, size)
        # Dictionaries to store the average pixel value (PV_after) and netOD per channel.
        self.pixel_values_after = {}  # {channel: PV_after}
        # self.netODs = {}              # {channel: netOD}
        self.independent_values = {}  # {channel: computed independent variable}

    def add_roi(self, x: int, y: int, size: int):
        """
        Adds a square ROI associated with this dose.

        Parameters
        ----------
        x : int
            X-coordinate (top-left corner) of the ROI.
        y : int
            Y-coordinate (top-left corner) of the ROI.
        size : int
            Length of the square defining the ROI.
        """
        self.rois.append((x, y, size))

    def compute_average_pv(self, channel: int = 0) -> float:
        """
        Computes the average pixel value (PV) for all ROIs associated with this dose,
        applying a median filter on the specified channel.

        Parameters
        ----------
        groundtruth_image : np.ndarray
            The original ground truth image from which ROIs are extracted.
        channel : int, optional
            The image channel to process (default is 0).

        Returns
        -------
        float
            The computed average pixel value for this dose.
        """
        roi_values = []
        for (x, y, size) in self.rois:
            # Extract the ROI from the selected channel
            roi = self.calibration.groundtruth_image[y:y+size, x:x+size, channel]
            # Apply the median filter
            filtered_roi = filter_image(roi, filter_type=self.calibration.filter_type)
            # Compute the average pixel value of the filtered ROI
            average_value = np.mean(filtered_roi)
            roi_values.append(average_value)
        # If there are ROI values, calculate the average; otherwise, pixel_value_after remains None.
        if roi_values:
            self.pixel_values_after[channel] = np.mean(roi_values)
        else:
            # TODO: añadir excepción: se invocó el método pero no había rois asociados
            pass
        return self.pixel_values_after.get(channel, None)
    
    def compute_independent_value(self, pixel_values_before, channel: int) -> float:
        """
        Computes the independent variable value for this dose based on the given type.

        Parameters
        ----------
        pixel_values_before : dict
            A dictionary with the pixel value before exposure for each channel.
        channel : int, optional
            The image channel to use.
        independent_variable : str
            The type of independent variable to compute ('netOD' or 'netT').
        bits_per_channel : int, optional
            Number of bits per channel (used for netT calculation), default is 8.

        Returns
        -------
        float
            The computed independent value.
        """
        independent_variable = self.calibration.fitting_func_instance.independent_variable

        if channel not in self.pixel_values_after or self.pixel_values_after[channel] is None:
            raise ValueError(f"The average pixel value after exposure for channel {channel} has not been computed yet.")

        if independent_variable == 'netOD':
            value = np.log10(pixel_values_before[channel] / self.pixel_values_after[channel])
        elif independent_variable == 'netT':
            bits_per_channel = self.calibration.bits_per_channel
            # netT = (PV_after - PV_before) / 2^(bits_per_channel)
            value = self.pixel_values_after[channel] - pixel_values_before[channel] # / (2 ** bits_per_channel)
        else:
            raise ValueError(f"Unsupported independent variable type: {independent_variable}")
            
        self.independent_values[channel] = value
        return value

    def get_roi_count(self) -> int:
        """
        Returns number of rois registered for this dose.

        Returns
        -------
        int
            Number of doses.
        """
        return len(self.rois)
    

    def __repr__(self):
        return (f"CalibrationDose(value={self.value}, num_rois={self.get_roi_count()}")
