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
        calibration : FilmCalibration, optional
            The FilmCalibration instance that this dose is part of.
        """
        self.calibration = calibration
        self.value = dose_value
        self.rois = []  # List of ROIs associated with this dose; each ROI is defined as a tuple (x, y, size)
        # Dictionary to store the average pixel value after exposure (PV_after) per channel.
        self.pixel_values_after = {}  # {channel: PV_after}
        # Dictionary to store the computed independent variable (e.g., netOD or netT) per channel.
        self.independent_values = {}  # {channel: computed independent variable}

    def add_roi(self, x: int, y: int, width: int, height: int):
        """
        Adds a square region of interest (ROI) associated with this dose.

        Parameters
        ----------
        x : int
            X-coordinate (top-left corner) of the ROI.
        y : int
            Y-coordinate (top-left corner) of the ROI.
        size : int
            The side length of the square ROI.
        """
        self.rois.append((x, y, width, height))

    def compute_average_pv(self, channel: int = 0) -> float:
        """
        Computes the average post-exposure pixel value (PV_after) for all ROIs associated with this dose.
        It applies a median filter on the specified channel of the ground truth image.

        Parameters
        ----------
        channel : int, optional
            The image channel to process (default is 0).

        Returns
        -------
        float
            The computed average pixel value for the specified channel or None if no ROI exists.
        """
        roi_values = []
        for (x, y, w, h) in self.rois:
            # Extract the ROI from the ground truth image for the specified channel.
            # use width and height to extract the ROI
            roi = self.calibration.groundtruth_image[y:y+h, x:x+w, channel]
            # Apply the median filter to the ROI to reduce noise.
            filtered_roi = filter_image(roi, filter_type=self.calibration.filter_type)
            # Compute the average pixel value of the filtered ROI.
            average_value = np.mean(filtered_roi)
            roi_values.append(average_value)
        # If there are computed ROI values, calculate and store the overall average.
        if roi_values:
            self.pixel_values_after[channel] = np.mean(roi_values)
        else:
            # TODO: Consider raising an exception if no ROIs were added for this dose.
            pass
        return self.pixel_values_after.get(channel, None)
    
    def compute_independent_value(self, pixel_values_before, channel: int) -> float:
        """
        Computes the independent variable for this dose based on the pre-exposure pixel value (PV_before)
        and the average post-exposure pixel value (PV_after). The type of independent variable (e.g., netOD or netT)
        is determined by the configuration of the calibration function.

        Parameters
        ----------
        pixel_values_before : dict
            A dictionary mapping each channel to its pre-exposure pixel value.
        channel : int
            The image channel to use for the calculation.

        Returns
        -------
        float
            The computed independent variable value for this dose on the specified channel.

        Raises
        ------
        ValueError
            If the average post-exposure pixel value for the given channel has not been computed.
        """
        independent_variable = self.calibration.fitting_func_instance.independent_variable

        # Check that the average pixel value after exposure has been computed for the channel.
        if channel not in self.pixel_values_after or self.pixel_values_after[channel] is None:
            raise ValueError(f"The average pixel value after exposure for channel {channel} has not been computed yet.")

        if independent_variable == 'netOD':
            # Compute net optical density: netOD = log10(PV_before / PV_after)
            value = np.log10(pixel_values_before[channel] / self.pixel_values_after[channel])
        elif independent_variable == 'netT':
            bits_per_channel = self.calibration.bits_per_channel
            # Compute net transmission (netT) as the absolute difference between PV_after and PV_before.
            # Optionally, this difference may be normalized by 2^(bits_per_channel) if needed.
            value = np.abs(self.pixel_values_after[channel] - pixel_values_before[channel])
        else:
            raise ValueError(f"Unsupported independent variable type: {independent_variable}")
            
        self.independent_values[channel] = value
        return value

    def get_roi_count(self) -> int:
        """
        Returns the number of regions of interest (ROIs) registered for this dose.

        Returns
        -------
        int
            The number of ROIs.
        """
        return len(self.rois)
    
    def __repr__(self):
        return f"CalibrationDose(value={self.value}, num_rois={self.get_roi_count()})"
