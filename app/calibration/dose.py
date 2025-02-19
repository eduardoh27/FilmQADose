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
        self.netODs = {}              # {channel: netOD}

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
            filtered_roi = filter_image(roi, filter_type='median')
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


    def compute_netOD(self, pixel_values_before, channel: int = 0) -> float:
        """
        Computes the net optical density (netOD) for this dose in a specific channel.
        The global PV_before (pixel value before exposure) is provided, and the PV_after
        is taken from the computed pixel value for that channel.
        
        Parameters
        ----------
        PV_before : float
            The pixel value before exposure (defined at the global calibration level).
        channel : int, optional
            The image channel to use (default is 0).
        
        Returns
        -------
        float
            The computed net optical density for this dose in the given channel.
        
        Raises
        ------
        ValueError
            If the pixel value for the given channel has not been computed yet.
        """
        if channel not in self.pixel_values_after or self.pixel_values_after[channel] is None:
            raise ValueError("The average pixel value after exposure for channel {} has not been computed yet.".format(channel))
        
        self.netODs[channel] = np.log10(pixel_values_before[channel] / self.pixel_values_after[channel])
        return self.netODs[channel]


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
        return (f"CalibrationDose(value={self.value}, num_rois={self.get_roi_count()}, "
                f"pixel_value={self.pixel_value}, netOD={self.netOD})")
