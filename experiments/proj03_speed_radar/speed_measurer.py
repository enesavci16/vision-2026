import logging
import numpy as np
from typing import Tuple, Dict, Any

# Logger configuration
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
MS_TO_KMH_MULTIPLIER = 3.6


class SpeedMeasurer:
    """Tracks objects and measures their speeds in Bird's Eye View (BEV) coordinates."""

    def __init__(self, pixel_per_meter: float) -> None:
        """Initializes the SpeedMeasurer.

        Args:
            pixel_per_meter (float): The ratio representing how many pixels correspond to 1 meter.

        Raises:
            ValueError: If pixel_per_meter is less than or equal to zero.
        """
        if pixel_per_meter <= 0:
            logger.error("Failed to initialize: pixel_per_meter must be positive.")
            raise ValueError("pixel_per_meter must be a positive value!")
        
        self.pixel_per_meter = pixel_per_meter
        self._history: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"SpeedMeasurer initialized successfully with ratio: {self.pixel_per_meter} px/m.")

    @staticmethod
    def compute_speed(
        pixel_per_meter: float, 
        first_coord: Tuple[float, float], 
        first_time: float, 
        second_coord: Tuple[float, float], 
        second_time: float
    ) -> float:
        """Computes the speed between two points in km/h.

        Args:
            pixel_per_meter (float): Pixel to meter ratio.
            first_coord (Tuple[float, float]): Previous BEV coordinate (x, y).
            first_time (float): Timestamp of the previous coordinate.
            second_coord (Tuple[float, float]): Current BEV coordinate (x, y).
            second_time (float): Timestamp of the current coordinate.

        Returns:
            float: Computed speed in kilometers per hour (km/h).

        Raises:
            ValueError: If the time difference (dt) is zero or negative.
        """
        dt = second_time - first_time
        if dt <= 0:
            logger.error("Invalid timestamps: Time difference (dt) cannot be zero or negative.")
            raise ValueError("Time difference must not be zero or negative!!!")
        
        # Euclidean distance calculation
        distance_pixel = np.sqrt((second_coord[1] - first_coord[1])**2 + (second_coord[0] - first_coord[0])**2)
        
        distance_meter = distance_pixel / pixel_per_meter
        distance_meter_s = distance_meter / dt
        distance_km_h = distance_meter_s * MS_TO_KMH_MULTIPLIER
        
        return distance_km_h

    def add_measurement(self, track_id: str, current_coord: Tuple[float, float], current_timestamp: float) -> None:
        """Adds a new measurement for a tracked object and computes its current speed if possible.

        Args:
            track_id (str): Unique identifier for the tracked object.
            current_coord (Tuple[float, float]): Current BEV coordinate (x, y).
            current_timestamp (float): Current timestamp in seconds.
        """
        if track_id not in self._history:
            self._history[track_id] = {
                "last_measurement": (current_coord, current_timestamp),
                "speeds": [], 
                "count": 1    
            }
            logger.debug(f"New track added: {track_id}")
        else:
            last_coord, last_time = self._history[track_id]["last_measurement"]
            
            calculated_speed = self.compute_speed(
                self.pixel_per_meter, 
                last_coord, 
                last_time, 
                current_coord, 
                current_timestamp
            )
            
            self._history[track_id]["speeds"].append(calculated_speed)
            self._history[track_id]["last_measurement"] = (current_coord, current_timestamp)
            self._history[track_id]["count"] += 1

    def get_report(self) -> Dict[str, Dict[str, Any]]:
        """Generates a speed report for all tracked objects.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary containing average_speed, max_speed, 
                                       current_speed, and measurement_count for each track_id.
        """
        report: Dict[str, Dict[str, Any]] = {}
        
        for track_id, data in self._history.items():
            report[track_id] = {}

            if len(data["speeds"]) > 0:
                average_speed = sum(data["speeds"]) / len(data["speeds"])
                max_speed = max(data["speeds"])
                current_speed = data["speeds"][-1]
            else:
                average_speed = 0.0
                max_speed = 0.0
                current_speed = 0.0

            report[track_id]["average_speed"] = average_speed
            report[track_id]["max_speed"] = max_speed
            report[track_id]["current_speed"] = current_speed
            report[track_id]["measurement_count"] = data["count"]

        logger.info(f"Report generated for {len(report)} tracks.")
        return report