import logging
from typing import Tuple, Dict, Any
import numpy as np

METER_S_TO_KM_H = 3.6

logging.basicConfig(level=logging.DEBUG, format="%(name)s - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class SpeedMeasurement:
    """System to calculate vehicle speeds from bird's eye view (BEV) coordinates."""

    def __init__(self, intersection_id: str, pixel_per_meter: float):
        """Initialize the SpeedMeasurement system.

        Args:
            intersection_id: Unique identifier of the intersection.
            pixel_per_meter: Calibration ratio in BEV (pixels per meter).
        """
        self.intersection_id = intersection_id
        self.pixel_per_meter = pixel_per_meter
        self.measurements = {}

    def add_measurement(
        self, vehicle_id: str, bev_coord: Tuple[float, float], timestamp: float
    ) -> None:
        """Record a single location and time measurement for a vehicle.

        If the vehicle is new, a list is created. If it already exists,
        the measurement is appended to the existing list.

        Args:
            vehicle_id: Unique identifier of the vehicle.
            bev_coord: The (x, y) pixel coordinates of the vehicle in BEV.
            timestamp: Time of the measurement in seconds.
        """
        if vehicle_id not in self.measurements:
            self.measurements[vehicle_id] = [(bev_coord[0], bev_coord[1], timestamp)]
        else:
            self.measurements[vehicle_id].append(
                (bev_coord[0], bev_coord[1], timestamp)
            )

    @staticmethod
    def compute_speed(
        bev_coord1: Tuple[float, float],
        timestamp1: float,
        bev_coord2: Tuple[float, float],
        timestamp2: float,
        pixel_per_meter: float,
    ) -> float:
        """Calculate the speed between two points in km/h.

        Args:
            bev_coord1: The (x, y) pixel coordinates of the first measurement.
            timestamp1: Time of the first measurement in seconds.
            bev_coord2: The (x, y) pixel coordinates of the second measurement.
            timestamp2: Time of the second measurement in seconds.
            pixel_per_meter: Conversion ratio from pixels to meters.

        Returns:
            Calculated speed in km/h. Returns 0.0 if the time difference is zero
            to prevent division by zero errors.
        """
        distance_pixel = np.sqrt(
            (bev_coord2[1] - bev_coord1[1]) ** 2 + (bev_coord2[0] - bev_coord1[0]) ** 2
        )
        time = timestamp2 - timestamp1

        if time == 0:
            return 0.0

        speed_km = ((distance_pixel / pixel_per_meter) / time) * METER_S_TO_KM_H
        return speed_km

    def get_report(self) -> Dict[str, Dict[str, Any]]:
        """Generate a comprehensive speed report for all measured vehicles.

        Only vehicles with at least two measurements are included. Calculates
        current, average, and maximum speeds for each vehicle.

        Returns:
            Dictionary containing detailed speed metrics for each vehicle.
            Example structure:
            {
                "01": {
                    "current_speed": 7.2,
                    "avg_speed": 5.4,
                    "max_speed": 7.2,
                    "measurement_count": 3
                }
            }
        """
        report_dict = {}

        for v_id, records in self.measurements.items():
            measurement_count = len(records)

            if measurement_count < 2:
                continue

            vehicle_speeds = []

            for i in range(measurement_count - 1):
                coord1 = (records[i][0], records[i][1])
                time1 = records[i][2]
                coord2 = (records[i + 1][0], records[i + 1][1])
                time2 = records[i + 1][2]

                speed = SpeedMeasurement.compute_speed(
                    coord1, time1, coord2, time2, self.pixel_per_meter
                )
                vehicle_speeds.append(speed)

            current_speed = vehicle_speeds[-1]
            avg_speed = sum(vehicle_speeds) / len(vehicle_speeds)
            max_speed = max(vehicle_speeds)

            report_dict[v_id] = {
                "current_speed": current_speed,
                "avg_speed": avg_speed,
                "max_speed": max_speed,
                "measurement_count": measurement_count,
            }

            logger.info(
                f"Vehicle {v_id} -> Current: {current_speed:.1f}, Avg: {avg_speed:.1f}, Max: {max_speed:.1f}, Count: {measurement_count}"
            )

        return report_dict


if __name__ == "__main__":
    deneme = SpeedMeasurement("001", 100.0)

    deneme.add_measurement("01", (10.0, 10.0), 1235.0)
    deneme.add_measurement("01", (110.0, 10.0), 1236.0)
    deneme.add_measurement("01", (310.0, 10.0), 1237.0)

    rapor = deneme.get_report()
    # print yerine projenin logger'ı kullanıldı
    logger.info(f"Final Report: {rapor}")
