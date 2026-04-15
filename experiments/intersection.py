import time
import logging

from traffic_light import TrafficLight

logger = logging.getLogger(__name__)


class Intersection:
    """An intersection with two perpendicular traffic lights."""

    def __init__(self):
        """Initialize intersection with North-South (red) and East-West (green) lights."""
        self.ns_light = TrafficLight(initial_state="red", direction="North-South")
        self.ew_light = TrafficLight(initial_state="green", direction="East-West")

    def switch_lights(self) -> None:
        """Perform a full coordinated switch cycle.

        East-West goes through yellow phase, then North-South activates.
        """
        self.ew_light.change_traffic_light()  # green -> green_yellow
        time.sleep(1)
        self.ew_light.change_traffic_light()  # green_yellow -> red
        self.ns_light.change_traffic_light()  # red -> red_yellow