import logging

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class TrafficLight:
    """Traffic light with 4-state cycle: red -> red_yellow -> green -> green_yellow."""

    def __init__(
        self,
        direction: str,
        initial_state: str = "red",
        red_time: float = 30.0,
        red_yellow_time: float = 5.0,
        green_time: float = 30.0,
        green_yellow_time: float = 5.0,
    ):
        """Initialize the traffic light.

        Args:
            direction: Direction of the light (e.g., 'North-South', 'East-West').
            initial_state: Starting state. One of: red, red_yellow, green, green_yellow.
            red_time: Duration of red state in seconds.
            red_yellow_time: Duration of red+yellow warning state in seconds.
            green_time: Duration of green state in seconds.
            green_yellow_time: Duration of green+yellow transition state in seconds.
        """
        self.direction = direction
        self.red_time = red_time
        self.green_time = green_time
        self.red_yellow_time = red_yellow_time
        self.green_yellow_time = green_yellow_time

        valid_states = {"red", "green", "red_yellow", "green_yellow"}
        if initial_state not in valid_states:
            raise ValueError(f"Invalid state: {initial_state}")

        self.traffic_light: str = initial_state
        self.transition_count = 0
        self.is_faulty: bool = False

    def get_duration(self) -> float:
        """Return the duration of the current state in seconds."""
        durations = {
            "red": self.red_time,
            "green": self.green_time,
            "red_yellow": self.red_yellow_time,
            "green_yellow": self.green_yellow_time,
        }
        return durations[self.traffic_light]

    def change_traffic_light(self, target_state: str | None = None) -> str:
        """Transition to the next state in the cycle.

        Args:
            target_state: Optional explicit target. If provided, must match expected next state.

        Returns:
            The new state after transition.

        Raises:
            ValueError: If the light is faulty or target_state is invalid.
        """
        if self.is_faulty:
            raise ValueError("Light is faulty!")

        next_state = {
            "green": "green_yellow",
            "green_yellow": "red",
            "red": "red_yellow",
            "red_yellow": "green",
        }
        expected = next_state[self.traffic_light]

        if target_state is not None and target_state != expected:
            raise ValueError(f"Invalid transition: {self.traffic_light} -> {target_state}")

        old_state = self.traffic_light
        self.traffic_light = expected
        self.transition_count += 1
        logger.info(f"{old_state} -> {self.traffic_light}")
        return self.traffic_light

    def mark_faulty(self) -> None:
        """Mark the light as faulty (out of service)."""
        self.is_faulty = True
        logger.warning("Light marked as faulty")

    def summary(self) -> dict:
        """Return a summary dict of the current state, transitions, and duration."""
        return {
            "direction": self.direction,
            "state": self.traffic_light,
            "transitions": self.transition_count,
            "duration": self.get_duration(),
        }