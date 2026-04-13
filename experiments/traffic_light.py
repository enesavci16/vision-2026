import logging

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class TrafficManagementSystem:
    def __init__(
        self,
        initial_state: str = "red",
        red_time: float = 30.0,
        yellow_time: float = 30.0,
        green_time: float = 30.0,
    ):
        self.red_time = red_time
        self.yellow_time = yellow_time
        self.green_time = green_time
        valid_states = {"red", "yellow", "green"}
        if initial_state not in valid_states:
            raise ValueError(f"Geçersiz durum: {initial_state}")
        self.traffic_light: str = initial_state
        self.transition_count = 0
        self.is_faulty: bool = False

    def get_duration(self) -> float:
        durations = {
            "red": self.red_time,
            "green": self.green_time,
            "yellow": self.yellow_time,
        }

        return durations[self.traffic_light]

    def change_traffic_light(self, target_state: str | None = None) -> str:
        if self.is_faulty:
            raise ValueError("Lamba arızalı!")

        next_state = {
            "green": "yellow",
            "yellow": "red",
            "red": "green",
        }
        expected = next_state[self.traffic_light]

        if target_state is not None and target_state != expected:
            raise ValueError(f"Geçersiz geçiş: {self.traffic_light} → {target_state}")
        old_state = self.traffic_light
        self.traffic_light = expected
        self.transition_count += 1
        logger.info(f"{old_state} → {self.traffic_light}")
        return self.traffic_light

    def mark_faulty(self) -> None:
        self.is_faulty = True
        logger.warning("Lamba arızalı olarak işaretlendi")

    def run(self) -> None:
        duration = self.get_duration()
        self.change_traffic_light()
        logger.info(f"Traffic_light: {self.traffic_light}")

    def summary(self) -> dict:
        return {
            "state": self.traffic_light,
            "transitions": self.transition_count,
            "duration": self.get_duration(),
        }


if __name__ == "__main__":
    tms = TrafficManagementSystem(red_time=1.0, yellow_time=1.0, green_time=1.0)
    logger.info(tms.summary())
    tms.change_traffic_light()
    tms.change_traffic_light()
    logger.info(tms.summary())
    tms.mark_faulty()
