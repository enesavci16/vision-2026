import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Set


class LaneCounter:
    """Counts tracked objects that cross a predefined virtual line."""

    def __init__(
        self,
        line_pt1: Tuple[float, float],
        line_pt2: Tuple[float, float],
        history_size: int = 5,
    ):
        self.A = np.array(line_pt1)
        self.B = np.array(line_pt2)

        # Maps track_id -> deque of historical numpy arrays (positions)
        self.trajectories: Dict[int, deque] = {}

        # Keeps track of IDs that have already been counted to prevent double-counting
        self.counted_ids: Set[int] = set()

        self.total_count: int = 0
        self.history_size: int = history_size

    def _ccw(self, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> bool:
        """Determines if three points are listed in a counter-clockwise order."""
        # Principal standard: Explicit return required.
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def _intersects(self, C_t_minus_1: np.ndarray, C_t: np.ndarray) -> bool:
        """Checks if the trajectory segment intersects the virtual lane line."""
        # Condition 1: Do A and B straddle the car's path?
        straddle_1 = self._ccw(self.A, C_t_minus_1, C_t) != self._ccw(
            self.B, C_t_minus_1, C_t
        )

        # Condition 2: Does the car's path straddle line A-B?
        straddle_2 = self._ccw(self.A, self.B, C_t_minus_1) != self._ccw(
            self.A, self.B, C_t
        )

        return straddle_1 and straddle_2

    def update(self, active_tracks: List[Tuple[int, np.ndarray]]) -> int:
        """
        Ingests the output from MultiObjectTracker.
        Returns the updated total_count.
        """
        current_ids = set()

        for track_id, position in active_tracks:
            current_ids.add(track_id)

            if track_id not in self.trajectories:
                self.trajectories[track_id] = deque(maxlen=self.history_size)

            self.trajectories[track_id].append(position)

            if len(self.trajectories[track_id]) >= 2:
                C_t_minus_1 = self.trajectories[track_id][0]
                C_t = self.trajectories[track_id][-1]

                if self._intersects(C_t_minus_1, C_t):
                    if track_id not in self.counted_ids:
                        self.counted_ids.add(track_id)
                        self.total_count += 1

        # Memory management: remove tracks that disappeared
        stale_ids = list(self.trajectories.keys())
        for t_id in stale_ids:
            if t_id not in current_ids:
                del self.trajectories[t_id]

        return self.total_count
