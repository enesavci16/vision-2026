import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from filterpy.kalman import KalmanFilter
from typing import List, Tuple


class Track:
    """Represents a single tracked object (vehicle) using a 4D Kalman Filter."""

    def __init__(self, track_id: int, initial_detection: np.ndarray):
        self.track_id = track_id

        # 1. Initialize 4D State [x, y, v_x, v_y] and 2D Measurement [x, y]
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        # 2. State Transition (F): physics model for X and Y independently
        self.kf.F = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],  # x = x + v_x
                [0.0, 1.0, 0.0, 1.0],  # y = y + v_y
                [0.0, 0.0, 1.0, 0.0],  # v_x = v_x
                [0.0, 0.0, 0.0, 1.0],
            ]
        )  # v_y = v_y

        # 3. Measurement Function (H): We only measure x and y
        self.kf.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

        # 4. Covariance Matrices
        self.kf.P *= 1000.0  # High initial uncertainty
        self.kf.R *= 5.0  # YOLO sensor noise
        self.kf.Q *= 0.1  # Process noise

        # 5. Initial State [x, y, 0, 0]
        self.kf.x = np.array(
            [[initial_detection[0]], [initial_detection[1]], [0.0], [0.0]]
        )

        self.position = initial_detection
        self.time_since_update = 0
        self.hits = 1

    def predict(self):
        """Advances the state vector using the physics model."""
        self.kf.predict()
        # Update the position property for easy access
        self.position = np.array([self.kf.x[0, 0], self.kf.x[1, 0]])
        self.time_since_update += 1

    def update(self, detection: np.ndarray):
        """Corrects the prediction using the actual YOLO measurement."""
        z = np.array([[detection[0]], [detection[1]]])
        self.kf.update(z)
        self.position = np.array([self.kf.x[0, 0], self.kf.x[1, 0]])
        self.time_since_update = 0
        self.hits += 1


class MultiObjectTracker:
    """The central tracker that associates YOLO detections with existing Tracks."""

    def __init__(self, max_age: int = 30, max_distance: float = 50.0):
        self.tracks: List[Track] = []
        self.next_id: int = 1
        self.max_age: int = max_age
        # The maximum allowed distance (pixels/meters) to consider a match valid
        self.max_distance: float = max_distance

    def _compute_cost_matrix(
        self, track_positions: np.ndarray, detections: np.ndarray
    ) -> np.ndarray:
        """Calculates the Euclidean distance between all Tracks and all Detections."""
        return cdist(track_positions, detections, metric="euclidean")

    def update(self, detections: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        """The main pipeline loop for each frame of the video."""

        # STEP 1: Predict the next location of all existing tracks
        for track in self.tracks:
            track.predict()

        # Handle edge cases: No detections
        if len(detections) == 0:
            self._delete_old_tracks()
            return [
                (t.track_id, t.position) for t in self.tracks if t.time_since_update < 1
            ]

        # Handle edge cases: No tracks exist yet
        if len(self.tracks) == 0:
            for det in detections:
                self.tracks.append(Track(self.next_id, det))
                self.next_id += 1
            return [(t.track_id, t.position) for t in self.tracks]

        # STEP 2: Compute the Cost Matrix
        track_positions = np.array([t.position for t in self.tracks])
        cost_matrix = self._compute_cost_matrix(track_positions, detections)

        # STEP 3: The Hungarian Algorithm
        # track_indices[i] is matched with detection_indices[i]
        track_indices, detection_indices = linear_sum_assignment(cost_matrix)

        # We need to keep track of which detections and tracks were successfully matched
        unmatched_detections = set(range(len(detections)))
        matched_tracks = set()

        # STEP 4: Update matched tracks (with a Distance Gate)
        for track_idx, det_idx in zip(track_indices, detection_indices):
            distance = cost_matrix[track_idx, det_idx]

            # Distance Gate: Only update if the YOLO box is reasonably close to the prediction
            if distance <= self.max_distance:
                self.tracks[track_idx].update(detections[det_idx])
                unmatched_detections.remove(det_idx)
                matched_tracks.add(track_idx)

        # STEP 5: Create NEW tracks for detections that didn't get a match
        for det_idx in unmatched_detections:
            self.tracks.append(Track(self.next_id, detections[det_idx]))
            self.next_id += 1

        # STEP 6: Delete old tracks
        self._delete_old_tracks()

        # Return the active IDs and their positions
        # We only return tracks that have been updated recently to avoid drawing "ghosts"
        return [
            (t.track_id, t.position) for t in self.tracks if t.time_since_update < 1
        ]

    def _delete_old_tracks(self):
        """Removes tracks that haven't been seen by YOLO for 'max_age' frames."""
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
