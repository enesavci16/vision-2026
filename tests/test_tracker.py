import pytest
import numpy as np

# Assuming your LaneCounter is saved in src/tracker/counter.py
from src.tracker.counter import LaneCounter


def test_successful_lane_crossing():
    """Simulates a vehicle driving directly across a horizontal virtual line."""
    # Setup: A horizontal line from x=0 to x=10 at y=5
    counter = LaneCounter(line_pt1=(0.0, 5.0), line_pt2=(10.0, 5.0))

    # Frame 1: Vehicle 1 is above the line
    count = counter.update([(1, np.array([5.0, 2.0]))])
    assert count == 0, "Count should be 0 before crossing."

    # Frame 2: Vehicle 1 has moved below the line
    count = counter.update([(1, np.array([5.0, 8.0]))])
    assert count == 1, "Count must increment to 1 after crossing the line."


def test_parallel_movement_no_crossing():
    """Simulates a vehicle driving parallel to the line without crossing it."""
    counter = LaneCounter(line_pt1=(0.0, 5.0), line_pt2=(10.0, 5.0))

    # Frame 1: Vehicle 2 starts above the line
    counter.update([(2, np.array([2.0, 2.0]))])

    # Frame 2: Vehicle 2 moves right, staying above the line
    count = counter.update([(2, np.array([8.0, 2.0]))])
    assert count == 0, "Count must remain 0 if the vehicle does not cross the line."


def test_double_count_prevention():
    """Ensures a single vehicle ID is never counted twice."""
    counter = LaneCounter(line_pt1=(0.0, 5.0), line_pt2=(10.0, 5.0))

    # Frame 1: Vehicle 3 approaches
    counter.update([(3, np.array([5.0, 4.0]))])

    # Frame 2: Vehicle 3 crosses
    count = counter.update([(3, np.array([5.0, 6.0]))])
    assert count == 1, "Count should be 1 after initial crossing."

    # Frame 3: Vehicle 3 continues moving forward
    count = counter.update([(3, np.array([5.0, 8.0]))])
    assert count == 1, "Count must not increment a second time for the same vehicle ID."
