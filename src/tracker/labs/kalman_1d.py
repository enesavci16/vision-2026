import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter


def run_1d_simulation():
    # --- 1. INITIALIZE THE FILTER ---
    kf = KalmanFilter(dim_x=2, dim_z=1)

    # State Transition Matrix (F): x = x_0 + v*dt
    kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])

    # Measurement Function (H): We only measure position
    kf.H = np.array([[1.0, 0.0]])

    # Covariance Matrices
    kf.P *= 1000.0  # High initial uncertainty
    kf.R = np.array([[5.0]])  # Sensor noise (YOLO variance)
    kf.Q = np.array([[0.1, 0.0], [0.0, 0.1]])  # Process noise (Small accelerations)

    # Initial State Guess [Position=0, Velocity=10m/s]
    kf.x = np.array([[0.0], [10.0]])

    # --- 2. SETUP FOR LOGGING ---
    true_positions = []
    yolo_measurements = []
    kalman_estimates = []

    true_pos = 0.0
    true_velocity = 10.0

    print("Starting 50-Frame Simulation...")

    # --- 3. THE 50-FRAME SIMULATION LOOP ---
    for step in range(50):
        # A. Simulate Reality (Physics)
        true_pos += true_velocity
        true_positions.append(true_pos)

        # B. Simulate Noisy YOLO Sensor
        z = np.random.normal(true_pos, np.sqrt(kf.R[0, 0]))
        yolo_measurements.append(z)

        # C. Kalman PREDICT Step
        kf.predict()

        # D. Kalman UPDATE Step
        kf.update(np.array([[z]]))

        # E. Save the Kalman Estimate
        kalman_estimates.append(kf.x[0, 0])

    # --- 4. VISUALIZATION ---
    plt.figure(figsize=(10, 5))
    plt.plot(
        true_positions, label="True Position (Reality)", linestyle="--", color="black"
    )
    plt.scatter(
        range(50),
        yolo_measurements,
        label="YOLO Detections (Noisy)",
        color="red",
        marker="x",
    )
    plt.plot(
        kalman_estimates, label="Kalman Filter Estimate", color="blue", linewidth=2
    )
    plt.title("1D Kalman Filter: Smoothing YOLO Detections")
    plt.xlabel("Time Step (Frames)")
    plt.ylabel("Position (Meters)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run_1d_simulation()
