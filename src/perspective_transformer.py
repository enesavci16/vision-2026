import cv2
import logging
import numpy as np
from itertools import combinations

# Logger configuration
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class PerspectiveTransformer:
    """Transforms coordinates between camera view and Bird's Eye View (BEV)."""

    def __init__(self, src_points: np.ndarray, dst_points: np.ndarray) -> None:
        """Initializes the PerspectiveTransformer.

        Args:
            src_points (np.ndarray): Array of 4 source points of shape (4, 2).
            dst_points (np.ndarray): Array of 4 destination points of shape (4, 2).

        Raises:
            ValueError: If input shapes are invalid or points are collinear.
            RuntimeError: If the homography matrix cannot be computed.
        """
        # Validate before mutate pattern
        PerspectiveTransformer._validate_points(src_points, dst_points)

        self.src_points = src_points
        self.dst_points = dst_points
        self.H = self.compute_homography(self.src_points, self.dst_points)
        self.H_inv = np.linalg.inv(self.H)

        logger.info("PerspectiveTransformer initialized successfully.")

    @staticmethod
    def _validate_points(src_points: np.ndarray, dst_points: np.ndarray) -> None:
        """Validates the input points for shape and collinearity.

        Args:
            src_points (np.ndarray): Source points to validate.
            dst_points (np.ndarray): Destination points to validate.

        Raises:
            ValueError: If inputs do not have shape (4, 2) or contain collinear points.
        """
        if src_points.shape != (4, 2) or dst_points.shape != (4, 2):
            raise ValueError("Input shape must be (4, 2) for both source and destination points.")

        # Check collinearity for both source and destination points
        for points, name in [(src_points, "Source"), (dst_points, "Destination")]:
            for subset in combinations(points, 3):
                p1, p2, p3 = subset
                v1 = p2 - p1
                v2 = p3 - p1
                if abs(np.cross(v1, v2)) < 1e-5:
                    raise ValueError(
                        f"{name} points are collinear! Homography cannot be computed. "
                        "Please select 4 points that form a polygon."
                    )

    @staticmethod
    def compute_homography(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
        """Computes the homography matrix using RANSAC.

        Args:
            src_points (np.ndarray): Source points.
            dst_points (np.ndarray): Destination points.

        Returns:
            np.ndarray: The computed 3x3 homography matrix.

        Raises:
            RuntimeError: If the homography matrix computation fails.
        """
        H, mask = cv2.findHomography(
            src_points,
            dst_points,
            cv2.RANSAC,
            5.0
        )
        if H is None:
            raise RuntimeError("Failed to compute homography matrix!")

        logger.info(f"Inliers: {mask.sum()}/4")
        return H

    def to_bev(self, pixel_coords: np.ndarray) -> np.ndarray:
        """Transforms camera pixel coordinates to Bird's Eye View (BEV) coordinates.

        Args:
            pixel_coords (np.ndarray): Camera pixel coordinates of shape (N, 2).

        Returns:
            np.ndarray: Transformed BEV coordinates of shape (N, 2).
        """
        # Cast to float32 to prevent OpenCV type errors
        safe_coords = pixel_coords.astype(np.float32)
        reshaped_bev = safe_coords.reshape(-1, 1, 2)

        transformed_coords = cv2.perspectiveTransform(reshaped_bev, self.H)
        bev_result = transformed_coords.reshape(-1, 2)

        return bev_result

    def to_pixel(self, bev_coords: np.ndarray) -> np.ndarray:
        """Transforms Bird's Eye View (BEV) coordinates back to camera pixel coordinates.

        Args:
            bev_coords (np.ndarray): BEV coordinates of shape (N, 2).

        Returns:
            np.ndarray: Transformed camera pixel coordinates of shape (N, 2).
        """
        # Cast to float32 to prevent OpenCV type errors
        safe_coords = bev_coords.astype(np.float32)
        reshaped_pix = safe_coords.reshape(-1, 1, 2)

        transformed_coords = cv2.perspectiveTransform(reshaped_pix, self.H_inv)
        pix_result = transformed_coords.reshape(-1, 2)

        return pix_result


# ==========================================
# MANUAL TEST AND USAGE SCENARIO
# ==========================================
if __name__ == "__main__":
    logger.info("--- 1. DATA DEFINITION ---")
    src_points = np.array([
        [0, 0],
        [200, 10],
        [180, 120],
        [10, 100]
    ], dtype=np.float32)

    dst_points = np.array([
        [30, 20],
        [220, 10],
        [210, 150],
        [50, 140]
    ], dtype=np.float32)

    logger.debug(f"Source Points Shape: {src_points.shape}, Type: {src_points.dtype}")
    logger.debug(f"Destination Points Shape: {dst_points.shape}, Type: {dst_points.dtype}")

    logger.info("--- 2. CLASS INITIALIZATION & MATRICES ---")
    transformer = PerspectiveTransformer(src_points=src_points, dst_points=dst_points)

    logger.info(f"Homography Matrix (H):\n{np.round(transformer.H, 3)}")
    logger.info(f"Inverse Homography Matrix (H_inv):\n{np.round(transformer.H_inv, 3)}")

    logger.info("--- 3. FORWARD TRANSFORM TEST (Camera -> BEV) ---")
    test_camera_pixels = np.array([
        [100, 50],
        [150, 80]
    ], dtype=np.float32)

    bev_results = transformer.to_bev(test_camera_pixels)
    logger.info(f"Input (Camera Pixels):\n{test_camera_pixels}")
    logger.info(f"Output (BEV Pixels):\n{bev_results}")

    logger.info("--- 4. REVERSE TRANSFORM TEST (BEV -> Camera) ---")
    reverse_results = transformer.to_pixel(bev_results)
    logger.info(f"Input (BEV Pixels):\n{bev_results}")
    logger.info(f"Output (Reverted Camera Pixels):\n{np.round(reverse_results, 1)}")

    logger.info("--- 5. STATIC METHOD TEST ---")
    den_hom = PerspectiveTransformer.compute_homography(src_points, dst_points)
    logger.info(f"Static Method Computed H Matrix:\n{np.round(den_hom, 3)}")