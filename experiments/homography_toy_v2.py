import cv2
import numpy as np
from typing import Tuple

def calculate_homography_matrix(
    src_pts: np.ndarray, 
    dst_pts: np.ndarray
) -> np.ndarray:
    """Calculates the 3x3 Homography matrix."""
    return cv2.getPerspectiveTransform(src_pts, dst_pts)

def generate_birds_eye_view(
    image: np.ndarray, 
    matrix: np.ndarray, 
    output_size: Tuple[int, int]
) -> np.ndarray:
    """Warps the image to the specified high-resolution size."""
    return cv2.warpPerspective(image, matrix, output_size)

if __name__ == "__main__":
    # 1. Path Setup
    image_path = r"experiments\proj03_speed_radar\micro_lab_homography\intersection_02.jpg"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load the image at {image_path}")
        exit()
    
    # 2. Resolution Settings (800 Width x 1200 Height)
    BEV_W = 800
    BEV_H = 1200
    
    # 3. Source Points (PASTE YOUR CLICKED COORDINATES HERE)
    # Replace these with the actual 4 points you got from click_points.py
    SOURCE_POINTS = np.float32([
        [449, 313],  # Top-Left (example)
        [542, 310],  # Top-Right (example)
        [157, 725],  # Bottom-Left (example)
        [982, 715]   # Bottom-Right (example)
    ])
    
    # 4. Destination Points (Scaled to 800x1200)
    # Mapping to the edges of our new high-res canvas
    DESTINATION_POINTS = np.float32([
        [0, 0],          # Top-Left
        [BEV_W, 0],      # Top-Right
        [0, BEV_H],      # Bottom-Left
        [BEV_W, BEV_H]   # Bottom-Right
    ])
    
    # 5. Math & Processing
    H = calculate_homography_matrix(SOURCE_POINTS, DESTINATION_POINTS)
    bev_image = generate_birds_eye_view(image, H, (BEV_W, BEV_H))
    
    # 6. Save Results
    # Draw markers on original for verification
    for pt in SOURCE_POINTS:
        cv2.circle(image, (int(pt[0]), int(pt[1])), 10, (0, 0, 255), -1)

    save_path_orig = r"experiments\proj03_speed_radar\micro_lab_homography\01_original_highres_dots.jpg"
    save_path_bev = r"experiments\proj03_speed_radar\micro_lab_homography\02_birds_eye_800x1200.jpg"
    
    cv2.imwrite(save_path_orig, image)
    cv2.imwrite(save_path_bev, bev_image)

    print(f"SUCCESS: High-Res BEV generated at {BEV_W}x{BEV_H}")
    print(f"Matrix H calculated as:\n{H}")