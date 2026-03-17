import cv2
import numpy as np
from typing import Tuple

def calculate_homography_matrix(
    src_pts: np.ndarray, 
    dst_pts: np.ndarray
) -> np.ndarray:
    """
    Calculates the 3x3 Homography matrix mapping source pixels to destination pixels.
    """
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return matrix

def generate_birds_eye_view(
    image: np.ndarray, 
    matrix: np.ndarray, 
    output_size: Tuple[int, int]
) -> np.ndarray:
    """
    Warps the image using the Homography matrix to create a Bird's Eye View.
    """
    warped_img = cv2.warpPerspective(image, matrix, output_size)
    return warped_img

if __name__ == "__main__":
    # 1. Load the real frame from the Bursa intersection
    image_path = r"experiments\proj03_speed_radar\micro_lab_homography\intersection_02.jpg"
    image = cv2.imread(image_path)
    
    # Safety check: If OpenCV can't find the file, it loads 'None'.
    if image is None:
        print(f"Error: Could not load the image. Please check this path:\n{image_path}")
        exit()
    
    # 2. Define the 4 points in the original image (Pixel coordinates: x, y)
    # WARNING: These are still the "dummy" points! 
    # You need to replace these with the real road coordinates you find using MS Paint.
    SOURCE_POINTS = np.float32([
    [161, 303],
    [515, 294],
    [191, 363],
    [523, 357],
])
    # 3. Define where these points should map to in the Bird's Eye View
    # (Mapping to a perfect 400x400 rectangle)
    DESTINATION_POINTS = np.float32([
        [100, 0],    # Top-Left
        [300, 0],    # Top-Right
        [100, 400],  # Bottom-Left
        [300, 400]   # Bottom-Right
    ])
    
    # 4. Compute H matrix
    H = calculate_homography_matrix(SOURCE_POINTS, DESTINATION_POINTS)
    print("Calculated Homography Matrix (H):\n", H)
    
    # 5. Warp the image
    bev_image = generate_birds_eye_view(image, H, (400, 400))
    
    # 6. --- VISUALIZATION (Save to Disk in the correct folder) ---
    # Draw red dots on the original image so you can see where your SOURCE_POINTS are
    for pt in SOURCE_POINTS:
        cv2.circle(image, (int(pt[0]), int(pt[1])), 8, (0, 0, 255), -1)

    # Save the images directly to your micro_lab_homography folder
    save_path_1 = r"experiments\proj03_speed_radar\micro_lab_homography\01_original_with_dots.jpg"
    save_path_2 = r"experiments\proj03_speed_radar\micro_lab_homography\02_birds_eye_view.jpg"
    
    cv2.imwrite(save_path_1, image)
    cv2.imwrite(save_path_2, bev_image)

    print(f"Success! Images saved to:\n- {save_path_1}\n- {save_path_2}")