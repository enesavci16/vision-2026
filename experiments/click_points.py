import cv2
import numpy as np
from typing import List, Any

# Global list to store our clicked coordinates
selected_points: List[List[int]] = []


def click_event(event: int, x: int, y: int, flags: int, param: Any) -> None:
    """
    OpenCV mouse callback function. Captures (x,y) on left mouse click.
    """
    global selected_points

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(selected_points) < 4:
            selected_points.append([x, y])
            print(f"Point {len(selected_points)} recorded: [{x}, {y}]")

            # Draw a visual marker (green circle) on the clicked location
            cv2.circle(image_display, (x, y), 6, (0, 255, 0), -1)
            cv2.imshow("Coordinate Extractor", image_display)

            # Once 4 points are clicked, output the final array
            if len(selected_points) == 4:
                print("\n" + "=" * 50)
                print("✅ 4 POINTS CAPTURED. COPY THE ARRAY BELOW:")
                print("=" * 50)
                print("SOURCE_POINTS = np.float32([")
                for pt in selected_points:
                    print(f"    [{pt[0]}, {pt[1]}],")
                print("])")
                print("=" * 50)
                print("Press any key in the image window to exit.")


if __name__ == "__main__":
    # Your exact local path
    image_path = (
        r"experiments\proj03_speed_radar\micro_lab_homography\intersection_02.jpg"
    )

    original_image = cv2.imread(image_path)

    # Safety check
    if original_image is None:
        print(f"Error: Could not load image. Check path:\n{image_path}")
        exit()

    image_display = original_image.copy()

    print("--- INSTRUCTIONS ---")
    print(
        "1. Find a real-world rectangle on the road (e.g., a lane segment between dashed lines)."
    )
    print(
        "2. Click 4 points in this EXACT order: Top-Left, Top-Right, Bottom-Left, Bottom-Right."
    )
    print("--------------------\n")

    cv2.namedWindow("Coordinate Extractor")
    cv2.setMouseCallback("Coordinate Extractor", click_event)

    cv2.imshow("Coordinate Extractor", image_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
