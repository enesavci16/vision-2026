import cv2
import logging
from pathlib import Path

def parse_line(line: str) -> tuple[str, int, int, int, int, str]:
    """Parse a single annotation line.

    Expected format:
        filename x y w h plate_text

    Args:
        line: A single annotation line from the dataset.

    Returns:
        A tuple containing:
            - filename (str)
            - x (int)
            - y (int)
            - w (int)
            - h (int)
            - plate_text (str)
    """
    filename, x, y, w, h, plate_text = line.strip().split("\t")
    return filename, int(x), int(y), int(w), int(h), plate_text

def get_image_size(image_path: str) -> tuple[int, int]:
    """Read an image from disk and return its dimensions using OpenCV.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tuple[int, int]: A tuple containing (width, height).

    Raises:
        ValueError: If the image cannot be read (e.g., invalid path or corrupted file).
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image could not be read: {image_path}")
    height, width = image.shape[:2]
    return width, height

def corner_to_center(x: int, y: int, w: int, h: int) -> tuple[float, float]:
    """Convert top-left corner coordinates to center coordinates.

    Args:
        x: Top-left x coordinate.
        y: Top-left y coordinate.
        w: Width value.
        h: Height value.

    Returns:
        tuple[float, float]: (cx, cy) center coordinates.

    Raises:
        ValueError: If w or h are negative or zero.
    """
    if w <= 0:
        raise ValueError("w must be a positive value.")
    if h <= 0:
        raise ValueError("h must be a positive value.")
    c_x = x + (w / 2.0)
    c_y = y + (h / 2.0)
    return float(c_x), float(c_y)

def normalize_bbox(
    c_x: float, 
    c_y: float, 
    w: int, 
    h: int, 
    img_width: int, 
    img_height: int
) -> tuple[float, float, float, float]:
    """Normalize center coordinates and dimensions relative to image size.

    Args:
        c_x: Center x coordinate.
        c_y: Center y coordinate.
        w: Bounding box width.
        h: Bounding box height.
        img_width: Total width of the image.
        img_height: Total height of the image.

    Returns:
        tuple[float, float, float, float]: Normalized (norm_cx, norm_cy, norm_w, norm_h).

    Raises:
        ValueError: If img_width or img_height are negative or zero.
    """
    if img_width <= 0:
        raise ValueError("img_width must be a positive value.")
    if img_height <= 0:
        raise ValueError("img_height must be a positive value.")
    norm_cx = c_x / img_width
    norm_cy = c_y / img_height
    norm_w = w / img_width
    norm_h = h / img_height
    return norm_cx, norm_cy, norm_w, norm_h

def write_yolo_txt(
    image_path: str, 
    norm_cx: float, 
    norm_cy: float, 
    norm_w: float, 
    norm_h: float, 
    class_id: int
) -> None:
    """Write a single YOLO format annotation to a text file derived from the image path.

    Args:
        image_path: Path to the image, used to automatically derive the .txt output path.
        norm_cx: Normalized center x coordinate.
        norm_cy: Normalized center y coordinate.
        norm_w: Normalized bounding box width.
        norm_h: Normalized bounding box height.
        class_id: Class label integer for the object.

    Raises:
        ValueError: If any normalized value is outside the [0.0, 1.0] range.
    """
    if not (0.0 <= norm_cx <= 1.0) or not (0.0 <= norm_cy <= 1.0):
        raise ValueError("Normalized coordinates (norm_cx, norm_cy) must be between 0.0 and 1.0")
    
    if not (0.0 <= norm_w <= 1.0) or not (0.0 <= norm_h <= 1.0):
        raise ValueError("Normalized dimensions (norm_w, norm_h) must be between 0.0 and 1.0")

    output_path = Path(image_path).with_suffix(".txt")
    
    with output_path.open("a", encoding="utf-8") as file:
        file.write(f"{class_id} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")

def convert_dataset_to_yolo(input_folder: str, output_folder: str, default_class_id: int = 0) -> None:
    """Process a dataset folder to convert bounding box annotations to YOLO format.

    Reads all .txt files in the input_folder, finds the corresponding .jpg images to 
    extract dimensions, converts corner coordinates to normalized center coordinates, 
    and writes the YOLO formatted lines to the output_folder.

    Args:
        input_folder: Directory path containing the raw .txt annotations and .jpg images.
        output_folder: Directory path where the YOLO formatted .txt files will be saved.
        default_class_id: The integer class ID to use for YOLO formatting.
        
    Raises:
        OSError: If there are permission issues creating the output directory.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    output_path.mkdir(parents=True, exist_ok=True)

    for txt_file in input_path.glob("*.txt"):
        
        img_file = txt_file.with_suffix(".jpg")
        
        if img_file.exists():
            img_width, img_height = get_image_size(str(img_file))
            
            target_img_path = output_path / img_file.name
            
            target_txt_path = target_img_path.with_suffix(".txt")
            if target_txt_path.exists():
                target_txt_path.unlink()
            
            with txt_file.open("r", encoding="utf-8") as file:
                for line in file:
                    clean_line = line.strip()
                    if not clean_line:
                        continue
                        
                    _, x, y, w, h, _ = parse_line(clean_line)
                    c_x, c_y = corner_to_center(x, y, w, h)
                    norm_cx, norm_cy, norm_w, norm_h = normalize_bbox(
                        c_x, c_y, w, h, img_width, img_height
                    )
                    
                    write_yolo_txt(
                        image_path=str(target_img_path), 
                        norm_cx=norm_cx, 
                        norm_cy=norm_cy, 
                        norm_w=norm_w, 
                        norm_h=norm_h, 
                        class_id=default_class_id
                    )
        else:
            logging.warning(f"{txt_file.name} için karşılık gelen {img_file.name} bulunamadı. Atlanıyor.")

if __name__ == "__main__":
    # Loglama konfigürasyonunu sadece bu script ana modül olarak çalıştırıldığında devreye alıyoruz.
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    INPUT_DIR = r"datasets\openalpr_benchmarks\endtoend\eu"
    OUTPUT_DIR = r"datasets\openalpr_benchmarks\endtoend\yolo_labels"
    CLASS_ID = 0
    
    logging.info(f"Dönüşüm işlemi başlıyor...\nKaynak: {INPUT_DIR}\nHedef: {OUTPUT_DIR}")
    
    try:
        convert_dataset_to_yolo(
            input_folder=INPUT_DIR, 
            output_folder=OUTPUT_DIR, 
            default_class_id=CLASS_ID
        )
        logging.info("Dönüşüm başarıyla tamamlandı!")
    except Exception as e:
        logging.error(f"Dönüşüm sırasında beklenmeyen bir hata oluştu: {e}")