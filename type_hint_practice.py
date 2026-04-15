import logging
from ultralytics import YOLO
import numpy as np

logger = logging.getLogger(__name__)


class TrafficDetector:
    """Adapter class for YOLOv8 to standardize interactions
    across the Vision-2026 architecture.
    """

    # TODO: model_path parametresine type hint ekle
    def __init__(self, model_path="yolov8n.pt") -> None:
        self.model_path = model_path
        self.model = YOLO(model_path)

    # TODO: frame parametresine ve return değerine type hint ekle
    def detect(self, frame: np.ndarray, confidence: float = 0.5) -> list:
        results = self.model.predict(frame, conf=confidence)
        return results

    # TODO: tüm parametrelere ve return değerine type hint ekle
    def _preprocess(self, frame: np.ndarray, target_size: int) -> np.ndarray:
        resized = frame[:target_size, :target_size]
        normalized = resized / 255.0
        return normalized

    # TODO: parametrelere ve return değerine type hint ekle
    def get_model_info(self) -> dict:
        return {
            "model_path": self.model_path,
            "model_type": type(self.model).__name__,
        }
