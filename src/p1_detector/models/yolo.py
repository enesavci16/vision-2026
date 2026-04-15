import logging
from typing import Dict, Any, Optional
from ultralytics import YOLO

# Configure Logging (Industry Standard)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TrafficDetector:
    """
    Adapter class for YOLOv8 to standardize interactions
    across the Vision-2026 architecture.
    """

    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model_path = model_path
        try:
            logger.info(f"⚖️ Initializing Model: {model_path}")
            self.model = YOLO(model_path)
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

    def train(self, config: Dict[str, Any]) -> None:
        """
        Executes the training pipeline with validated config.
        """
        logger.info("🚀 Starting Training Pipeline...")
        try:
            results = self.model.train(
                data=config["data_yaml"],
                epochs=config["epochs"],
                imgsz=config["image_size"],
                batch=config["batch_size"],
                project=config["project_name"],
                name=config["run_name"],
                verbose=True,
            )
            logger.info(f"✅ Training Complete. stored in {results.save_dir}")
        except Exception as e:
            logger.critical(f"💥 Training Crash: {e}")
            raise

    def predict(self, source: str) -> list:
        return self.model(source)
