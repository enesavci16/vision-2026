import logging

logger = logging.getLogger(__name__)

MAX_RESOLUTION: int = 1280

class IntersectionCamera:
    """Kavşaktaki bir kamerayı temsil eder."""

    def __init__(self, camera_id: int, location: str,
        resolution: int = MAX_RESOLUTION,
    ) -> None:
        self.camera_id = camera_id
        self.location = location
        self.resolution = resolution

    def capture_frame(self, resolution: int) -> None:
        """Verilen çözünürlükte kare yakalar."""
        self._validate_resolution(resolution)
        logger.info(f"Kamera {self.camera_id} kare yakalıyor...")

    def _validate_resolution(self, resolution: int) -> None:
        """Çözünürlüğün izin verilen maksimumu aşmadığını doğrular."""
        if resolution > MAX_RESOLUTION:
            raise ValueError(
                f"Çözünürlük {resolution} maksimum {MAX_RESOLUTION} olabilir."
            )