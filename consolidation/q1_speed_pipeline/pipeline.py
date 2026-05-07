import logging
import json
from typing import Dict, Any, Optional

import cv2
import numpy as np

from p1_detector.models.yolo import TrafficDetector
from speed_measurer import SpeedMeasurer
from perspective_transformer import PerspectiveTransformer


logger = logging.getLogger(__name__)


class IntersectionPipeline:
    """Pipeline for vehicle speed measurement at intersections.

    Loads video, runs YOLO detection,
    BEV transformation, and speed estimation end-to-end.

    Args:
        config: Dictionary containing pipeline configuration.
                See config.py for required keys.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        logger.info("⚙️ Pipeline başlatılıyor...")

        self.config = config

        self.video_path: str = self.config.get("VIDEO_PATH", "")
        self.model_path: str = self.config.get("MODEL_PATH", "yolov8n.pt")
        self.confidence_thr: float = self.config.get("CONFIDENCE_THR", 0.5)
        self.bev_width: int = self.config.get("BEV_WIDTH", 640)
        self.bev_height: int = self.config.get("BEV_HEIGHT", 640)
        self.pixel_per_meter: float = self.config.get("PIXEL_PER_METER", 10.0)
        self.output_dir: str = self.config.get("OUTPUT_DIR", "")
        self.output_json: str = self.config.get("OUTPUT_JSON", "speed_report.json")
        self.output_video: str = self.config.get("OUTPUT_VIDEO", "annotated.mp4")
        self.src_points: Optional[np.ndarray] = self.config.get("SRC_POINTS", None)
        self.dst_points: Optional[np.ndarray] = self.config.get("DST_POINTS", None)

        if not self.video_path:
            raise ValueError("VIDEO_PATH boş olamaz!")
        if self.src_points is None or self.dst_points is None:
            raise ValueError("SRC_POINTS ve DST_POINTS config içinde olmalı!")

        self.cap: Optional[cv2.VideoCapture] = None
        self.video_writer: Optional[cv2.VideoWriter] = None

        self.detector = TrafficDetector(model_path=self.model_path)
        self.perspective_transformer = PerspectiveTransformer(self.src_points, self.dst_points)
        self.speed_measurer = SpeedMeasurer(pixel_per_meter=self.pixel_per_meter)

        logger.info("✅ Pipeline bileşenleri başarıyla yüklendi.")

    def run(self) -> None:
        """Execute the full detection-tracking-speed pipeline."""
        logger.info("🎥 Video açılıyor...")

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Video açılamadı: {self.video_path}")

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"📌 Video info: fps={fps}, width={width}, height={height}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(self.output_video, fourcc, fps, (width, height))
        if not self.video_writer.isOpened():
            raise RuntimeError(f"VideoWriter açılamadı: {self.output_video}")
        logger.info("✅ VideoWriter hazırlandı.")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.info("📌 Video bitti, döngü durduruluyor...")
                break

            timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            results = self.detector.model.track(frame, persist=True, verbose=False)

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    if box.id is None:
                        continue
                    track_id = int(box.id[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    bev_coord = self.perspective_transformer.to_bev(
                        np.array([cx, cy], dtype=np.float32)
                    ).flatten()
                    self.speed_measurer.update(track_id, bev_coord, timestamp)

                    speed_report = self.speed_measurer.get_report()
                    logger.debug(f"Report keys: {list(speed_report.keys())}, track_id: {track_id}")

                    speed_kmh = 0.0
                    if track_id in speed_report:
                        speed_kmh = speed_report[track_id].get("average_speed", 0.0)

                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"ID:{track_id} {speed_kmh:.1f} km/h",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

            if len(frame.shape) == 2 or frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            self.video_writer.write(frame)

        self.cap.release()
        self.video_writer.release()
        logger.info(f"✅ Video kaydedildi: {self.output_video}")

        final_report = self.speed_measurer.get_report()
        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(
                final_report,
                f,
                indent=4,
                default=lambda x: float(x) if hasattr(x, "item") else x,
            )
        logger.info(f"✅ Speed report kaydedildi: {self.output_json}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    import config

    CONFIG = {
        "VIDEO_PATH": config.VIDEO_PATH,
        "MODEL_PATH": config.MODEL_PATH,
        "CONFIDENCE_THR": config.CONFIDENCE_THR,
        "BEV_WIDTH": config.BEV_WIDTH,
        "BEV_HEIGHT": config.BEV_HEIGHT,
        "PIXEL_PER_METER": config.PIXEL_PER_METER,
        "OUTPUT_DIR": config.OUTPUT_DIR,
        "OUTPUT_JSON": config.OUTPUT_JSON,
        "OUTPUT_VIDEO": config.OUTPUT_VIDEO,
        "SRC_POINTS": config.SRC_POINTS,
        "DST_POINTS": config.DST_POINTS,
    }

    pipeline = IntersectionPipeline(CONFIG)
    pipeline.run()