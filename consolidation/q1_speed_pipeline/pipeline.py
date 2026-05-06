# DÖNGÜ ÖNCESİ
# 1. Config'den video yolunu al, VideoCapture aç
# 2. YOLO modeli yükle
# 3. PerspectiveTransformer nesnesi oluştur
# 4. SpeedMeasurer nesnesi oluştur
# 5. VideoWriter hazırla

# DÖNGÜ:
# 1. cap.read() ile kareyi oku, video bittiyse döngüden çık
# 2. YOLO'ya kareyi ver, bounding box + confidence al
# 3. ByteTrack'e YOLO çıktısını ver, track_id + bbox al
# 4. Her track için bbox merkezini PerspectiveTransformer ile BEV koordinatına çevir
# 5. BEV koordinatını SpeedMeasurer'a ver, hız hesapla
# 6. Overlay çiz (bbox + track_id + km/h), VideoWriter'a yaz

# DÖNGÜ SONRASI:
# 1. VideoCapture ve VideoWriter'ı kapat
# 2. SpeedMeasurer'dan raporu al, JSON olarak diske yaz



import logging
import json
from typing import Dict, Any, Optional
import numpy as np

import cv2


from yolo import TrafficDetector
from speed_measurer import SpeedMeasurer
from perspective_transformer import PerspectiveTransformer


# Logger modül seviyesinde olur (doğru kullanım)
logger = logging.getLogger(__name__)


class IntersectionPipeline:
    """
    Pipeline for vehicle speed measurement at intersections.

    Loads video, runs YOLO detection,
    BEV transformation, and speed estimation end-to-end.

    Args:
        config: Dictionary containing pipeline configuration.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        logger.info("⚙️ Pipeline başlatılıyor...")

        self.config = config

        # Config values
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

        # OpenCV objects
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_writer: Optional[cv2.VideoWriter] = None

        # Pipeline components (DÖNGÜ ÖNCESİ)
        self.detector = TrafficDetector(model_path=self.model_path)
        self.perspective_transformer = PerspectiveTransformer(self.src_points, self.dst_points)
        self.speed_measurer = SpeedMeasurer(pixel_per_meter=self.pixel_per_meter)

        logger.info("✅ Pipeline bileşenleri başarıyla yüklendi.")

    def run(self) -> None:
        """Execute the full detection-tracking-speed pipeline."""
        logger.info("🎥 Video açılıyor...")

        # 1) VideoCapture aç
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            raise RuntimeError(f"Video açılamadı: {self.video_path}")

        # 2) Video bilgilerini al
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"📌 Video info: fps={fps}, width={width}, height={height}")

        # 3) VideoWriter hazırla
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_path = self.output_video

        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not self.video_writer.isOpened():
            raise RuntimeError(f"VideoWriter açılamadı: {output_path}")

        logger.info("✅ VideoWriter hazırlandı.")

        # 4) DÖNGÜ
        while True:
            ret, frame = self.cap.read()

            if not ret:
                logger.info("📌 Video bitti, döngü durduruluyor...")
                break

            # timestamp al (ms -> sec)
            timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # YOLO detections
            detections = self.detector.detect(frame)

            for i, det in enumerate(detections):
                bbox = det["bbox"]  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = bbox

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # BEV dönüşümü
                bev_coord = self.perspective_transformer.to_bev([cx, cy])

                # Şimdilik track_id olarak index kullanalım (ByteTrack yok)
                track_id = i

                # SpeedMeasurer update
                self.speed_measurer.update(track_id, bev_coord, timestamp)

                # Speed al (ilk frame'de hesaplanmayabilir)
                try:
                    speed_kmh = self.speed_measurer.get_speed(track_id)
                except KeyError:
                    speed_kmh = 0.0

                # Overlay çiz
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"ID:{track_id} {speed_kmh:.1f} km/h",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            self.video_writer.write(frame)

        # 5) DÖNGÜ SONRASI - Kaynakları kapat
        self.cap.release()
        self.video_writer.release()

        logger.info(f"✅ Video kaydedildi: {output_path}")

        # 6) Speed report al ve JSON kaydet
        report = self.speed_measurer.get_report()

        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        logger.info(f"✅ Speed report kaydedildi: {self.output_json}")


if __name__ == "__main__":
    # basicConfig sadece entry-point'te yapılır (doğru kullanım)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
        "DST_POINTS": config.DST_POINTS
    }

    pipeline = IntersectionPipeline(CONFIG)
    pipeline.run()