import logging
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
from collections import defaultdict

# ------------------- Hata Veri Yapısı -------------------
@dataclass
class ValidationError:
    dataset: str
    file_path: str
    line_num: int
    error_type: str
    details: str
    raw_line: str

# ------------------- Doğrulayıcı Sınıfı -------------------
class YOLOLabelValidator:
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
    EXPECTED_CLASS_ID = 0  # Tasarım gereği tek sınıf
    EDGE_TOLERANCE = 0.002  # Yuvarlama/annotator payı — PJT2905 vakası (Oturum 22)

    def __init__(self, root_path: Path, datasets: Optional[List[str]] = None):
        self.root_path = Path(root_path)
        self.datasets = datasets or ["br", "eu", "us"]
        self.errors: List[ValidationError] = []
        self.stats: Dict[str, Dict] = {}

    def validate_all(self) -> None:
        """Tüm datasetleri gezerek doğrulama yapar."""
        for dataset_name in self.datasets:
            dataset_path = self.root_path / dataset_name
            if not dataset_path.is_dir():
                self._add_error(
                    dataset_name,
                    str(dataset_path),
                    0,
                    "DIRECTORY_NOT_FOUND",
                    f"Klasör bulunamadı: {dataset_path}",
                    "",
                )
                continue

            logging.info(f"Taranıyor: {dataset_path}")
            self.stats[dataset_name] = {
                "txt_count": 0,
                "image_count": 0,
                "line_count": 0,
                "error_count": 0,
            }

            image_files = self._find_images(dataset_path)
            self.stats[dataset_name]["image_count"] = len(image_files)

            txt_files = list(dataset_path.glob("*.txt"))
            self.stats[dataset_name]["txt_count"] = len(txt_files)

            for txt_path in txt_files:
                self._validate_file(dataset_name, txt_path)

            if self.stats[dataset_name]["txt_count"] != self.stats[dataset_name]["image_count"]:
                self._add_error(
                    dataset_name,
                    str(dataset_path),
                    0,
                    "COUNT_MISMATCH",
                    f"Txt sayısı ({self.stats[dataset_name]['txt_count']}) "
                    f"ile görüntü sayısı ({self.stats[dataset_name]['image_count']}) eşleşmiyor.",
                    "",
                )

    def _validate_file(self, dataset_name: str, txt_path: Path) -> None:
        """Bir .txt dosyasındaki tüm satırları doğrular."""
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            self._add_error(
                dataset_name,
                str(txt_path),
                0,
                "FILE_READ_ERROR",
                f"Dosya okunamadı: {e}",
                "",
            )
            return

        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            self.stats[dataset_name]["line_count"] += 1
            self._validate_line(dataset_name, txt_path, idx, stripped)

    def _validate_line(self, dataset_name: str, txt_path: Path, line_num: int, raw_line: str) -> None:
        """Tek bir YOLO satırını doğrular."""
        parts = raw_line.split()

        # --- 1. Format kontrolü (class_id cx cy w h) ---
        if len(parts) != 5:
            self._add_error(
                dataset_name,
                str(txt_path),
                line_num,
                "INVALID_LINE_FORMAT",
                f"Beklenen 5 bileşen, bulunan: {len(parts)}",
                raw_line,
            )
            return

        # --- 2. Class ID kontrolü (SADECE 0 KABUL EDİLİR) ---
        try:
            class_id = int(parts[0])
            if class_id != self.EXPECTED_CLASS_ID:
                self._add_error(
                    dataset_name,
                    str(txt_path),
                    line_num,
                    "INVALID_CLASS_ID",
                    f"Beklenen sınıf ID = {self.EXPECTED_CLASS_ID}, gelen = {class_id}",
                    raw_line,
                )
        except ValueError:
            self._add_error(
                dataset_name,
                str(txt_path),
                line_num,
                "INVALID_CLASS_ID",
                f"Geçersiz class_id formatı: '{parts[0]}' (integer olmalı)",
                raw_line,
            )

        # --- 3. cx, cy, w, h float ve [0,1] aralığı kontrolü ---
        try:
            cx, cy, w, h = map(float, parts[1:5])
        except ValueError:
            self._add_error(
                dataset_name,
                str(txt_path),
                line_num,
                "INVALID_COORDINATE_FORMAT",
                f"cx,cy,w,h sayısal olmalı: '{' '.join(parts[1:5])}'",
                raw_line,
            )
            return

        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
            self._add_error(
                dataset_name,
                str(txt_path),
                line_num,
                "OUT_OF_BOUNDS",
                f"cx={cx}, cy={cy}, w={w}, h={h} → [0,1] aralığında değil",
                raw_line,
            )

        # --- 4. Sıkı Kenar Kontrolü (tolerans dahil) ---
        if not (w / 2.0 - self.EDGE_TOLERANCE <= cx <= 1.0 - w / 2.0 + self.EDGE_TOLERANCE):
            self._add_error(
                dataset_name,
                str(txt_path),
                line_num,
                "EDGE_CONSTRAINT_X",
                f"cx={cx}, w={w} → tolerans dahil w/2={w/2:.4f} ≤ cx ≤ {1-w/2:.4f} "
                f"(±{self.EDGE_TOLERANCE}) koşulu sağlanmıyor",
                raw_line,
            )

        if not (h / 2.0 - self.EDGE_TOLERANCE <= cy <= 1.0 - h / 2.0 + self.EDGE_TOLERANCE):
            self._add_error(
                dataset_name,
                str(txt_path),
                line_num,
                "EDGE_CONSTRAINT_Y",
                f"cy={cy}, h={h} → tolerans dahil h/2={h/2:.4f} ≤ cy ≤ {1-h/2:.4f} "
                f"(±{self.EDGE_TOLERANCE}) koşulu sağlanmıyor",
                raw_line,
            )

    def _find_images(self, dataset_path: Path) -> List[Path]:
        """Klasördeki tüm görüntü dosyalarını döndürür (case-insensitive sistemlerde tekilleştirilmiş)."""
        images = set()
        for ext in self.IMAGE_EXTENSIONS:
            images.update(dataset_path.glob(f"*{ext}"))
            images.update(dataset_path.glob(f"*{ext.upper()}"))
        return list(images)

    def _add_error(self, dataset: str, file_path: str, line_num: int, err_type: str, details: str, raw_line: str) -> None:
        error = ValidationError(
            dataset=dataset,
            file_path=file_path,
            line_num=line_num,
            error_type=err_type,
            details=details,
            raw_line=raw_line,
        )
        self.errors.append(error)
        if dataset in self.stats:
            self.stats[dataset]["error_count"] += 1

    def print_summary(self) -> None:
        """Tüm hataları ve özet istatistikleri loglar."""
        if not self.errors:
            logging.info("🎉 Harika! Hiçbir hata bulunamadı. Tüm etiketler geçerli.")
            return

        logging.warning(f"⚠️ Toplam {len(self.errors)} hata tespit edildi.")

        grouped: Dict[str, List[ValidationError]] = defaultdict(list)
        for err in self.errors:
            grouped[err.error_type].append(err)

        for err_type, err_list in grouped.items():
            logging.warning(f"  └─ [{err_type}] → {len(err_list)} adet")
            for err in err_list[:5]:
                logging.warning(
                    f"       Dosya: {err.file_path} (Satır: {err.line_num}) -> {err.details}"
                )
            if len(err_list) > 5:
                logging.warning(f"       ... ve {len(err_list) - 5} hata daha.")

        logging.info("📊 Dataset Özeti:")
        for ds, stat in self.stats.items():
            logging.info(
                f"  {ds}: Txt={stat['txt_count']}, Görüntü={stat['image_count']}, "
                f"Satır={stat['line_count']}, Hata={stat['error_count']}"
            )

# ------------------- Fonksiyonel API (Colab'da import edip kullanmak için) -------------------
def run_validation(root_path: str, datasets: Optional[List[str]] = None) -> YOLOLabelValidator:
    """
    Colab notebook'undan doğrudan çağırmak için fonksiyon.

    Örnek:
        validator = run_validation("/content/drive/MyDrive/endtoend")
        validator.print_summary()
    """
    validator = YOLOLabelValidator(Path(root_path), datasets)
    validator.validate_all()
    return validator

# ------------------- Komut Satırı Girişi -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO etiket dosyalarını doğrula.")
    parser.add_argument(
        "--root-path",
        type=str,
        required=True,
        help="Ana dizin yolu (br/eu/us alt klasörlerini içermeli)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    validator = run_validation(args.root_path)
    validator.print_summary()