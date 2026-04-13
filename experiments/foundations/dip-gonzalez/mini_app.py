import cv2
import logging
import time
from typing import Dict, Tuple
import numpy as np
from functools import wraps
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


IMAGE_PATH = r"C:/Users/enesa/projeler/lead-traffic-projects/project-01/vision_2026/datasets/Bursa_Traffic_v1/train/images/taksi_08_jpg.rf.a36aebeb77df4983a5d10ee927187399.jpg"

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def timing_decorator(func):
    """Measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} tamamlandı: {elapsed:.3f}s")
        return result
    return wrapper


class PlateSharpener:
    """Foggy license plate recovery system using Laplacian sharpening."""

    def __init__(self, image_path: str, bbox: Tuple[int, int, int, int]):
        self.image_path = image_path
        self.bbox = bbox
        self.plate_crop = None
        self.results: Dict = {
            "preprocess": {},
            "sharpen": {}
        }
        logger.info(f"PlateSharpener oluşturuldu: {image_path}, bbox={bbox}")

    def load_and_crop(self) -> None:
        """Load image and crop the license plate region."""
        img = cv2.imread(self.image_path)
        if img is None:
            raise FileNotFoundError(f"Görüntü yüklenemedi: {self.image_path}")
        x, y, w, h = self.bbox
        self.plate_crop = img[y:y+h, x:x+w]
        logger.info(f"Plaka crop edildi: {w}x{h} piksel")

    @staticmethod
    def compute_gradient(image: np.ndarray) -> float:
        """Compute mean gradient magnitude as sharpness metric."""
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return float(np.mean(magnitude))

    @classmethod
    def create_from_config(cls, config_dict: Dict) -> "PlateSharpener":
        """Construct PlateSharpener from a config dictionary."""
        return cls(
            image_path=config_dict["image_path"],
            bbox=tuple(config_dict["bbox"])
        )

    def preprocess(self, method: str) -> np.ndarray:
        """Apply preprocessing to the plate crop."""
        if self.plate_crop is None:
            raise ValueError("Önce load_and_crop() çağırmalısın!")

        gray = cv2.cvtColor(self.plate_crop, cv2.COLOR_BGR2GRAY)

        if method == "raw":
            result = gray
        elif method == "histeq":
            result = cv2.equalizeHist(gray)
        elif method == "maf":
            histeq = cv2.equalizeHist(gray)
            result = cv2.GaussianBlur(histeq, (5, 5), 0)
        else:
            raise ValueError(f"Geçersiz method: {method}")

        self.results["preprocess"][method] = result
        logger.info(f"Ön işleme tamamlandı: {method}")
        return result

    @timing_decorator
    def sharpen(self, method: str, alpha: float) -> np.ndarray:
        """Apply Laplacian sharpening to a preprocessed image."""
        if method not in self.results["preprocess"]:
            raise ValueError(f"{method} ön işlemesi yapılmamış!")

        preprocessed = self.results["preprocess"][method]
        laplacian = cv2.Laplacian(preprocessed, cv2.CV_64F)
        sharpened = preprocessed + alpha * laplacian
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        self.results["sharpen"][(method, alpha)] = sharpened
        logger.info(f"Keskinleştirme: {method} + α={alpha}")
        return sharpened

    def visualize_matrix(self, alphas: list) -> None:
        """Produce 3x3 grid (preprocess x alpha) and gradient bar chart."""
        METHODS = ["raw", "histeq", "maf"]
        METHOD_LABELS = {"raw": "RAW", "histeq": "HISTEQ", "maf": "MAF"}

        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.4)
        gs_top = gridspec.GridSpecFromSubplotSpec(
            3, 3, subplot_spec=gs[0], hspace=0.05, wspace=0.05
        )

        for row, method in enumerate(METHODS):
            for col, alpha in enumerate(alphas):
                ax = fig.add_subplot(gs_top[row, col])
                img = self.results["sharpen"].get((method, alpha))
                if img is not None:
                    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                ax.axis("off")
                if row == 0:
                    ax.set_title(f"α={alpha}", fontsize=9)
                if col == 0:
                    ax.set_ylabel(
                        METHOD_LABELS[method],
                        fontsize=9,
                        rotation=0,
                        labelpad=35,
                        va="center",
                    )

        ax_bar = fig.add_subplot(gs[1])
        bar_labels = []
        bar_values = []

        for method in METHODS:
            for alpha in alphas:
                img = self.results["sharpen"].get((method, alpha))
                if img is not None:
                    bar_labels.append(f"{METHOD_LABELS[method]}\nα={alpha}")
                    bar_values.append(self.compute_gradient(img))

        x_pos = range(len(bar_labels))
        ax_bar.bar(x_pos, bar_values, color="steelblue", edgecolor="white")
        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels(bar_labels, fontsize=7)
        ax_bar.set_ylabel("Ort. Gradyan")
        ax_bar.set_title("Keskinlik Karşılaştırması (Gradyan Büyüklüğü)")

        plt.suptitle(
            "Sisli Plaka Kurtarma — 3×3 Deneme Matrisi",
            fontsize=12,
            fontweight="bold",
        )
        plt.savefig("plaka_matrix.png", dpi=150, bbox_inches="tight")
        logger.info("Görsel kaydedildi: plaka_matrix.png")
        plt.show()


def main() -> None:
    """Pipeline: 3 preprocessing methods x 3 alpha values = 9 combinations."""
    BBOX = (120, 310, 200, 60)
    ALPHAS = [0.5, 1.0, 1.5]
    METHODS = ["raw", "histeq", "maf"]

    sharpener = PlateSharpener(image_path=IMAGE_PATH, bbox=BBOX)
    sharpener.load_and_crop()

    for method in METHODS:
        sharpener.preprocess(method)

    for method in METHODS:
        for alpha in ALPHAS:
            sharpener.sharpen(method, alpha)

    sharpener.visualize_matrix(alphas=ALPHAS)
    logger.info("Pipeline tamamlandı.")


if __name__ == "__main__":
    main()