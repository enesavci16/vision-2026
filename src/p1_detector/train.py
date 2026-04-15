import sys
import yaml
from pathlib import Path

# --- SYS PATH FIX: Force Python to see 'src' ---
# This ensures the script works even if the package isn't pip installed yet.
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[2]  # Go up 2 levels to 'vision_2026'
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))
# -----------------------------------------------

# Now we can import our custom module
try:
    from p1_detector.models.yolo import TrafficDetector
except ImportError as e:
    print(f"❌ IMPORT ERROR: {e}")
    print(f"   Debug Info: looking for modules in {SRC_DIR}")
    sys.exit(1)

CONFIG_PATH = PROJECT_ROOT / "config" / "model_config.yaml"


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    print(f"📂 Loading Configuration from: {CONFIG_PATH}")

    try:
        # 1. Load Config
        full_config = load_config(CONFIG_PATH)
        # Ensure 'project_1' key exists
        if "project_1" not in full_config:
            raise KeyError("'project_1' key missing in yaml config")

        p1_config = full_config["project_1"]

        # 2. Initialize Adapter
        print(
            f"⚖️  Initializing TrafficDetector with {p1_config.get('base_model', 'unknown')}"
        )
        detector = TrafficDetector(model_path=p1_config["base_model"])

        # 3. Run Training
        detector.train(p1_config)

    except Exception as e:
        print(f"❌ FATAL ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
