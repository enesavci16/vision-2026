import os
import sys
from pathlib import Path
from roboflow import Roboflow
import yaml

# --- PATH SETUP ---
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[3]  # Up 3 levels to vision_2026
DATASET_DIR = PROJECT_ROOT / "datasets"


def download_dataset(api_key: str, workspace: str, project: str, version: int):
    """
    Programmatically downloads dataset from Roboflow.
    """
    print(f"⬇️  Initializing Download to {DATASET_DIR}...")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    dataset = project.version(version).download("yolov8", location=str(DATASET_DIR))

    # Return the path to the data.yaml file that YOLO needs
    return dataset.location


if __name__ == "__main__":
    # ---------------------------------------------------------
    # ⚠️ SECURITY WARNING:
    # In a real production env, use os.getenv("ROBOFLOW_KEY")
    # For now, paste your details below.
    # ---------------------------------------------------------

    # 1. GO TO ROBOFLOW -> VERSIONS -> "GET SNIPPET" -> "PYTHON"
    # 2. COPY YOUR VALUES HERE:

    API_KEY = "YOUR_API_KEY_HERE"  # <--- REPLACE THIS
    WORKSPACE = "YOUR_WORKSPACE"  # <--- REPLACE THIS
    PROJECT_NAME = "YOUR_PROJECT"  # <--- REPLACE THIS
    VERSION_NUM = 1  # <--- REPLACE THIS

    try:
        data_path = download_dataset(API_KEY, WORKSPACE, PROJECT_NAME, VERSION_NUM)
        print(f"✅ Data Ingested Successfully!")
        print(f"📍 Location: {data_path}")
        print(f"📝 Update your config/model_config.yaml with this path!")

    except Exception as e:
        print(f"❌ Download Failed: {e}")
