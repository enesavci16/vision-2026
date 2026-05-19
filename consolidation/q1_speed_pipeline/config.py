
import numpy as np

VIDEO_PATH = r"consolidation/q1_speed_pipeline/video/traffic.mp4"
MODEL_PATH = r"yolov8n.pt"
CONFIDENCE_THR =0.5
BEV_WIDTH=640
BEV_HEIGHT=640
PIXEL_PER_METER=10
OUTPUT_DIR = r"consolidation/q1_speed_pipeline/outputs"
OUTPUT_JSON  = r"consolidation/q1_speed_pipeline/outputs/speed_report.json"
OUTPUT_VIDEO = r"consolidation/q1_speed_pipeline/outputs/annotated.mp4"


SRC_POINTS = np.float32([[216, 71], [412, 73], [559, 202], [133, 209]])

# BEV canvas: 400x300 piksel = 20m x 15m
DST_POINTS = np.float32([
    [0,   0],
    [400, 0],
    [400, 300],
    [0,   300],
])











