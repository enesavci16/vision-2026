
import numpy as np

VIDEO_PATH = r"consolidation/q1_speed_pipeline/video/xx.mp4"
MODEL_PATH = r"yolov8n.pt"
CONFIDENCE_THR =0.5
BEV_WIDTH=640
BEV_HEIGHT=640
PIXEL_PER_METER=10
OUTPUT_DIR = r"consolidation/q1_speed_pipeline/outputs"
OUTPUT_JSON  = r"consolidation/q1_speed_pipeline/outputs/speed_report.json"
OUTPUT_VIDEO = r"consolidation/q1_speed_pipeline/outputs/annotated.mp4"

SRC_POINTS = np.array([
    [100,200],
    [540,200],
    [640,400],
    [0,400]
    
], dtype=np.float32)

DST_POINTS= np.array([
    [0,   0],    
    [640, 0],    
    [640, 200],   
    [0,   200]   
], dtype=np.float32)









