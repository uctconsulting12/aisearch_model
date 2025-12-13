import cv2
import json
import time
import asyncio
import logging
import sys
import os
import base64
from concurrent.futures import ThreadPoolExecutor
# from src.models.people_counting import people_counting

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.local_models.ai_search.inference import model_fn,input_fn,predict_fn,output_fn

# ---------- Parameters ----------
FRAME_SKIP = 1        # Process every 5th frame
BATCH_SIZE = 32      # Number of frames per batch

model_dir = "."
model = model_fn(model_dir)


logger = logging.getLogger("people_counting")
logger.setLevel(logging.INFO)


def run_ai_search(
    video_url: str,
    camera_id: int,
    process_id: int,
    org_id: int,
    search_text: str,
    
):
    """
    Runs People Counting detection in a separate thread.
    Sends WebSocket messages safely and stores frames to S3/DB in background threads to avoid blocking inference.
    """

    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        logger.error(f"[{camera_id}] Unable to open video stream: {video_url}")
        return

    
    frame_idx = 0
    total_time = 0
    batch_frames = []
    batch_ids = []

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame_idx += 1

        # Skip frames for efficiency
        if frame_idx % FRAME_SKIP != 0:
            continue

        # Encode frame to base64
        _, buffer = cv2.imencode(".jpg", frame)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        batch_frames.append(img_base64)
        batch_ids.append(frame_idx)

        # When batch full -> run inference
        if len(batch_frames) == BATCH_SIZE:
            start_time = time.time()

            input_data = {
                "orgid": org_id,
                "processid": process_id,
                "cam_id": camera_id,
                "search_text": search_text,
                "frames": batch_frames,
                "batch_size": BATCH_SIZE,
                "annotated_frame": True
            }

        try:
            
            # ---------------- People Counting Inference ----------------
            detections= predict_fn(input_data, model)

            print(detections)

            

                

            

        except Exception as e:
            logger.exception(f"[{camera_id}] Frame {frame_idx}: Pipeline error -> {e}")

    cap.release()
    
    logger.info(f"[{camera_id}] People counting stopped and resources released")


if __name__ == "__main__":
    run_ai_search(r"C:\Users\uct\Desktop\ai_search_backend\test_videos\istockphoto-1404365178-640_adpp_is.mp4",1,2,3,"person with white cap")