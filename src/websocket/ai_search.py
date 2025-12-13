import cv2
import json
import time
import asyncio
import logging
import sys
import os
import base64
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.local_models.ai_search.inference import model_fn, predict_fn

# ---------- Parameters ----------
FRAME_SKIP = 10
BATCH_SIZE = 32

model_dir = "."
model = model_fn(model_dir)

logger = logging.getLogger("ai_search")
logger.setLevel(logging.INFO)


def run_ai_search(
    client_id: str,
    video_url: str,
    camera_id: int,
    process_id: int,
    org_id: int,
    search_text: str,
    sessions: dict,
    loop: asyncio.AbstractEventLoop,
    storage_executor: ThreadPoolExecutor
):
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        logger.error(f"[{client_id}] Unable to open video stream: {video_url}")
        return

    frame_idx = 0
    batch_frames = []
    last_detections = None

    logger.info(f"[{client_id}] Starting AI search for video: {video_url}")

    while cap.isOpened() and sessions.get(client_id, {}).get("streaming", False):
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        frame_idx += 1

        # Skip frames
        if frame_idx % FRAME_SKIP != 0:
            continue

        # Encode frame
        _, buffer = cv2.imencode(".jpg", frame)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        batch_frames.append(img_base64)

        # Process when batch full
        if len(batch_frames) == BATCH_SIZE:
            try:
                input_data = {
                    "orgid": org_id,
                    "processid": process_id,
                    "cam_id": camera_id,
                    "search_text": search_text,
                    "frames": batch_frames,
                    "batch_size": len(batch_frames),
                    "annotated_frame": True,
                }
                detections = predict_fn(input_data, model)
                if detections:
                    last_detections = detections
                    logger.info(f"[{client_id}] Batch processed, detections found.")
                else:
                    logger.info(f"[{client_id}] Batch processed, no detections.")
            except Exception as e:
                logger.exception(f"[{client_id}] Batch error -> {e}")

            batch_frames = []  # Reset batch

    # ---- Process remaining frames if any ----
    if batch_frames:
        try:
            input_data = {
                "orgid": org_id,
                "processid": process_id,
                "cam_id": camera_id,
                "search_text": search_text,
                "frames": batch_frames,
                "batch_size": len(batch_frames),
                "annotated_frame": True,
            }
            detections = predict_fn(input_data, model)
            if detections:
                last_detections = detections
                logger.info(f"[{client_id}] Final (partial) batch processed, detections found.")
        except Exception as e:
            logger.exception(f"[{client_id}] Final batch error -> {e}")

    # ---- Send ONLY last detections ----
    ws = sessions.get(client_id, {}).get("ws")
    if last_detections and ws:
        try:
            asyncio.run_coroutine_threadsafe(
                ws.send_text(json.dumps({"detections": last_detections})),
                loop
            )
            logger.info(f"[{client_id}] ✅ Sent last detections to client.")
        except Exception as e:
            logger.error(f"[{client_id}] ❌ Error sending last detections -> {e}")
    else:
        logger.warning(f"[{client_id}] ⚠️ No active WebSocket or no detections to send.")

    cap.release()
    if client_id in sessions:
        sessions[client_id]["streaming"] = False

    logger.info(f"[{client_id}] AI search completed and resources released.")
