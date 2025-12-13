from brach ak

once again chenging from ak

{
    "action": "start_stream",
    "stream_name": "Cam424",
    "camera_id": 10,
    "user_id": 10,
    "org_id": 10,
    "threshold": 80,
    "alert_rate": 90,
    "region": "us-east-1"
}

https://andymerry.s3.us-east-1.amazonaws.com/1191560-hd_1920_1080_25fps.mp4




import cv2
import json
import time
import asyncio
import logging
import base64
from concurrent.futures import ThreadPoolExecutor
from src.models.theft_detection import theft_detection
from src.store_s3.theft_store import upload_to_s3
from src.database.theft_query import insert_theft_detection

logger = logging.getLogger("theft_detection")
logger.setLevel(logging.INFO)

JPEG_QUALITY = 80


def encode_frame_to_b64(frame_rgb, quality=JPEG_QUALITY):
    """Encode RGB frame to base64 JPEG."""
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode frame")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def run_theft_detection(
    client_id: str,
    video_url: str,
    camera_id: int,
    user_id: int,
    org_id: int,
    sessions: dict,
    loop: asyncio.AbstractEventLoop,
    storage_executor: ThreadPoolExecutor,
):
    """
    Runs Theft Detection in a separate thread.
    Sends WebSocket messages safely and stores frames to S3/DB in background threads to avoid blocking inference.
    """

    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        logger.error(f"[{client_id}] Unable to open video stream: {video_url}")
        return

    frame_num = 0

    while cap.isOpened() and sessions.get(client_id, {}).get("streaming", False):
         ret, frame_bgr = cap.read()
         if not ret:
                print("✅ End of video reached.")
                break

        # Resize if too wide
         h, w = frame_bgr.shape[:2]
         if w > 1280:
            scale = 1280 / w
            frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
            h, w = frame_bgr.shape[:2]

    

        # Convert to RGB for encoder (existing behavior)
         frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
         frame_b64 = encode_frame_to_b64(frame_rgb)
                

         try:
            # ---------------- Theft Detection Inference ----------------
            detections,error= theft_detection(frame_b64, camera_id, user_id, org_id)
            # print(detections)
            ws = sessions.get(client_id, {}).get("ws")

            if detections:

                # ---------------- WebSocket Send ----------------
                if ws:
                    try:
                        asyncio.run_coroutine_threadsafe(
                            ws.send_text(json.dumps(detections)), loop
                        )
                        logger.info(f"[{client_id}] Frame {frame_num}: Detections sent to client")
                    except Exception as e:
                        logger.error(f"[{client_id}] Frame {frame_num}: WebSocket send error -> {e}")
                        break

                #---------------- Background Storage ----------------
                # annotated_frame = detections.get("annotated_frame")
                # if annotated_frame is not None:
                #     def store_frame():
                #         try:
                #             s3_url = upload_to_s3(annotated_frame, frame_num)
                #             insert_theft_detection(detections, s3_url)
                #             logger.info(f"[{client_id}] Frame {frame_num} stored successfully")
                #         except Exception as e:
                #             logger.error(f"[{client_id}] Frame {frame_num} store error -> {e}")

                #     # Schedule S3/DB storage without blocking inference
                #     storage_executor.submit(store_frame)
            else:
                if ws:
                    asyncio.run_coroutine_threadsafe(
                        ws.send_text(json.dumps({"success": False, "message": error})), loop
                    )
                logger.warning(f"[{client_id}] Frame {frame_num}: No detections - {error}")
                break

         except Exception as e:
            logger.exception(f"[{client_id}] Frame {frame_num}: Pipeline error -> {e}")

         time.sleep(0.03)

    cap.release()
    if client_id in sessions:
        sessions[client_id]["streaming"] = False

    logger.info(f"[{client_id}] Theft detection stopped and resources released")