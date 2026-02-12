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
# Note: BATCH_SIZE and FRAME_SKIP are now calculated dynamically based on video FPS

model_dir = "."
model = model_fn(model_dir)

logger = logging.getLogger("ai_search")
logger.setLevel(logging.INFO)


def calculate_video_timestamp(frame_number: int, fps: float) -> str:
    """
    Calculate video timestamp from frame number and FPS.

    Args:
        frame_number: The frame number in the video
        fps: Frames per second of the video

    Returns:
        Timestamp in format "MM:SS" or "HH:MM:SS" for videos longer than 1 hour

    Examples:
        - Frame 150 at 30 FPS -> "00:05" (5 seconds)
        - Frame 9000 at 30 FPS -> "05:00" (5 minutes)
        - Frame 162000 at 30 FPS -> "1:30:00" (1 hour 30 minutes)
    """
    try:
        # Calculate total seconds
        total_seconds = frame_number / fps

        # Calculate hours, minutes, seconds
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)

        # Format based on video length
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
    except Exception as e:
        logger.warning(f"Failed to calculate timestamp for frame {frame_number}: {e}")
        return "00:00"


def calculate_smart_batch_size(fps: float) -> tuple:
    """
    Calculate optimal batch size and frame skip based on FPS.
    Goal: Process approximately 1 batch per second for efficiency.

    Logic:
    - Try divisors from 5 down to 2
    - Find the largest divisor that divides FPS evenly
    - batch_size = fps / divisor
    - This ensures we process ~1 second worth of frames per batch

    Examples:
        fps=25 -> 25/5=5 batch_size, process 5 frames per second
        fps=30 -> 30/5=6 batch_size, process 6 frames per second
        fps=32 -> 32/4=8 batch_size, process 8 frames per second
        fps=24 -> 24/4=6 batch_size, process 6 frames per second
        fps=60 -> 60/5=12 batch_size, process 12 frames per second

    Returns:
        (batch_size, frame_skip): Optimal values for processing
    """
    try:
        fps = int(round(fps))

        # Try divisors from 5 down to 2
        for divisor in [5, 4, 3, 2]:
            if fps % divisor == 0:
                batch_size = fps // divisor
                frame_skip = divisor
                logger.info(
                    f"[Smart Batching] FPS={fps}, Divisor={divisor} -> Batch Size={batch_size}, Frame Skip={frame_skip}")
                return batch_size, frame_skip

        # If no divisor works, use default approach
        # Process approximately 1 batch per second
        if fps >= 30:
            batch_size = max(6, fps // 5)
            frame_skip = 5
        elif fps >= 20:
            batch_size = max(4, fps // 5)
            frame_skip = 5
        else:
            batch_size = max(3, fps // 4)
            frame_skip = 4

        logger.info(
            f"[Smart Batching] FPS={fps} (no clean divisor) -> Batch Size={batch_size}, Frame Skip={frame_skip}")
        return batch_size, frame_skip

    except Exception as e:
        logger.warning(f"Failed to calculate smart batch size: {e}. Using defaults.")
        return 6, 5  # Default fallback


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

    # ✅ Extract FPS from video
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:  # Fallback for streams without FPS metadata
        fps = 30.0  # Default assumption
        logger.warning(f"[{client_id}] Unable to detect video FPS, using default: {fps}")
    else:
        logger.info(f"[{client_id}] Video FPS detected: {fps}")

    # ✅ Calculate smart batch size and frame skip based on FPS
    BATCH_SIZE, FRAME_SKIP = calculate_smart_batch_size(fps)
    logger.info(f"[{client_id}] Using BATCH_SIZE={BATCH_SIZE}, FRAME_SKIP={FRAME_SKIP}")

    frame_idx = 0
    batch_frames = []
    batch_frame_numbers = []  # Track actual frame numbers for timestamp calculation

    # ✅ Accumulate ALL detections throughout the video
    all_detections = {}  # frame_id -> base64_image
    all_timestamps = []  # List of timestamps
    all_confidences = []  # List of confidence scores
    all_frame_numbers = []  # List of actual frame numbers for sorting
    last_inference_metadata = {}  # Store inference metadata

    logger.info(f"[{client_id}] Starting AI search for video: {video_url}")

    while cap.isOpened() and sessions.get(client_id, {}).get("streaming", False):
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        frame_idx += 1

        # Skip frames based on calculated frame skip
        if frame_idx % FRAME_SKIP != 0:
            continue

        # Encode frame
        _, buffer = cv2.imencode(".jpg", frame)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        batch_frames.append(img_base64)
        batch_frame_numbers.append(frame_idx)  # Store actual frame number

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

                # Run inference
                detections = predict_fn(input_data, model)

                if detections and detections.get("detected_frames"):
                    # ✅ Calculate timestamps for detected frames
                    detected_frame_ids = list(detections.get("detected_frames", {}).keys())
                    confidences = detections.get("confidence", [])
                    detected_frames_dict = detections.get("detected_frames", {})

                    # Create timestamp list based on batch frame numbers
                    timestamps = []
                    for i, frame_num in enumerate(batch_frame_numbers):
                        if i < len(detected_frame_ids):  # If this frame was detected
                            timestamp = calculate_video_timestamp(frame_num, fps)
                            timestamps.append(timestamp)

                    # ✅ ACCUMULATE detections instead of overwriting
                    all_detections.update(detected_frames_dict)
                    all_timestamps.extend(timestamps)
                    all_confidences.extend(confidences)
                    all_frame_numbers.extend(batch_frame_numbers[:len(detected_frame_ids)])

                    # ✅ Capture all metadata from inference response
                    last_inference_metadata = {
                        "orgid": detections.get("orgid", org_id),
                        "processid": detections.get("processid", process_id),
                        "cam_id": detections.get("cam_id", camera_id),
                        "search_text": detections.get("search_text", search_text),
                        "total_frames_processed": detections.get("total_frames_processed", len(batch_frames)),
                        "total_frames_found": detections.get("total_frames_found", len(detected_frame_ids)),
                        "detection_rate": detections.get("detection_rate", 0.0),
                        "processing_time": detections.get("processing_time", "0ms"),
                        "total_inference_time": detections.get("total_inference_time", "0ms"),
                        "average_inference_per_frame": detections.get("average_inference_per_frame", "0ms"),
                        "device": detections.get("device", "unknown"),
                        "box_threshold_used": detections.get("box_threshold_used", 0.6),
                        "context_threshold_used": detections.get("context_threshold_used", 0.6)
                    }

                    logger.info(
                        f"[{client_id}] Batch processed, {len(detected_frame_ids)} detections found. Total so far: {len(all_detections)}")
                else:
                    logger.info(f"[{client_id}] Batch processed, no detections.")

                # ✅ Update metadata even if no detections (for final "no results" response)
                if not detections.get("detected_frames"):
                    last_inference_metadata = {
                        "orgid": detections.get("orgid", org_id),
                        "processid": detections.get("processid", process_id),
                        "cam_id": detections.get("cam_id", camera_id),
                        "search_text": detections.get("search_text", search_text),
                        "total_frames_processed": detections.get("total_frames_processed", len(batch_frames)),
                        "processing_time": detections.get("processing_time", "0ms"),
                        "total_inference_time": detections.get("total_inference_time", "0ms"),
                        "average_inference_per_frame": detections.get("average_inference_per_frame", "0ms"),
                        "device": detections.get("device", "unknown"),
                        "box_threshold_used": detections.get("box_threshold_used", 0.6),
                        "context_threshold_used": detections.get("context_threshold_used", 0.6)
                    }

            except Exception as e:
                logger.exception(f"[{client_id}] Batch error -> {e}")

            # Reset batch
            batch_frames = []
            batch_frame_numbers = []

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

            if detections and detections.get("detected_frames"):
                # ✅ Calculate timestamps for final batch
                detected_frame_ids = list(detections.get("detected_frames", {}).keys())
                confidences = detections.get("confidence", [])
                detected_frames_dict = detections.get("detected_frames", {})

                timestamps = []
                for i, frame_num in enumerate(batch_frame_numbers):
                    if i < len(detected_frame_ids):
                        timestamp = calculate_video_timestamp(frame_num, fps)
                        timestamps.append(timestamp)

                # ✅ ACCUMULATE final batch detections
                all_detections.update(detected_frames_dict)
                all_timestamps.extend(timestamps)
                all_confidences.extend(confidences)
                all_frame_numbers.extend(batch_frame_numbers[:len(detected_frame_ids)])

                # ✅ Capture all metadata from inference response
                last_inference_metadata = {
                    "orgid": detections.get("orgid", org_id),
                    "processid": detections.get("processid", process_id),
                    "cam_id": detections.get("cam_id", camera_id),
                    "search_text": detections.get("search_text", search_text),
                    "total_frames_processed": detections.get("total_frames_processed", len(batch_frames)),
                    "total_frames_found": detections.get("total_frames_found", len(detected_frame_ids)),
                    "detection_rate": detections.get("detection_rate", 0.0),
                    "processing_time": detections.get("processing_time", "0ms"),
                    "total_inference_time": detections.get("total_inference_time", "0ms"),
                    "average_inference_per_frame": detections.get("average_inference_per_frame", "0ms"),
                    "device": detections.get("device", "unknown"),
                    "box_threshold_used": detections.get("box_threshold_used", 0.6),
                    "context_threshold_used": detections.get("context_threshold_used", 0.6)
                }

                logger.info(
                    f"[{client_id}] Final batch processed, {len(detected_frame_ids)} detections found. Total: {len(all_detections)}")
            else:
                # ✅ Update metadata even if no detections in final batch
                last_inference_metadata = {
                    "orgid": detections.get("orgid", org_id),
                    "processid": detections.get("processid", process_id),
                    "cam_id": detections.get("cam_id", camera_id),
                    "search_text": detections.get("search_text", search_text),
                    "total_frames_processed": detections.get("total_frames_processed", len(batch_frames)),
                    "processing_time": detections.get("processing_time", "0ms"),
                    "total_inference_time": detections.get("total_inference_time", "0ms"),
                    "average_inference_per_frame": detections.get("average_inference_per_frame", "0ms"),
                    "device": detections.get("device", "unknown"),
                    "box_threshold_used": detections.get("box_threshold_used", 0.6),
                    "context_threshold_used": detections.get("context_threshold_used", 0.6)
                }

        except Exception as e:
            logger.exception(f"[{client_id}] Final batch error -> {e}")

    # ---- Send TOP 5 results sorted by timestamp to client OR send "No results found" ----
    ws = sessions.get(client_id, {}).get("ws")
    if ws:
        try:
            if all_detections:
                # ✅ CASE 1: Detections found - Sort and send top 5
                # Create list of tuples: (frame_number, frame_id, timestamp, confidence)
                detection_list = []
                frame_ids = list(all_detections.keys())

                for i, frame_id in enumerate(frame_ids):
                    detection_list.append({
                        "frame_number": all_frame_numbers[i] if i < len(all_frame_numbers) else i,
                        "frame_id": frame_id,
                        "image": all_detections[frame_id],
                        "timestamp": all_timestamps[i] if i < len(all_timestamps) else "00:00",
                        "confidence": all_confidences[i] if i < len(all_confidences) else 0.0
                    })

                # Sort by frame number (earliest timestamp first)
                detection_list.sort(key=lambda x: x["frame_number"])

                # ✅ Take top 5
                top_5 = detection_list[:5]

                # Build final response with top 5
                top_5_detections = {item["frame_id"]: item["image"] for item in top_5}
                top_5_timestamps = [item["timestamp"] for item in top_5]
                top_5_confidences = [item["confidence"] for item in top_5]

                response = {
                    # Status
                    "status": "success",

                    # Detection results (TOP 5)
                    "detected_frames": top_5_detections,
                    "confidence": top_5_confidences,
                    "timestamps": top_5_timestamps,

                    # Video-specific metadata (added by ai_search.py)
                    "video_fps": fps,
                    "batch_size_used": BATCH_SIZE,
                    "frame_skip_used": FRAME_SKIP,
                    "total_detections_found": len(all_detections),  # Total found
                    "detections_returned": len(top_5)  # Top 5 returned
                }

                # ✅ Merge all inference metadata
                response.update(last_inference_metadata)

                asyncio.run_coroutine_threadsafe(
                    ws.send_text(json.dumps(response)),
                    loop
                )
                logger.info(
                    f"[{client_id}] ✅ Sent top {len(top_5)} detections (out of {len(all_detections)} total) sorted by timestamp.")
                logger.info(f"[{client_id}] Top 5 Timestamps: {top_5_timestamps}")

            else:
                # ✅ CASE 2: No detections found - Send "No results found" notification
                response = {
                    # Status
                    "status": "No results found",

                    # Basic metadata
                    "orgid": last_inference_metadata.get("orgid", org_id),
                    "processid": last_inference_metadata.get("processid", process_id),
                    "cam_id": last_inference_metadata.get("cam_id", camera_id),
                    "search_text": last_inference_metadata.get("search_text", search_text),

                    # Detection results (marked as N/A)
                    "detected_frames": "N/A",
                    "confidence": "N/A",
                    "timestamps": "N/A",
                    "total_detections_found": 0,
                    "detections_returned": 0,

                    # Video-specific metadata
                    "video_fps": fps,
                    "batch_size_used": BATCH_SIZE,
                    "frame_skip_used": FRAME_SKIP,

                    # Processing stats (from last batch if available, otherwise defaults)
                    "total_frames_processed": last_inference_metadata.get("total_frames_processed", "N/A"),
                    "processing_time": last_inference_metadata.get("processing_time", "N/A"),
                    "total_inference_time": last_inference_metadata.get("total_inference_time", "N/A"),
                    "average_inference_per_frame": last_inference_metadata.get("average_inference_per_frame", "N/A"),
                    "device": last_inference_metadata.get("device", "unknown"),
                    "box_threshold_used": last_inference_metadata.get("box_threshold_used", 0.6),
                    "context_threshold_used": last_inference_metadata.get("context_threshold_used", 0.6),
                    "detection_rate": 0.0
                }

                asyncio.run_coroutine_threadsafe(
                    ws.send_text(json.dumps(response)),
                    loop
                )
                logger.info(
                    f"[{client_id}] ℹ️ No detections found in entire video. Sent 'No results found' notification.")

        except Exception as e:
            logger.error(f"[{client_id}] ❌ Error sending response -> {e}")
    else:
        logger.warning(f"[{client_id}] ⚠️ No active WebSocket connection found.")

    cap.release()
    if client_id in sessions:
        sessions[client_id]["streaming"] = False

    logger.info(f"[{client_id}] AI search completed and resources released.")