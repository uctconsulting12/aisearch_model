# inference.py
"""
SageMaker OWL-ViT v2 - Enhanced Visual Search with Context Understanding & Batch Processing
===============================================================================
OWL-ViT v2 FOR CUDA 12.x COMPATIBILITY

Key Features:
- Batch processing support (dynamic batch_size)
- Returns ONLY the frame with maximum confidence
- Contextual phrase understanding
- Strict semantic matching
- Enhanced prompt engineering for visual search
- Timestamp-based Frame IDs (YYYYMMDDHHMMSS)
- CUDA 12.x compatible
- Identical JSON input/output format as Grounding DINO version
"""

import os
import io
import sys
import time
import json
import gc
import base64
import logging
import traceback
import datetime
import math
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import boto3
import botocore
import requests

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_ID = os.getenv("MODEL_ID", "google/owlv2-base-patch16-ensemble")
BOX_THRESHOLD = float(os.getenv("BOX_THRESHOLD", "0.05"))  # OWL-ViT uses lower threshold
NMS_THRESHOLD = float(os.getenv("NMS_THRESHOLD", "0.3"))
CONTEXT_MATCH_THRESHOLD = float(os.getenv("CONTEXT_MATCH_THRESHOLD", "0.4"))

S3_ERROR_LOG_BUCKET = os.getenv("S3_ERROR_LOG_BUCKET")
SAGEMAKER_ENDPOINT_NAME_ENV = os.getenv("SAGEMAKER_ENDPOINT_NAME")
CW_FETCH_ON_LOAD = os.getenv("CW_FETCH_ON_LOAD", "false").lower() in ("1", "true", "yes")

DISABLE_FP16 = os.getenv("DISABLE_FP16", "true").lower() in ("1", "true", "yes")
DISABLE_COMPILE = os.getenv("DISABLE_COMPILE", "true").lower() in ("1", "true", "yes")

ENABLE_SUPERIMPOSITION = os.getenv("ENABLE_SUPERIMPOSITION", "true").lower() in ("1", "true", "yes")
ENABLE_CONTEXT_VALIDATION = os.getenv("ENABLE_CONTEXT_VALIDATION", "true").lower() in ("1", "true", "yes")

# =============================================================================
# LOGGING SETUP
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

inference_handler = None


# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass
class ContextualPhrase:
    """Represents a contextual search phrase with its components"""
    full_phrase: str
    main_object: str
    context_words: List[str]
    action_words: List[str]
    spatial_relations: List[str]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def generate_frame_id() -> str:
    """
    Generate frame ID in format: YYYYMMDDHHMMSS + milliseconds
    Example: 20251219143525123
    """
    try:
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        milliseconds = now.strftime("%f")[:3]
        frame_id = f"{timestamp}{milliseconds}"
        return frame_id
    except Exception as e:
        logger.warning(f"Failed to generate frame_id: {e}")
        return datetime.datetime.now().strftime("%Y%m%d%H%M%S")


def log_error(exc: Exception, context: Optional[Dict[str, Any]] = None, upload_to_s3: bool = True) -> None:
    """Log errors locally and optionally upload to S3"""
    try:
        ctx = context or {}
        tb = traceback.format_exc()
        msg = f"Exception: {str(exc)}\nContext: {json.dumps(ctx, default=str)}\nTraceback:\n{tb}\n\n"
        logger.error(msg)
        
        log_path = "/tmp/inference_errors.log"
        with open(log_path, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}")
        
        bucket = S3_ERROR_LOG_BUCKET
        if upload_to_s3 and bucket:
            try:
                s3 = boto3.client("s3")
                key = f"inference_errors/{time.strftime('%Y/%m/%d')}/inference_errors.log"
                s3.upload_file(log_path, bucket, key)
                logger.info(f"Uploaded error log to s3://{bucket}/{key}")
            except Exception as s3e:
                logger.warning(f"Failed to upload error log to S3: {s3e}")
    except Exception as log_ex:
        logger.critical(f"Failed to run log_error helper: {log_ex}", exc_info=True)


def xyxy_to_xywh(box: List[float], img_w: Optional[int] = None, img_h: Optional[int] = None) -> Dict[str, int]:
    """Convert bounding box from xyxy to xywh format"""
    x0, y0, x1, y1 = box
    x0 = max(0, int(round(x0)))
    y0 = max(0, int(round(y0)))
    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    
    if img_w is not None:
        x0 = min(x0, img_w)
        x1 = min(x1, img_w)
    if img_h is not None:
        y0 = min(y0, img_h)
        y1 = min(y1, img_h)
    
    x = int(x0)
    y = int(y0)
    w = int(max(0, x1 - x0))
    h = int(max(0, y1 - y0))
    
    return {'x': x, 'y': y, 'w': w, 'h': h}


# =============================================================================
# CONTEXT UNDERSTANDING FUNCTIONS
# =============================================================================
def parse_contextual_search(search_text: str) -> ContextualPhrase:
    """
    Parse search text to understand context and relationships.
    Examples:
    - "Student standing in the class" -> main: student, context: standing, spatial: in the class
    - "Person sitting on chair" -> main: person, context: sitting, spatial: on chair
    - "Woman in red dress" -> main: woman, context: red, dress
    """
    search_lower = search_text.lower().strip()
    
    # Common action words
    action_words = ['standing', 'sitting', 'running', 'walking', 'lying', 'jumping', 
                   'holding', 'eating', 'drinking', 'reading', 'writing', 'playing',
                   'wearing']
    
    # Common spatial relations
    spatial_preps = ['in', 'on', 'at', 'near', 'beside', 'under', 'over', 'behind', 
                    'in front of', 'next to', 'between', 'among']
    
    # Common objects that are usually the main subject
    common_subjects = ['person', 'student', 'teacher', 'man', 'woman', 'women', 'child', 'boy', 
                      'girl', 'dog', 'cat', 'car', 'bicycle', 'bird', 'horse']
    
    words = search_lower.split()
    
    # Identify main object
    main_object = None
    for subject in common_subjects:
        if subject in search_lower:
            main_object = subject
            break
    
    # If no common subject found, take the first noun-like word
    if not main_object and words:
        main_object = words[0]
    
    # Identify actions
    found_actions = [word for word in words if word in action_words]
    
    # Identify spatial context
    spatial_context = []
    for prep in spatial_preps:
        if prep in search_lower:
            idx = search_lower.index(prep)
            after_prep = search_lower[idx:].split(None, 2)
            if len(after_prep) > 1:
                spatial_context.append(' '.join(after_prep[:2]))
    
    # Extract context words (all words except main object)
    context_words = [w for w in words if w != main_object and w not in ['the', 'a', 'an']]
    
    return ContextualPhrase(
        full_phrase=search_text,
        main_object=main_object or search_text,
        context_words=context_words,
        action_words=found_actions,
        spatial_relations=spatial_context
    )


def format_contextual_prompt(phrase: ContextualPhrase) -> List[str]:
    """
    Format the prompt for OWL-ViT v2 (returns list of strings)
    This is different from Grounding DINO which uses period-separated strings
    """
    prompts = []
    
    # Add the full phrase as primary
    prompts.append(phrase.full_phrase.lower())
    
    # Add contextual variations that preserve meaning
    if phrase.action_words and phrase.main_object:
        for action in phrase.action_words:
            prompts.append(f"{action} {phrase.main_object}")
            prompts.append(f"{phrase.main_object} {action}")
    
    # Add with "wearing" if clothing mentioned
    if any(word in phrase.context_words for word in ['dress', 'shirt', 'hat', 'clothes', 'red', 'blue', 'green', 'black', 'white']):
        prompts.append(f"{phrase.main_object} wearing {' '.join(phrase.context_words)}")
    
    # Add spatial context
    if phrase.spatial_relations and phrase.main_object:
        for spatial in phrase.spatial_relations:
            prompts.append(f"{phrase.main_object} {spatial}")
    
    # Add the main object as fallback
    prompts.append(phrase.main_object)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_prompts = []
    for p in prompts:
        p_clean = p.strip().lower()
        if p_clean and p_clean not in seen:
            seen.add(p_clean)
            unique_prompts.append(p_clean)
    
    logger.info(f"[Contextual Prompt] Original: '{phrase.full_phrase}' -> Queries: {unique_prompts}")
    
    return unique_prompts


def validate_contextual_match(detected_label: str, original_phrase: ContextualPhrase, 
                             confidence: float) -> Tuple[bool, float]:
    """
    Validate if the detection matches the contextual search intent.
    Returns (is_valid, adjusted_confidence)
    """
    if not ENABLE_CONTEXT_VALIDATION:
        return True, confidence
    
    detected_lower = detected_label.lower()
    
    # Check if main object is detected
    if original_phrase.main_object not in detected_lower:
        # If we're looking for "student" but found "person", it might be acceptable with lower confidence
        if original_phrase.main_object in ['student', 'teacher'] and 'person' in detected_lower:
            return True, confidence * 0.8
        # Not the object we're looking for
        return False, 0.0
    
    # If action words were specified, boost confidence if they're in the detection
    if original_phrase.action_words:
        action_found = any(action in detected_lower for action in original_phrase.action_words)
        if action_found:
            return True, min(confidence * 1.2, 1.0)
        elif len(original_phrase.action_words) > 0:
            return True, confidence * 0.7
    
    return True, confidence


# =============================================================================
# DETECTION PROCESSING FUNCTIONS
# =============================================================================
def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union between two boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def smart_nms_with_context(boxes: List[List[float]], scores: List[float], 
                          labels: List[str], iou_threshold: float = 0.5,
                          original_phrase: Optional[ContextualPhrase] = None) -> Tuple[List, List, List]:
    """Smart NMS that considers context when filtering overlapping detections"""
    if len(boxes) == 0:
        return [], [], []
    
    logger.info(f"[Smart NMS] Processing {len(boxes)} detections")
    
    # Validate and adjust scores based on context
    if original_phrase:
        adjusted_scores = []
        valid_indices = []
        for i, (label, score) in enumerate(zip(labels, scores)):
            is_valid, adj_score = validate_contextual_match(label, original_phrase, score)
            if is_valid and adj_score >= BOX_THRESHOLD:
                adjusted_scores.append(adj_score)
                valid_indices.append(i)
        
        if not valid_indices:
            logger.info("[Smart NMS] No contextually valid detections found")
            return [], [], []
        
        # Filter to valid detections
        boxes = [boxes[i] for i in valid_indices]
        scores = adjusted_scores
        labels = [labels[i] for i in valid_indices]
    
    # Standard NMS with confidence ranking
    sorted_indices = np.argsort(scores)[::-1]
    keep_indices = []
    
    for idx in sorted_indices:
        current_box = boxes[idx]
        should_keep = True
        
        for kept_idx in keep_indices:
            kept_box = boxes[kept_idx]
            iou = calculate_iou(current_box, kept_box)
            
            if iou > iou_threshold:
                should_keep = False
                break
        
        if should_keep:
            keep_indices.append(idx)
    
    kept_boxes = [boxes[i] for i in keep_indices]
    kept_scores = [scores[i] for i in keep_indices]
    kept_labels = [labels[i] for i in keep_indices]
    
    logger.info(f"[Smart NMS] Kept {len(kept_boxes)} detections after NMS")
    
    return kept_boxes, kept_scores, kept_labels


# =============================================================================
# IMAGE PROCESSING FUNCTIONS
# =============================================================================
def enhance_image(img: Image.Image) -> Image.Image:
    """Apply adaptive enhancement based on image characteristics"""
    try:
        img_array = np.array(img)
        mean_brightness = np.mean(img_array)
        
        if mean_brightness < 70:  # Dark image
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.3)
        elif mean_brightness > 200:  # Bright image
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.3)
        else:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.2)
        
        # Always apply slight sharpening
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.1)
        
        return img
    except Exception as e:
        logger.warning(f"Image enhancement failed: {e}")
        return img


def draw_detections_with_context(image: Image.Image, boxes: List[List[float]], 
                                 labels: List[str], scores: List[float],
                                 search_text: str) -> Image.Image:
    """Draw bounding boxes with context-aware labeling"""
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    
    def get_color(score):
        if score > 0.7:
            return "lime"
        elif score > 0.5:
            return "orange"
        else:
            return "red"
    
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        x0, y0, x1, y1 = [int(coord) for coord in box]
        color = get_color(score)
        thickness = max(2, int(3 * (0.5 + score * 0.5)))
        
        # Draw box
        draw.rectangle([x0, y0, x1, y1], outline=color, width=thickness)
        
        # Prepare text
        text = f"{label} ({score:.2f})"
        if search_text.lower() != label.lower():
            text = f"{search_text}: {text}"
        
        try:
            bbox = draw.textbbox((x0, y0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            tw, th = len(text) * 8, 16
        
        text_y = max(0, y0 - th - 6)
        draw.rectangle([x0, text_y, x0 + tw + 8, text_y + th + 6], fill=color)
        draw.text((x0 + 4, text_y + 3), text, fill="white", font=font)
    
    # Add search context as title
    title = f"Search: {search_text}"
    draw.text((10, 10), title, fill="yellow", font=font)
    
    return img_draw


# =============================================================================
# MODEL LOADING FUNCTIONS
# =============================================================================
def load_model_components(model_id: str = MODEL_ID, device: str = "cpu"):
    """Load processor and model from HuggingFace - OWL-ViT v2 specific"""
    try:
        logger.info(f"[load_model_components] Loading OWL-ViT v2 from HuggingFace: {model_id}")
        processor = Owlv2Processor.from_pretrained(model_id)
        model = Owlv2ForObjectDetection.from_pretrained(model_id)
        model.to(device)
        model.eval()
        logger.info("[load_model_components] Successfully loaded OWL-ViT v2 model and moved to device")
        return processor, model
    except Exception as e:
        log_error(e, {"fn": "load_model_components", "model_id": model_id})
        raise


def enable_gpu_optimizations(model: torch.nn.Module) -> Dict[str, Any]:
    """Enable GPU optimizations for the model"""
    status = {"fp16": False, "compiled": False, "cudnn_benchmark": False, "tf32": False, "model": model}
    
    try:
        if not torch.cuda.is_available():
            logger.info("[enable_gpu_optimizations] CUDA not available; skipping GPU optimizations")
            return status

        try:
            torch.backends.cudnn.benchmark = True
            status["cudnn_benchmark"] = True
        except Exception:
            logger.debug("cudnn.benchmark not set")

        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            status["tf32"] = True
        except Exception:
            logger.debug("TF32 not enabled")

        if not DISABLE_FP16:
            try:
                model = model.half()
                status["fp16"] = True
                logger.info("[enable_gpu_optimizations] Model converted to FP16")
            except Exception as e:
                logger.warning(f"[enable_gpu_optimizations] FP16 conversion failed: {e}")

        if not DISABLE_COMPILE:
            try:
                compiled = torch.compile(model)
                model = compiled
                status["compiled"] = True
                logger.info("[enable_gpu_optimizations] torch.compile applied")
            except Exception as e:
                logger.info(f"[enable_gpu_optimizations] torch.compile not applied: {e}")

        status["model"] = model
    except Exception as e:
        log_error(e, {"fn": "enable_gpu_optimizations"})
    
    return status


# =============================================================================
# INFERENCE CLASS
# =============================================================================
class VisualSearchInference:
    """Enhanced inference class with visual search capabilities - OWL-ViT v2"""
    
    def __init__(self):
        self.device = None
        self.processor: Optional[Owlv2Processor] = None
        self.model: Optional[Owlv2ForObjectDetection] = None
        self.model_loaded = False
        self.use_fp16 = False
        logger.info("VisualSearchInference instance created (OWL-ViT v2)")

    def decode_base64_image(self, base64_string: str) -> Image.Image:
        """Decode base64 encoded image"""
        try:
            if base64_string.startswith('data:image'):
                base64_string = base64_string.split(',', 1)[1]
            image_bytes = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            log_error(e, {"fn": "decode_base64_image"})
            raise ValueError(f"Image decode failed: {e}")

    def encode_image_to_base64(self, image: Image.Image) -> str:
        """Encode image to base64 string"""
        buffered = io.BytesIO()
        image.convert('RGB').save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

    def preprocess_image(self, image: Image.Image, max_size: int = 800) -> Image.Image:
        """Preprocess and resize image if needed"""
        try:
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            return image
        except Exception as e:
            log_error(e, {"fn": "preprocess_image"})
            return image

    def detect_with_context(self, image: Image.Image, search_text: str, 
                           box_threshold: float = BOX_THRESHOLD,
                           context_threshold: float = CONTEXT_MATCH_THRESHOLD) -> Dict[str, Any]:
        """Perform contextual visual search detection with OWL-ViT v2"""
        try:
            # Parse the search phrase for context
            context_phrase = parse_contextual_search(search_text)
            
            # Apply enhancement
            processed_img = enhance_image(image)
            processed_img = self.preprocess_image(processed_img)
            
            # Format prompt for OWL-ViT (list of strings)
            text_queries = format_contextual_prompt(context_phrase)
            
            logger.info(f"[Visual Search] Original: '{search_text}'")
            logger.info(f"[Visual Search] Main object: '{context_phrase.main_object}'")
            logger.info(f"[Visual Search] Actions: {context_phrase.action_words}")
            logger.info(f"[Visual Search] Spatial: {context_phrase.spatial_relations}")
            logger.info(f"[Visual Search] Box threshold: {box_threshold}, Context threshold: {context_threshold}")
            
            # Run detection with OWL-ViT v2
            inputs = self.processor(
                text=text_queries,  # List of strings for OWL-ViT
                images=processed_img,
                return_tensors="pt"
            )
            
            # Handle FP16
            if self.use_fp16:
                safe_inputs = {}
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        if v.is_floating_point():
                            safe_inputs[k] = v.to(self.device).half()
                        else:
                            safe_inputs[k] = v.to(self.device)
                    else:
                        safe_inputs[k] = v
                inputs = safe_inputs
            else:
                inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            target_sizes = torch.tensor([processed_img.size[::-1]], dtype=torch.long).to(self.device)
            
            # OWL-ViT v2 post-processing
            results = self.processor.post_process_object_detection(
                outputs=outputs,
                threshold=box_threshold,  # Note: 'threshold' not 'box_threshold'
                target_sizes=target_sizes
            )[0]
            
            # Extract results
            boxes = results["boxes"].cpu().numpy().tolist() if len(results["boxes"]) > 0 else []
            scores = results["scores"].cpu().numpy().tolist() if len(results["scores"]) > 0 else []
            labels_idx = results["labels"].cpu().numpy().tolist() if len(results["labels"]) > 0 else []
            
            # Map label indices back to text queries
            processed_labels = [text_queries[idx] if idx < len(text_queries) else "object" 
                              for idx in labels_idx]
            
            logger.info(f"[Detection] Raw: {len(boxes)} detections")
            
            # Apply contextual validation and filtering
            if boxes:
                validated_boxes = []
                validated_scores = []
                validated_labels = []
                
                for box, score, label in zip(boxes, scores, processed_labels):
                    is_valid, adjusted_score = validate_contextual_match(
                        label, context_phrase, score
                    )
                    
                    if is_valid and adjusted_score >= context_threshold:
                        validated_boxes.append(box)
                        validated_scores.append(adjusted_score)
                        validated_labels.append(label)
                        logger.info(f"[Validation] Accepted: {label} (score: {score:.3f} -> {adjusted_score:.3f})")
                    else:
                        logger.info(f"[Validation] Rejected: {label} (score: {score:.3f}, valid: {is_valid})")
                
                # Apply Smart NMS with context
                final_boxes, final_scores, final_labels = smart_nms_with_context(
                    validated_boxes, validated_scores, validated_labels,
                    iou_threshold=NMS_THRESHOLD, original_phrase=context_phrase
                )
                
                logger.info(f"[Final] {len(final_boxes)} contextually valid detections")
                
                return {
                    'boxes': final_boxes,
                    'scores': final_scores,
                    'labels': final_labels,
                    'detected': len(final_boxes) > 0,
                    'context_phrase': context_phrase
                }
            
            # No detections found
            logger.info("[Detection] No valid detections found for the search context")
            return {
                'boxes': [],
                'scores': [],
                'labels': [],
                'detected': False,
                'context_phrase': context_phrase
            }
            
        except Exception as e:
            log_error(e, {"fn": "detect_with_context", "search_text": search_text})
            return {
                'boxes': [],
                'scores': [],
                'labels': [],
                'detected': False,
                'error': str(e)
            }


# =============================================================================
# SAGEMAKER ENTRYPOINTS
# =============================================================================
def model_fn(model_dir):
    """SageMaker model load entrypoint"""
    global inference_handler
    logger.info("=" * 80)
    logger.info("OWL-ViT v2 VISUAL SEARCH MODEL LOADING STARTED")
    logger.info("=" * 80)
    start_time = time.time()
    
    try:
        inference_handler = VisualSearchInference()
        
        if torch.cuda.is_available():
            inference_handler.device = "cuda"
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                cuda_version = torch.version.cuda
                logger.info(f"[OK] GPU AVAILABLE: {gpu_name} ({gpu_memory:.2f} GB)")
                logger.info(f"[OK] CUDA VERSION: {cuda_version}")
            except Exception:
                logger.debug("Could not query GPU name/memory")
        else:
            inference_handler.device = "cpu"
            logger.warning("[WARNING] CUDA NOT AVAILABLE - USING CPU")
        
        logger.info(f"[INFO] Loading OWL-ViT v2 from HuggingFace: {MODEL_ID}")
        proc, model = load_model_components(MODEL_ID, inference_handler.device)
        inference_handler.processor = proc
        
        optim = enable_gpu_optimizations(model)
        inference_handler.model = optim.get("model", model)
        inference_handler.use_fp16 = optim.get("fp16", False)
        inference_handler.model_loaded = True
        
        load_time = time.time() - start_time
        
        logger.info("=" * 80)
        logger.info(f"[SUCCESS] OWL-ViT v2 MODEL LOADED IN {load_time:.2f}s")
        logger.info(f"[INFO] Device: {inference_handler.device.upper()} | FP16: {inference_handler.use_fp16}")
        logger.info(f"[INFO] Context Validation: {ENABLE_CONTEXT_VALIDATION}")
        logger.info(f"[INFO] Context Threshold: {CONTEXT_MATCH_THRESHOLD}")
        logger.info(f"[INFO] Box Threshold: {BOX_THRESHOLD}")
        logger.info("=" * 80)
        
        return inference_handler
    except Exception as e:
        log_error(e, {"fn": "model_fn"})
        raise


def input_fn(request_body, content_type='application/json'):
    """Deserialize input JSON payload"""
    try:
        if isinstance(request_body, (bytes, bytearray)):
            request_body = request_body.decode('utf-8')
        if content_type == 'application/json':
            return json.loads(request_body)
        raise ValueError(f"Unsupported content type: {content_type}")
    except Exception as e:
        log_error(e, {"fn": "input_fn", "content_type": content_type})
        raise


def predict_fn(input_data, model):
    """Main prediction with batch processing - returns ONLY best frame"""
    global inference_handler
    logger.info("=" * 80)
    logger.info("VISUAL SEARCH BATCH PREDICTION STARTED")
    logger.info("=" * 80)
    start_time = time.time()
    
    try:
        if not inference_handler or not inference_handler.model_loaded:
            raise RuntimeError("Model not loaded")
        
        # Extract input parameters
        orgid = input_data.get('orgid')
        processid = input_data.get('processid')
        cam_id = input_data.get('cam_id')
        search_text = input_data.get('search_text', '')
        frames = input_data.get('frames', [])
        batch_size = input_data.get('batch_size', 16)
        annotated_frame = input_data.get('annotated_frame', True)
        
        # Validate required fields
        if orgid is None:
            raise ValueError("orgid is required and must be an integer")
        if processid is None:
            raise ValueError("processid is required and must be an integer")
        if cam_id is None:
            raise ValueError("cam_id is required and must be an integer")
        if not search_text:
            raise ValueError("search_text cannot be empty")
        if not frames or not isinstance(frames, list):
            raise ValueError("frames must be a non-empty array")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        # Configuration parameters
        box_threshold = float(input_data.get('box_threshold', BOX_THRESHOLD))
        context_threshold = float(input_data.get('context_threshold', CONTEXT_MATCH_THRESHOLD))
        nms_threshold = float(input_data.get('nms_threshold', NMS_THRESHOLD))
        
        total_frames = len(frames)
        logger.info(f"Batch Processing Request: '{search_text}'")
        logger.info(f"Org={orgid} Proc={processid} Cam={cam_id}")
        logger.info(f"Total Frames={total_frames}, Batch Size={batch_size}")
        logger.info(f"Thresholds: box={box_threshold}, context={context_threshold}, nms={nms_threshold}")
        
        # Initialize tracking for best frame
        best_frame_data = {
            'frame_id': None,
            'annotated_image': None,
            'confidence': 0.0,
            'global_frame_idx': -1
        }
        
        total_frames_found = 0
        total_inference_time = 0
        processing_errors = []
        
        # Calculate number of batches
        num_batches = math.ceil(total_frames / batch_size)
        logger.info(f"Processing in {num_batches} batches")
        
        # Process frames in batches
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_frames)
            current_batch = frames[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_idx + 1}/{num_batches} (frames {start_idx}-{end_idx-1})")
            
            # Process each frame in current batch
            for frame_idx, frame_encoding in enumerate(current_batch):
                global_frame_idx = start_idx + frame_idx
                
                try:
                    # Generate unique frame_id
                    frame_id = generate_frame_id()
                    
                    # Add tiny delay to ensure unique timestamps
                    time.sleep(0.001)
                    
                    # Decode image
                    decode_start = time.time()
                    image = inference_handler.decode_base64_image(frame_encoding)
                    decode_time = time.time() - decode_start
                    
                    # Perform contextual visual search
                    inference_start = time.time()
                    detection = inference_handler.detect_with_context(
                        image, search_text, box_threshold, context_threshold
                    )
                    inference_time = time.time() - inference_start
                    total_inference_time += inference_time
                    
                    # Check if object was detected
                    if detection.get('detected') and detection.get('boxes'):
                        boxes = detection['boxes']
                        scores = detection['scores']
                        labels = detection['labels']
                        
                        # Select the best detection (highest confidence)
                        best_idx = int(np.argmax(scores))
                        best_box = boxes[best_idx]
                        best_score = scores[best_idx]
                        best_label = labels[best_idx] if best_idx < len(labels) else "object"
                        
                        confidence = round(float(best_score), 2)
                        
                        logger.info(f"[Frame {global_frame_idx}] FOUND: '{best_label}' with confidence {confidence}")
                        
                        total_frames_found += 1
                        
                        # Check if this is the best frame so far
                        if confidence > best_frame_data['confidence']:
                            logger.info(f"[Frame {global_frame_idx}] NEW BEST FRAME: {confidence} > {best_frame_data['confidence']}")
                            
                            # Generate annotated frame if requested
                            if annotated_frame:
                                try:
                                    annotated_img = draw_detections_with_context(
                                        image, [best_box], [best_label], [best_score], search_text
                                    )
                                    annotated_encoding = inference_handler.encode_image_to_base64(annotated_img)
                                    
                                    # Update best frame data
                                    best_frame_data['frame_id'] = frame_id
                                    best_frame_data['annotated_image'] = annotated_encoding
                                    best_frame_data['confidence'] = confidence
                                    best_frame_data['global_frame_idx'] = global_frame_idx
                                    
                                except Exception as e:
                                    logger.error(f"[Frame {global_frame_idx}] Failed to generate annotation: {e}")
                                    processing_errors.append({
                                        "frame_index": global_frame_idx,
                                        "error": f"Annotation failed: {str(e)}"
                                    })
                            else:
                                # No annotation needed, just track metadata
                                best_frame_data['frame_id'] = frame_id
                                best_frame_data['confidence'] = confidence
                                best_frame_data['global_frame_idx'] = global_frame_idx
                    else:
                        logger.debug(f"[Frame {global_frame_idx}] Not found")
                    
                    # Cleanup after each frame
                    try:
                        del detection
                        del image
                        if inference_handler.device == "cuda":
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    
                except Exception as e:
                    logger.error(f"[Frame {global_frame_idx}] Processing error: {e}")
                    processing_errors.append({
                        "frame_index": global_frame_idx,
                        "error": str(e),
                        "error_type": type(e).__name__
                    })
                    continue
            
            # Cleanup after each batch
            gc.collect()
            if inference_handler.device == "cuda":
                torch.cuda.empty_cache()
        
        # Calculate statistics
        detection_rate = round((total_frames_found / total_frames) * 100, 2) if total_frames > 0 else 0.0
        total_time = time.time() - start_time
        average_inference_per_frame = round((total_inference_time / total_frames) * 1000, 2) if total_frames > 0 else 0
        
        logger.info("=" * 80)
        logger.info(f"BATCH PROCESSING COMPLETE")
        logger.info(f"Total Frames: {total_frames} | Found: {total_frames_found} | Rate: {detection_rate}%")
        
        # Prepare detected_frames dictionary (empty or single best frame)
        detected_frames = {}
        confidence_list = []
        
        if best_frame_data['frame_id'] is not None:
            if annotated_frame and best_frame_data['annotated_image']:
                detected_frames[best_frame_data['frame_id']] = best_frame_data['annotated_image']
            confidence_list.append(best_frame_data['confidence'])
            logger.info(f"BEST FRAME: Frame {best_frame_data['global_frame_idx']} | ID: {best_frame_data['frame_id']} | Confidence: {best_frame_data['confidence']}")
        else:
            logger.info("NO FRAMES DETECTED")
        
        logger.info(f"Total Time: {int(total_time * 1000)}ms | Inference Time: {int(total_inference_time * 1000)}ms")
        logger.info(f"Average per Frame: {average_inference_per_frame}ms")
        logger.info("=" * 80)
        
        # Prepare response - IDENTICAL FORMAT AS GROUNDING DINO
        response = {
            "orgid": orgid,
            "processid": processid,
            "cam_id": cam_id,
            "search_text": search_text,
            "total_frames_processed": total_frames,
            "total_frames_found": total_frames_found,
            "detection_rate": detection_rate,
            "detected_frames": detected_frames,  # Dictionary with single best frame or empty
            "confidence": confidence_list,  # List with single confidence or empty
            "processing_time": f"{int(total_time * 1000)}ms",
            "total_inference_time": f"{int(total_inference_time * 1000)}ms",
            "average_inference_per_frame": f"{average_inference_per_frame}ms",
            "device": inference_handler.device,
            "box_threshold_used": box_threshold,
            "context_threshold_used": context_threshold
        }
        
        # Add errors if any occurred
        if processing_errors:
            response["errors"] = processing_errors
            logger.warning(f"Encountered {len(processing_errors)} processing errors")
        
        return response
        
    except Exception as e:
        log_error(e, {"fn": "predict_fn", "search_text": input_data.get('search_text', 'N/A') if isinstance(input_data, dict) else 'N/A'})
        
        # Return error response - IDENTICAL FORMAT AS GROUNDING DINO
        return {
            'error': str(e),
            'error_type': type(e).__name__,
            'orgid': input_data.get('orgid') if isinstance(input_data, dict) else None,
            'processid': input_data.get('processid') if isinstance(input_data, dict) else None,
            'cam_id': input_data.get('cam_id') if isinstance(input_data, dict) else None,
            'search_text': input_data.get('search_text', '') if isinstance(input_data, dict) else '',
            'total_frames_processed': 0,
            'total_frames_found': 0,
            'detection_rate': 0.0,
            'detected_frames': {},
            'confidence': [],
            'processing_time': f"{int((time.time() - start_time) * 1000)}ms",
            'total_inference_time': "0ms",
            'average_inference_per_frame': "0ms",
            'device': inference_handler.device if inference_handler else "unknown"
        }


def output_fn(prediction, content_type='application/json'):
    """Serialize prediction output"""
    try:
        if content_type == 'application/json':
            return json.dumps(prediction)
        raise ValueError(f"Unsupported content type: {content_type}")
    except Exception as e:
        log_error(e, {"fn": "output_fn"})
        raise


# =============================================================================
# CLOUDWATCH METRICS (OPTIONAL)
# =============================================================================
def get_sagemaker_endpoint_name() -> Optional[str]:
    """Get SageMaker endpoint name from environment"""
    return SAGEMAKER_ENDPOINT_NAME_ENV or os.getenv("SAGEMAKER_ENDPOINT_NAME")


def fetch_gpu_cloudwatch_metrics(namespace: str = "DCGM",
                                 gpu_util_metric: str = "GPUUtilization",
                                 gpu_mem_metric: str = "GPUMemoryUsed",
                                 period: int = 60,
                                 stat: str = "Average",
                                 endpoint_name: Optional[str] = None) -> Dict[str, Optional[float]]:
    """Fetch GPU metrics from CloudWatch"""
    result = {"gpu_utilization": None, "gpu_memory_used": None, "notes": []}
    
    try:
        ep = endpoint_name or get_sagemaker_endpoint_name()
        if ep:
            cw = boto3.client("cloudwatch")
            dims = [{"Name": "EndpointName", "Value": ep}]
            
            # Get GPU utilization
            try:
                resp = cw.get_metric_statistics(
                    Namespace=namespace,
                    MetricName=gpu_util_metric,
                    Dimensions=dims,
                    StartTime=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(time.time() - period * 2)),
                    EndTime=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(time.time())),
                    Period=period,
                    Statistics=[stat],
                )
                if resp.get("Datapoints"):
                    latest = max(resp["Datapoints"], key=lambda d: d.get("Timestamp"))
                    result["gpu_utilization"] = float(latest.get(stat))
            except Exception:
                pass
            
            # Get GPU memory
            try:
                resp = cw.get_metric_statistics(
                    Namespace=namespace,
                    MetricName=gpu_mem_metric,
                    Dimensions=dims,
                    StartTime=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(time.time() - period * 2)),
                    EndTime=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(time.time())),
                    Period=period,
                    Statistics=[stat],
                )
                if resp.get("Datapoints"):
                    latest = max(resp["Datapoints"], key=lambda d: d.get("Timestamp"))
                    result["gpu_memory_used"] = float(latest.get(stat))
            except Exception:
                pass
            
            result["notes"].append(f"used EndpointName={ep}")
    except Exception as e:
        logger.debug(f"CloudWatch metrics fetch failed: {e}")
    
    return result
