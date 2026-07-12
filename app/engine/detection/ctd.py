"""
Text Detection - Dual approach:
1. Try CTD YOLO ONNX if model available and compatible (high quality)
2. Fall back to OpenCV-based detection (always works)
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("detection")

_ort_session = None
_use_onnx = False


def _resolve_model_path() -> str | None:
    """Resolve the location of the comictextdetector.pt.onnx model."""
    search_paths = [
        # Check environment variable
        os.path.join(os.environ.get("MODEL_DIR", ""), "detection", "comictextdetector.pt.onnx"),
        os.path.join(os.environ.get("MODEL_DIR", ""), "comictextdetector.pt.onnx"),
        # Docker volume path
        "/app/models/weights/detection/comictextdetector.pt.onnx",
        # Local paths relative to project root
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "detection", "comictextdetector.pt.onnx"),
        os.path.join("models", "detection", "comictextdetector.pt.onnx"),
    ]
    for path in search_paths:
        if path and os.path.exists(path):
            log.info(f"Resolved detection model path: {path}")
            return os.path.abspath(path)
    return None


async def load_model(model_path: str = None, device: str = "cpu"):
    """Load CTD ONNX model using onnxruntime. Falls back to OpenCV on failure."""
    global _ort_session, _use_onnx

    if _ort_session is not None:
        return

    path = model_path or _resolve_model_path()
    if not path:
        log.warning("comictextdetector.pt.onnx model file not found. Falling back to OpenCV detection.")
        _use_onnx = False
        return

    def _init_session():
        global _ort_session, _use_onnx
        try:
            import onnxruntime as ort
            
            # Select execution providers
            # We force CPUExecutionProvider to save VRAM and avoid memory corruption / shape collisions with Ollama/PyTorch
            providers = ['CPUExecutionProvider']
            log.info(f"Initializing ONNX session for detection with providers: {providers}")
            _ort_session = ort.InferenceSession(path, providers=providers)
            _use_onnx = True
            log.info(f"Loaded CTD ONNX model successfully on: {_ort_session.get_providers()}")
        except Exception as e:
            log.error(f"Failed to load ONNX model: {e}. Falling back to OpenCV.")
            _ort_session = None
            _use_onnx = False

    await asyncio.get_event_loop().run_in_executor(None, _init_session)


async def detect(
    img_rgb: np.ndarray,
    detection_size: int = 1024,
    text_threshold: float = 0.3,
    box_threshold: float = 0.3,
    device: str = "cpu",
) -> tuple[list, np.ndarray, np.ndarray]:
    """Detect text regions. Uses ONNX if loaded, otherwise OpenCV contours."""
    if _use_onnx and _ort_session is not None:
        try:
            return await _detect_onnx(img_rgb, text_threshold, box_threshold)
        except Exception as e:
            log.error(f"ONNX detection failed: {e}. Falling back to OpenCV.")
            
    return await _detect_opencv(img_rgb, box_threshold)


async def _detect_onnx(
    img_rgb: np.ndarray,
    text_threshold: float,
    box_threshold: float,
) -> tuple[list, np.ndarray, np.ndarray]:
    """ONNX-based text and bubble detection + text segmentation."""
    global _ort_session
    loop = asyncio.get_event_loop()

    def _run():
        h, w = img_rgb.shape[:2]
        
        # Preprocessing: resize to 1024x1024
        img_resized = cv2.resize(img_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        img_float = img_resized.astype(np.float32) / 255.0
        # HWC to CHW
        img_chw = np.transpose(img_float, (2, 0, 1))
        # Add batch dim
        input_data = np.expand_dims(img_chw, axis=0)

        # Run ONNX inference
        outputs = _ort_session.run(None, {"images": input_data})
        blk, seg, det = outputs  # blk: [1, 64512, 7], seg: [1, 1, 1024, 1024]
        
        # --- 1. Parse Text Box Detections (blk) ---
        boxes = []
        confidences = []
        
        # Scale factors back to original image
        scale_x = w / 1024.0
        scale_y = h / 1024.0
        
        # blk columns: x_center, y_center, box_w, box_h, conf, prob_class_0, prob_class_1
        # Class 0: bubble, Class 1: text
        detections = blk[0]
        for det_row in detections:
            obj_conf = det_row[4]
            if obj_conf < box_threshold:
                continue
                
            # Class score for text (class 1 in CTD ONNX)
            text_score = obj_conf * det_row[6]
            if text_score < text_threshold:
                continue
                
            cx, cy, bw, bh = det_row[0], det_row[1], det_row[2], det_row[3]
            
            # Map back to original image coordinates
            x_min = int((cx - bw / 2.0) * scale_x)
            y_min = int((cy - bh / 2.0) * scale_y)
            x_max = int((cx + bw / 2.0) * scale_x)
            y_max = int((cy + bh / 2.0) * scale_y)
            
            # Clip values
            x_min = max(0, min(w - 1, x_min))
            y_min = max(0, min(h - 1, y_min))
            x_max = max(0, min(w - 1, x_max))
            y_max = max(0, min(h - 1, y_max))
            
            bw_orig = x_max - x_min
            bh_orig = y_max - y_min
            
            if bw_orig > 4 and bh_orig > 4:
                boxes.append([x_min, y_min, bw_orig, bh_orig])
                confidences.append(float(text_score))

        # Apply Non-Maximum Suppression (NMS)
        textlines = []
        mask_raw = np.zeros((h, w), dtype=np.uint8)
        
        if boxes:
            # cv2.dnn.NMSBoxes returns indices of the kept boxes
            nms_indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=text_threshold, nms_threshold=0.3)
            
            # NMSBoxes return type can vary by OpenCV version
            if len(nms_indices) > 0:
                indices = nms_indices.flatten() if hasattr(nms_indices, "flatten") else nms_indices
                for idx_i, idx in enumerate(indices):
                    x, y, bw_box, bh_box = boxes[idx]
                    bbox = [x, y, x + bw_box, y + bh_box]
                    textlines.append({
                        "index": idx_i,
                        "bbox": bbox,
                        "polygon": [
                            [bbox[0], bbox[1]],
                            [bbox[2], bbox[1]],
                            [bbox[2], bbox[3]],
                            [bbox[0], bbox[3]]
                        ],
                        "confidence": confidences[idx]
                    })
                    # Only use character-level segmentation for inpainting to prevent boxy leaks outside bubble borders
                    # cv2.rectangle(mask_raw, (x, y), (x + bw_box, y + bh_box), 255, -1)

        # --- 2. Parse Text Segmentation Map (seg) ---
        seg_prob = seg[0, 0, :, :]  # 1024x1024
        
        # Resize to original size
        seg_prob_resized = cv2.resize(seg_prob, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Threshold to create binary mask
        seg_mask = (seg_prob_resized > 0.25).astype(np.uint8) * 255
        
        # Merge bounding box raw mask and segmentation mask to make it extra robust
        mask_raw = cv2.bitwise_or(mask_raw, seg_mask)
        
        # Dilate mask for optimal inpainting boundary coverage
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask_raw, kernel, iterations=4)

        return textlines, mask_raw, mask

    return await loop.run_in_executor(None, _run)


async def _detect_opencv(img_rgb: np.ndarray, threshold: float) -> tuple[list, np.ndarray, np.ndarray]:
    """OpenCV-based text detection using contour analysis (CPU fallback)."""
    loop = asyncio.get_event_loop()

    def _run():
        h, w = img_rgb.shape[:2]
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # Adaptive threshold to find text-like regions
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)

        # Morphological operations to connect text components
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        dilated_h = cv2.dilate(binary, kernel_h, iterations=2)
        dilated_v = cv2.dilate(binary, kernel_v, iterations=2)
        combined = cv2.bitwise_or(dilated_h, dilated_v)

        # Close gaps
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=3)

        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        textlines = []
        mask_raw = np.zeros((h, w), dtype=np.uint8)

        for i, contour in enumerate(contours):
            x, y, bw, bh = cv2.boundingRect(contour)
            area = bw * bh

            # Filter: minimum size
            if bw < 15 or bh < 15 or area < 200:
                continue

            # Filter: not too large (max 40% of image)
            if area > (w * h * 0.4):
                continue

            # Filter: aspect ratio
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if aspect > 10:
                continue

            bbox = [x, y, x + bw, y + bh]

            contour_area = cv2.contourArea(contour)
            confidence = min(contour_area / area, 1.0)

            textlines.append({
                "index": i,
                "bbox": bbox,
                "polygon": [[x,y],[x+bw,y],[x+bw,y+bh],[x,y+bh]],
                "confidence": confidence,
            })
            cv2.drawContours(mask_raw, [contour], -1, 255, -1)

        # Dilate mask for inpainting coverage
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.dilate(mask_raw, kernel, iterations=5)

        return textlines, mask_raw, mask

    return await loop.run_in_executor(None, _run)


def unload():
    global _ort_session, _use_onnx
    _ort_session = None
    _use_onnx = False
    log.info("CTD ONNX model unloaded.")
