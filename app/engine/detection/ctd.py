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


def _merge_overlapping_textlines(
    textlines: list[dict], overlap_threshold: float = 0.22
) -> list[dict]:
    """Merge duplicate detector boxes that would render on top of each other.

    CTD may return a broad text block and one or more sub-blocks for the same
    speech bubble. NMS does not always suppress these nested boxes. Merging
    only when the intersection covers a substantial part of the smaller box
    preserves separate nearby bubbles while preventing duplicate rendering.
    """
    if len(textlines) < 2:
        return textlines

    parents = list(range(len(textlines)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(textlines)):
        ax1, ay1, ax2, ay2 = textlines[left]["bbox"]
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        for right in range(left + 1, len(textlines)):
            bx1, by1, bx2, by2 = textlines[right]["bbox"]
            intersection_w = max(0, min(ax2, bx2) - max(ax1, bx1))
            intersection_h = max(0, min(ay2, by2) - max(ay1, by1))
            intersection = intersection_w * intersection_h
            if not intersection:
                continue
            area_b = max(1, (bx2 - bx1) * (by2 - by1))
            if intersection / min(area_a, area_b) >= overlap_threshold:
                union(left, right)

    groups: dict[int, list[dict]] = {}
    for index, textline in enumerate(textlines):
        groups.setdefault(find(index), []).append(textline)

    merged: list[dict] = []
    for group in groups.values():
        if len(group) == 1:
            result = dict(group[0])
        else:
            x1 = min(item["bbox"][0] for item in group)
            y1 = min(item["bbox"][1] for item in group)
            x2 = max(item["bbox"][2] for item in group)
            y2 = max(item["bbox"][3] for item in group)
            result = dict(max(group, key=lambda item: item.get("confidence", 0.0)))
            result["bbox"] = [x1, y1, x2, y2]
            result["polygon"] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            result["confidence"] = max(
                item.get("confidence", 0.0) for item in group
            )
        result["index"] = len(merged)
        merged.append(result)

    if len(merged) != len(textlines):
        log.info(
            "Merged %s overlapping detector boxes into %s text regions",
            len(textlines),
            len(merged),
        )
    return merged


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
            
            # Force CPUExecutionProvider to prevent GPU memory collisions and OOM errors with PyTorch/MangaOCR
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


def _chunk_ranges(length: int, chunk: int, overlap: int) -> list[tuple[int, int]]:
    """Split [0, length) into overlapping chunks."""
    ranges = []
    start = 0
    while start < length:
        end = min(start + chunk, length)
        ranges.append((start, end))
        if end == length:
            break
        start = end - overlap
    return ranges


def _needs_tiled_detection(height: int, width: int) -> bool:
    """Tile only genuine strips, not ordinary high-resolution manga pages."""
    long_side, short_side = max(height, width), max(1, min(height, width))
    return long_side > 3600 or long_side > 2.2 * short_side


async def _detect_single(
    img_rgb: np.ndarray,
    text_threshold: float,
    box_threshold: float,
    merge_opencv: bool,
) -> tuple[list, np.ndarray, np.ndarray]:
    """Single near-square region detection (ONNX + optional OpenCV merge)."""
    onnx_lines, onnx_mask_raw, onnx_mask = await _detect_onnx(img_rgb, text_threshold, box_threshold)
    if not merge_opencv:
        return onnx_lines, onnx_mask_raw, onnx_mask

    cv_lines, _, _ = await _detect_opencv(img_rgb, box_threshold)

    all_lines = onnx_lines + cv_lines
    if not all_lines:
        return [], onnx_mask_raw, onnx_mask

    # NMS to suppress overlapping OpenCV boxes
    nms_boxes = []
    for tl in all_lines:
        x1, y1, x2, y2 = tl["bbox"]
        nms_boxes.append([x1, y1, x2 - x1, y2 - y1])

    # Prioritize ONNX boxes by giving them confidence=1.0, OpenCV gets 0.1
    confidences = [1.0 if i < len(onnx_lines) else 0.1 for i in range(len(all_lines))]

    nms_indices = cv2.dnn.NMSBoxes(nms_boxes, confidences, score_threshold=0.05, nms_threshold=0.3)
    final_lines = []

    if len(nms_indices) > 0:
        indices = nms_indices.flatten() if hasattr(nms_indices, "flatten") else nms_indices
        for idx in indices:
            tl = all_lines[idx]
            tl["index"] = len(final_lines)
            if idx >= len(onnx_lines):
                tl["from_opencv"] = True
            final_lines.append(tl)

    return final_lines, onnx_mask_raw, onnx_mask


async def detect(
    img_rgb: np.ndarray,
    detection_size: int = 1024,
    text_threshold: float = 0.3,
    box_threshold: float = 0.3,
    device: str = "cpu",
    merge_opencv: bool = False,
) -> tuple[list, np.ndarray, np.ndarray]:
    """Detect text regions. Uses ONNX if loaded, otherwise OpenCV contours.

    Very tall (webtoon strips) or very wide images are sliced into near-square
    chunks first: resizing an 8000px strip to 1024x1024 shrinks text below what
    CTD can see, so detection must run per chunk at native resolution.
    """
    if not _use_onnx or _ort_session is None:
        return await _detect_opencv(img_rgb, box_threshold)

    h, w = img_rgb.shape[:2]
    needs_tiling = _needs_tiled_detection(h, w)

    try:
        if not needs_tiling:
            return await _detect_single(img_rgb, text_threshold, box_threshold, merge_opencv)

        overlap = 200
        if h >= w:
            chunk = int(min(2048, max(1024, w * 2)))
            ranges = _chunk_ranges(h, chunk, overlap)
            chunks = [(y1, y2, 0, w) for y1, y2 in ranges]
        else:
            chunk = int(min(2048, max(1024, h * 2)))
            ranges = _chunk_ranges(w, chunk, overlap)
            chunks = [(0, h, x1, x2) for x1, x2 in ranges]

        log.info(f"Tiled detection: {len(chunks)} chunks of ~{chunk}px for {w}x{h} image")

        all_lines: list = []
        mask_raw = np.zeros((h, w), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)

        for y1, y2, x1, x2 in chunks:
            sub = img_rgb[y1:y2, x1:x2]
            lines, mr, m = await _detect_single(sub, text_threshold, box_threshold, merge_opencv)
            for tl in lines:
                tl["bbox"] = [tl["bbox"][0] + x1, tl["bbox"][1] + y1, tl["bbox"][2] + x1, tl["bbox"][3] + y1]
                tl["polygon"] = [[p[0] + x1, p[1] + y1] for p in tl["polygon"]]
            all_lines.extend(lines)
            mask_raw[y1:y2, x1:x2] = np.maximum(mask_raw[y1:y2, x1:x2], mr)
            mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], m)

        all_lines = _merge_overlapping_textlines(all_lines)
        for i, tl in enumerate(all_lines):
            tl["index"] = i

        return all_lines, mask_raw, mask

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
        # Scale factors back to original image
        scale_x = w / 1024.0
        scale_y = h / 1024.0
        
        # blk columns: x_center, y_center, box_w, box_h, conf, prob_class_0, prob_class_1
        # Class 0: bubble, Class 1: text
        detections = blk[0]  # shape: [64512, 7]
        
        # Vectorized filtering (replaces 64K Python loop)
        obj_conf = detections[:, 4]
        text_score = obj_conf * detections[:, 6]
        mask = (obj_conf >= box_threshold) & (text_score >= text_threshold)
        valid = detections[mask]
        valid_scores = text_score[mask]
        
        # Vectorized coordinate mapping
        cx = valid[:, 0]
        cy = valid[:, 1]
        bw = valid[:, 2]
        bh = valid[:, 3]
        
        x_min = np.clip(((cx - bw / 2.0) * scale_x).astype(int), 0, w - 1)
        y_min = np.clip(((cy - bh / 2.0) * scale_y).astype(int), 0, h - 1)
        x_max = np.clip(((cx + bw / 2.0) * scale_x).astype(int), 0, w - 1)
        y_max = np.clip(((cy + bh / 2.0) * scale_y).astype(int), 0, h - 1)
        
        bw_orig = x_max - x_min
        bh_orig = y_max - y_min
        
        # Filter boxes that are too small
        size_mask = (bw_orig > 4) & (bh_orig > 4)
        
        boxes = np.column_stack([x_min[size_mask], y_min[size_mask], bw_orig[size_mask], bh_orig[size_mask]]).tolist()
        confidences = valid_scores[size_mask].tolist()

        # Apply Non-Maximum Suppression (NMS)
        textlines = []
        mask_raw = np.zeros((h, w), dtype=np.uint8)
        
        # --- 2. Parse Text Segmentation Map (seg) ---
        # Moving segmentation parsing before NMS loop to split merged boxes
        seg_prob = seg[0, 0, :, :]  # 1024x1024
        
        # Resize to original size
        seg_prob_resized = cv2.resize(seg_prob, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Threshold to create binary mask - 0.45 captures anti-aliased edges better
        # while still being tight enough to not touch bubble outlines.
        seg_mask = (seg_prob_resized > 0.45).astype(np.uint8) * 255

        # Save YOLO boxes to check for overlaps
        yolo_boxes = boxes.copy()

        # Fallback: Find contours in the segmentation mask itself to catch text missed by YOLO (e.g. English)
        k_scale = max(1.0, max(h, w) / 1024.0)
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(15 * k_scale), int(15 * k_scale)))
        seg_closed = cv2.morphologyEx(seg_mask, cv2.MORPH_CLOSE, kernel_connect)
        seg_contours, _ = cv2.findContours(seg_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in seg_contours:
            cx, cy, cw, ch = cv2.boundingRect(c)
            # Add to boxes if reasonable size
            if cw > int(10 * k_scale) and ch > int(10 * k_scale) and (cw * ch) > int(100 * k_scale * k_scale):
                # Only add if it doesn't intersect with ANY YOLO box
                overlaps = False
                for yb in yolo_boxes:
                    yx, yy, yw, yh = yb
                    if (cx < yx + yw and cx + cw > yx and
                        cy < yy + yh and cy + ch > yy):
                        overlaps = True
                        break
                
                if not overlaps:
                    boxes.append([cx, cy, cw, ch])
                    confidences.append(0.3)  # Lower confidence than YOLO boxes (0.5+)

        if boxes:
            # cv2.dnn.NMSBoxes returns indices of the kept boxes
            nms_indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.1, nms_threshold=0.3)
            
            # NMSBoxes return type can vary by OpenCV version
            if len(nms_indices) > 0:
                indices = nms_indices.flatten() if hasattr(nms_indices, "flatten") else nms_indices
                for idx in indices:
                    x, y, bw_box, bh_box = boxes[idx]
                    
                    # Extract mask region for this box
                    region_mask = seg_mask[y:y+bh_box, x:x+bw_box]
                    
                    # 1. Dynamically estimate font size to properly scale the morphological closing kernel.
                    # This bridges gaps between lines in the SAME bubble, but keeps separate bubbles APART.
                    raw_contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    char_sizes = []
                    for c in raw_contours:
                        cx, cy, cw, ch = cv2.boundingRect(c)
                        if cw > 3 and ch > 3:  # ignore tiny noise
                            # For manga, min(w, h) robustly estimates font size whether text is connected into lines or not
                            char_sizes.append(min(cw, ch))
                            
                    if char_sizes:
                        char_size = np.median(char_sizes)
                        # A multiplier of 1.8 bridges inter-line gaps (typically 0.5-1.0x font size)
                        # but preserves inter-bubble gaps (typically 2.0-4.0x font size).
                        # This perfectly groups all lines of a sentence together without merging separate bubbles!
                        k_size = max(5, int(char_size * 1.8))
                        k_size = k_size if k_size % 2 == 1 else k_size + 1
                    else:
                        k_size = 17
                    
                    # Apply dynamic morphological closing
                    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                    closed_region = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, kernel_close)
                    
                    # Find grouped contours (bubbles) in the closed region
                    contours, _ = cv2.findContours(closed_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    valid_sub_boxes = []
                    for contour in contours:
                        cx, cy, cw, ch = cv2.boundingRect(contour)
                        # Filter out noise
                        if cw > 5 and ch > 5 and (cw * ch) > 50:
                            valid_sub_boxes.append([x + cx, y + cy, x + cx + cw, y + cy + ch])
                    
                    # Fallback to the original YOLO box if no valid contours were found
                    if not valid_sub_boxes:
                        valid_sub_boxes = [[x, y, x + bw_box, y + bh_box]]
                        
                    for sub_box in valid_sub_boxes:
                        bbox = sub_box
                        textlines.append({
                            "index": len(textlines),
                            "bbox": bbox,
                            "polygon": [
                                [bbox[0], bbox[1]],
                                [bbox[2], bbox[1]],
                                [bbox[2], bbox[3]],
                                [bbox[0], bbox[3]]
                            ],
                            "confidence": confidences[idx]
                        })
                        
                        # Use the neural network's highly accurate text mask directly.
                        # We do not use adaptive thresholding here because it incorrectly picks up 
                        # background screentones and creates solid rectangular inpainting artifacts!
                        stroke_mask = seg_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                        
                        mask_raw[bbox[1]:bbox[3], bbox[0]:bbox[2]] = cv2.bitwise_or(
                            mask_raw[bbox[1]:bbox[3], bbox[0]:bbox[2]],
                            stroke_mask
                        )

        # Gaussian blur the raw mask to soften edges before dilation.
        # This prevents hard pixel boundaries that cause visible seams after inpainting.
        mask_blurred = cv2.GaussianBlur(mask_raw, (5, 5), 0)

        # Dilate mask (5x5) to catch anti-aliased font edges that the threshold may miss.
        # Slightly larger than before to ensure full text coverage including soft edges.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask_blurred, kernel, iterations=1)

        textlines = _merge_overlapping_textlines(textlines)
        return textlines, mask_raw, mask

    return await loop.run_in_executor(None, _run)


async def _detect_opencv(img_rgb: np.ndarray, threshold: float) -> tuple[list, np.ndarray, np.ndarray]:
    """OpenCV-based text detection using contour analysis (CPU fallback)."""
    loop = asyncio.get_event_loop()

    def _run():
        h, w = img_rgb.shape[:2]
        # OpenCV is a recovery detector behind CTD. Running its expensive
        # adaptive morphology over every pixel of a 2K/4K page dominated the
        # complete request (up to 79 seconds). Detect on a bounded copy and
        # map contours back to the source image instead.
        processing_scale = min(1.0, 1600.0 / max(h, w))
        if processing_scale < 1.0:
            work_w = max(1, int(round(w * processing_scale)))
            work_h = max(1, int(round(h * processing_scale)))
            work_rgb = cv2.resize(
                img_rgb, (work_w, work_h), interpolation=cv2.INTER_AREA
            )
        else:
            work_rgb = img_rgb
        work_h, work_w = work_rgb.shape[:2]
        k_scale = max(1.0, max(work_h, work_w) / 1024.0)
        gray = cv2.cvtColor(work_rgb, cv2.COLOR_RGB2GRAY)

        block_size = int(15 * k_scale)
        if block_size % 2 == 0:
            block_size += 1

        # Adaptive threshold to find text-like regions
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 10)

        # Morphological operations to connect text components
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (int(35 * k_scale), int(max(1, 3 * k_scale))))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (int(max(1, 3 * k_scale)), int(35 * k_scale)))
        dilated_h = cv2.dilate(binary, kernel_h, iterations=2)
        dilated_v = cv2.dilate(binary, kernel_v, iterations=2)
        combined = cv2.bitwise_or(dilated_h, dilated_v)

        # Close gaps
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (int(11 * k_scale), int(11 * k_scale)))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=3)

        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        textlines = []
        mask_raw = np.zeros((h, w), dtype=np.uint8)

        for i, contour in enumerate(contours):
            x, y, bw, bh = cv2.boundingRect(contour)
            area = bw * bh

            # Filter: minimum size (scaled)
            if bw < int(10 * k_scale) or bh < int(10 * k_scale) or area < int(100 * k_scale * k_scale):
                continue

            # Filter: not too large (max 40% of image)
            if area > (work_w * work_h * 0.4):
                continue

            # Filter: aspect ratio
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            # Japanese dialogue is commonly arranged in tall vertical columns.
            # Keep those while retaining a stricter limit for horizontal panel
            # lines, which are a frequent OpenCV false positive.
            max_aspect = 20 if bh >= bw else 12
            if aspect > max_aspect:
                continue

            if processing_scale < 1.0:
                source_contour = np.rint(contour / processing_scale).astype(np.int32)
                sx, sy, sbw, sbh = cv2.boundingRect(source_contour)
            else:
                source_contour = contour
                sx, sy, sbw, sbh = x, y, bw, bh
            bbox = [sx, sy, min(w, sx + sbw), min(h, sy + sbh)]

            contour_area = cv2.contourArea(contour)
            confidence = min(contour_area / area, 1.0)

            textlines.append({
                "index": i,
                "bbox": bbox,
                "polygon": [
                    [bbox[0], bbox[1]], [bbox[2], bbox[1]],
                    [bbox[2], bbox[3]], [bbox[0], bbox[3]],
                ],
                "confidence": confidence,
            })
            cv2.drawContours(mask_raw, [source_contour], -1, 255, -1)

        # Dilate mask for inpainting coverage
        # Reduced from 5 iterations of 7x7 to 1 iteration of 5x5 to prevent massive 
        # solid blobs that destroy the original background texture during inpainting.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask_raw, kernel, iterations=1)

        return textlines, mask_raw, mask

    return await loop.run_in_executor(None, _run)


def unload():
    global _ort_session, _use_onnx
    _ort_session = None
    _use_onnx = False
    log.info("CTD ONNX model unloaded.")
