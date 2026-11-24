"""
Text Rendering - manga2eng style.
Renders translated text into speech bubbles with proper typesetting.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("rendering.manga2eng")

# Font search paths
FONT_DIRS = [
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "fonts"),
    os.path.join(os.path.dirname(__file__), "..", "manga_translator", "fonts"),
    "/usr/share/fonts",
]

_font_cache: dict[str, ImageFont.FreeTypeFont] = {}
_font_path_cache: dict[str, Optional[str]] = {}


def find_font(font_name: str = "BarlowSemiCondensed-Medium.ttf") -> Optional[str]:
    """Find font file by name, falling back to standard system fonts if not found."""
    if font_name in _font_path_cache:
        return _font_path_cache[font_name]
    
    result = None
    for d in FONT_DIRS:
        path = os.path.join(d, font_name)
        if os.path.exists(path):
            result = path
            break
        # Search recursively
        for root, dirs, files in os.walk(d):
            if font_name in files:
                result = os.path.join(root, font_name)
                break
        if result:
            break
    
    if not result:
        # Prefer regular faces for dialogue. The previous bold-first fallback,
        # combined with an outline, made ordinary balloon text look too heavy.
        fallbacks = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ]
        for fb in fallbacks:
            if os.path.exists(fb):
                log.info(f"Requested font '{font_name}' not found. Using fallback: {fb}")
                result = fb
                break
    
    _font_path_cache[font_name] = result
    return result


def get_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    """Get cached font or load it."""
    key = f"{font_path}:{size}"
    if key not in _font_cache:
        try:
            _font_cache[key] = ImageFont.truetype(font_path, size)
        except Exception:
            _font_cache[key] = ImageFont.load_default()
    return _font_cache[key]


def calculate_font_size(
    text: str,
    bbox: list[int],
    font_path: str,
    max_size: int = 28,
    min_size: int = 10,
) -> int:
    """Calculate optimal font size to fit text in bbox."""
    x1, y1, x2, y2 = bbox
    max_w = (x2 - x1) * 0.9
    max_h = (y2 - y1) * 0.9

    for size in range(max_size, min_size - 1, -1):
        font = get_font(font_path, size)
        lines = wrap_text(text, font, max_w)
        line_h = size * 1.3
        total_h = len(lines) * line_h
        if total_h <= max_h:
            return size

    return min_size


def wrap_text(
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: float,
    *,
    break_long_words: bool = True,
) -> list[str]:
    """Wrap text to fit within ``max_width`` without allowing overflow.

    Normal words stay intact.  A translated Spanish word can nevertheless be
    wider than a narrow Japanese text column (``DESAFORTUNADAMENTE`` is a
    common example).  Pillow does not clip that word, so split only this
    exceptional case and add a hyphen where it fits.
    """
    def _width(value: str) -> int:
        bounds = font.getbbox(value)
        return bounds[2] - bounds[0]

    def _split_long_word(word: str) -> list[str]:
        if _width(word) <= max_width:
            return [word]

        chunks: list[str] = []
        remaining = word
        while remaining:
            # Find the longest prefix that fits. Reserve room for a hyphen
            # whenever another chunk will follow.
            best = ""
            for length in range(1, len(remaining) + 1):
                has_more = length < len(remaining)
                candidate = remaining[:length] + ("-" if has_more else "")
                if _width(candidate) <= max_width:
                    best = candidate
                else:
                    break
            if not best:
                # Extremely narrow detector boxes may not fit even one glyph.
                # Returning one character still guarantees forward progress;
                # the per-region render surface provides the final hard clip.
                best = remaining[0]
                consumed = 1
            else:
                consumed = len(best) - (1 if best.endswith("-") else 0)
            chunks.append(best)
            remaining = remaining[consumed:]
        return chunks

    words = text.split()
    if not words:
        return [text]

    expanded_words: list[str] = []
    for word in words:
        if break_long_words:
            expanded_words.extend(_split_long_word(word))
        else:
            expanded_words.append(word)
        
    lines = []
    current_line = ""
    
    for word in expanded_words:
        test_line = current_line + " " + word if current_line else word
        w = _width(test_line)
        
        if w > max_width:
            if current_line:
                lines.append(current_line)
                current_line = word
            else:
                lines.append(word)
                current_line = ""
        else:
            current_line = test_line
            
    if current_line:
        lines.append(current_line)
        
    return lines if lines else [text]


def sanitize_text(text: str) -> str:
    """Sanitize translated text to prevent any unsupported unicode character from rendering as a tofu block."""
    if not text:
        return ""
        
    # Replacements dictionary for unicode symbols to standard ASCII
    replacements = {
        "\u3000": " ",       # Fullwidth space
        "\u2026": "...",     # Horizontal ellipsis
        "\u22ef": "...",     # Midline ellipsis
        "\u2014": "--",      # Em dash
        "\u2015": "--",      # Horizontal bar
        "\u2013": "-",       # En dash
        "\u2212": "-",       # Minus sign
        "\u201c": '"',       # Left double quote
        "\u201d": '"',       # Right double quote
        "\u2018": "'",       # Left single quote
        "\u2019": "'",       # Right single quote
        "\u3002": ". ",      # Japanese period
        "\u3001": ", ",      # Japanese comma
        "\u300c": '"',       # Japanese corner bracket
        "\u300d": '"',
        "\u300e": '"',       # Japanese white corner bracket
        "\u300f": '"',
        "\u3010": "[",       # Japanese black lenticular bracket
        "\u3011": "]",
        # Any other weird dashes or symbols commonly returned by translator:
        "\u2015": "--",
        "\u2014": "--",
        "\u2013": "-",
        "\u2212": "-",
        "\u2026": "...",
    }
    for uni_char, ascii_char in replacements.items():
        text = text.replace(uni_char, ascii_char)
        
    text = text.replace("\n", " ").replace("\r", " ")
    
    # Clean up spaces before punctuation that cause weird line wrapping
    text = text.replace(" ?", "?").replace(" !", "!").replace(" .", ".").replace(" ,", ",")
    text = text.replace("? ?", "??").replace("! !", "!!")
    
    # Whitelist printable characters: letters, numbers, basic punctuation, Spanish accents & inverted punctuation
    allowed_chars = set(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        ".,;:!?'\"()[]<>+-*/=%$@&_"
        "áéíóúüñÁÉÍÓÚÜÑ"
        "¡¿"
        " "
    )
    text = "".join(ch for ch in text if ch in allowed_chars)
    
    # Clean up double spaces
    words = text.split()
    return " ".join(words)


def _build_bubble_components(
    img_rgb: np.ndarray,
    light_threshold: int = 205,
) -> tuple[np.ndarray, np.ndarray]:
    """Label light areas once so pages with many regions stay inexpensive."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # Seal tiny outline gaps caused by antialiasing, tails and inpainting.
    # Otherwise the balloon leaks into the page background, gets rejected as
    # an enormous component and the renderer falls back to a tiny OCR box.
    dark = (gray < light_threshold).astype(np.uint8)
    dark = cv2.morphologyEx(
        dark,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    dark = cv2.dilate(
        dark,
        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
        iterations=1,
    )
    light = (dark == 0).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(light, 8)
    return labels, stats


def _largest_inner_rectangle(mask: np.ndarray) -> list[int] | None:
    """Return the largest axis-aligned rectangle completely inside ``mask``."""
    if mask.size == 0 or not np.any(mask):
        return None

    source_height, source_width = mask.shape
    # The histogram algorithm below is linear in the number of pixels but is
    # implemented in Python. A large connected page background can otherwise
    # consume a CPU core for minutes, once per OCR region. Balloon geometry
    # does not need source-pixel precision, so solve on a bounded mask and map
    # the rectangle back to the original coordinates.
    geometry_scale = min(1.0, 384.0 / max(source_height, source_width))
    if geometry_scale < 1.0:
        width = max(1, int(round(source_width * geometry_scale)))
        height = max(1, int(round(source_height * geometry_scale)))
        work_mask = cv2.resize(
            mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        work_mask = mask
        height, width = source_height, source_width
    heights = np.zeros(width, dtype=np.int32)
    best_area = 0
    best: list[int] | None = None

    for row in range(height):
        heights = np.where(work_mask[row] != 0, heights + 1, 0)
        stack: list[tuple[int, int]] = []
        for col in range(width + 1):
            current = int(heights[col]) if col < width else 0
            start = col
            while stack and stack[-1][1] > current:
                left, bar_height = stack.pop()
                area = bar_height * (col - left)
                if area > best_area:
                    best_area = area
                    best = [left, row - bar_height + 1, col, row + 1]
                start = left
            if current and (not stack or stack[-1][1] < current):
                stack.append((start, current))
    if best is None or geometry_scale >= 1.0:
        return best
    scale_x = source_width / width
    scale_y = source_height / height
    return [
        max(0, min(source_width, int(round(best[0] * scale_x)))),
        max(0, min(source_height, int(round(best[1] * scale_y)))),
        max(0, min(source_width, int(round(best[2] * scale_x)))),
        max(0, min(source_height, int(round(best[3] * scale_y)))),
    ]


def _inner_rectangle_near_point(
    mask: np.ndarray,
    point: tuple[float, float],
) -> list[int] | None:
    """Grow a safe rectangle around the OCR fragment that owns a balloon.

    Connected or overlapping speech balloons can form one light component.
    The globally largest inner rectangle then belongs to only one lobe and
    makes every fragment render there.  Keeping the rectangle anchored to the
    source text gives each lobe its own typesetting area.
    """
    if mask.size == 0 or not np.any(mask):
        return None

    height, width = mask.shape
    px = max(0, min(width - 1, int(round(point[0]))))
    py = max(0, min(height - 1, int(round(point[1]))))
    safe = mask != 0
    # Constant-time rectangle sums replace thousands of increasingly long
    # np.all() scans when a balloon is connected to a large white component.
    integral = cv2.integral(safe.astype(np.uint8), sdepth=cv2.CV_32S)

    def _all_safe(x1: int, y1: int, x2: int, y2: int) -> bool:
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area == 0:
            return False
        total = (
            int(integral[y2, x2])
            - int(integral[y1, x2])
            - int(integral[y2, x1])
            + int(integral[y1, x1])
        )
        return total == area

    if not safe[py, px]:
        ys, xs = np.nonzero(safe)
        if len(xs) == 0:
            return None
        distances = (xs - px) ** 2 + (ys - py) ** 2
        nearest = int(np.argmin(distances))
        # A far-away safe pixel almost certainly belongs to another balloon.
        max_distance = max(10, int(min(width, height) * 0.28))
        if int(distances[nearest]) > max_distance * max_distance:
            return None
        px, py = int(xs[nearest]), int(ys[nearest])

    left = right = px
    top = bottom = py
    while True:
        choices: list[tuple[int, str]] = []
        current_w = right - left + 1
        current_h = bottom - top + 1
        if left > 0 and _all_safe(left - 1, top, left, bottom + 1):
            choices.append((current_h, "left"))
        if right + 1 < width and _all_safe(right + 1, top, right + 2, bottom + 1):
            choices.append((current_h, "right"))
        if top > 0 and _all_safe(left, top - 1, right + 1, top):
            choices.append((current_w, "top"))
        if bottom + 1 < height and _all_safe(left, bottom + 1, right + 1, bottom + 2):
            choices.append((current_w, "bottom"))
        if not choices:
            break

        # Prefer the side that adds the most safe area.  Stable tie-breaking
        # keeps the rectangle balanced enough for centred manga lettering.
        _, direction = max(choices, key=lambda item: item[0])
        if direction == "left":
            left -= 1
        elif direction == "right":
            right += 1
        elif direction == "top":
            top -= 1
        else:
            bottom += 1

    return [left, top, right + 1, bottom + 1]


def expand_bbox_to_bubble(
    img_rgb: np.ndarray,
    bbox: list[int],
    component_data: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[int]:
    """Expand a detector text box to the connected light bubble interior.

    Text detectors usually return the bounds of the original glyphs, not the
    speech balloon. This is especially restrictive for one or two vertical
    Japanese characters. The inpainted image has a clean, connected interior,
    which lets us recover a larger typesetting area without crossing its dark
    outline. Implausibly large components are rejected as panel background.
    """
    image_h, image_w = img_rgb.shape[:2]
    x1 = max(0, min(image_w, int(bbox[0])))
    y1 = max(0, min(image_h, int(bbox[1])))
    x2 = max(0, min(image_w, int(bbox[2])))
    y2 = max(0, min(image_h, int(bbox[3])))
    original = [x1, y1, x2, y2]
    if x2 - x1 <= 4 or y2 - y1 <= 4:
        return original

    labels, stats = component_data or _build_bubble_components(img_rgb)
    count = len(stats)
    if count <= 1:
        return original

    original_area = max(1, (x2 - x1) * (y2 - y1))
    image_area = image_w * image_h
    pad = max(8, int(max(x2 - x1, y2 - y1) * 0.45))
    sx1, sy1 = max(0, x1 - pad), max(0, y1 - pad)
    sx2, sy2 = min(image_w, x2 + pad), min(image_h, y2 + pad)

    # Residual antialiasing or imperfect inpainting can leave the exact OCR
    # rectangle dark. Looking just around it recovers the surrounding balloon
    # instead of forcing a one-word translation into a glyph-sized box.
    overlap = labels[y1:y2, x1:x2]
    nearby = labels[sy1:sy2, sx1:sx2]
    overlap_counts = np.bincount(overlap.ravel(), minlength=count)
    nearby_counts = np.bincount(nearby.ravel(), minlength=count)
    overlap_counts[0] = 0
    nearby_counts[0] = 0
    candidate_labels = np.flatnonzero(nearby_counts)
    candidate_labels = sorted(
        (int(label) for label in candidate_labels),
        key=lambda label: (int(overlap_counts[label]), int(nearby_counts[label])),
        reverse=True,
    )

    best: tuple[float, list[int]] | None = None
    original_cx = (x1 + x2) / 2
    original_cy = (y1 + y2) / 2
    for label in candidate_labels[:8]:
        cx, cy, cw, ch, area = [int(v) for v in stats[label]]
        component_box_area = cw * ch

        # Reject open page/panel backgrounds and very sparse components. The
        # density floor is intentionally tolerant of tails and irregular manga
        # balloons, while the absolute limits still exclude the white page.
        if (
            area > image_area * 0.28
            or cw > image_w * 0.65
            or ch > image_h * 0.78
            or component_box_area > image_area * 0.32
            or area / max(1, component_box_area) < 0.40
            or component_box_area < original_area * 1.20
            or cw > max(x2 - x1, 20) * 18
            or ch > max(y2 - y1, 20) * 18
        ):
            continue

        component = (labels[cy:cy + ch, cx:cx + cw] == label).astype(np.uint8)
        inset = max(2, int(min(cw, ch) * 0.035))
        kernel_size = inset * 2 + 1
        safe = cv2.erode(
            component,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            ),
            iterations=1,
        )
        inner = _largest_inner_rectangle(safe)
        if inner is None:
            continue
        source_point = (original_cx - cx, original_cy - cy)
        global_contains_source = (
            inner[0] <= source_point[0] <= inner[2]
            and inner[1] <= source_point[1] <= inner[3]
        )
        if not global_contains_source:
            local_inner = _inner_rectangle_near_point(safe, source_point)
            if local_inner is not None:
                inner = local_inner
        expanded = [cx + inner[0], cy + inner[1], cx + inner[2], cy + inner[3]]
        expanded_w = expanded[2] - expanded[0]
        expanded_h = expanded[3] - expanded[1]
        expanded_area = expanded_w * expanded_h
        if expanded_w <= 10 or expanded_h <= 10 or expanded_area < original_area * 1.15:
            continue

        # A nearby component must either contain the source centre or overlap
        # the source box. This prevents two close balloons from being merged.
        contains_center = (
            expanded[0] <= original_cx <= expanded[2]
            and expanded[1] <= original_cy <= expanded[3]
        )
        if not contains_center and overlap_counts[label] == 0:
            continue
        score = (
            float(overlap_counts[label]) * 1000.0
            + (500.0 if contains_center else 0.0)
            + min(float(expanded_area) / original_area, 200.0)
        )
        if best is None or score > best[0]:
            best = (score, expanded)

    return best[1] if best is not None else original


def _expand_bbox_locally(
    img_rgb: np.ndarray,
    bbox: list[int],
    light_threshold: int = 225,
) -> list[int]:
    """Recover a safe balloon interior around one OCR fragment.

    A tiny outline gap can connect a balloon to the white page and make the
    global component look implausibly large. A bounded local pass keeps the
    search attached to this fragment and rejects page-like areas that escape
    through most sides of the crop.
    """
    image_h, image_w = img_rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, x2 = max(0, x1), min(image_w, x2)
    y1, y2 = max(0, y1), min(image_h, y2)
    original = [x1, y1, x2, y2]
    box_w, box_h = x2 - x1, y2 - y1
    if box_w <= 4 or box_h <= 4:
        return original

    pad = max(72, int(max(box_w, box_h) * 3.5))
    pad = min(pad, max(96, int(max(image_w, image_h) * 0.28)))
    rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
    rx2, ry2 = min(image_w, x2 + pad), min(image_h, y2 + pad)
    roi = img_rgb[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return original

    # Bound component work per fragment, including dense pages. Coordinates
    # are restored after finding the safe interior on the reduced crop.
    scale = min(1.0, 384.0 / max(roi.shape[:2]))
    if scale < 1.0:
        small = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        small_box = [round((x1-rx1)*scale), round((y1-ry1)*scale),
                     round((x2-rx1)*scale), round((y2-ry1)*scale)]
        recovered = _expand_bbox_locally(small, small_box, light_threshold)
        if recovered == small_box:
            return original
        return [rx1 + round(recovered[0]/scale), ry1 + round(recovered[1]/scale),
                rx1 + round(recovered[2]/scale), ry1 + round(recovered[3]/scale)]

    labels, stats = _build_bubble_components(roi, light_threshold)
    local_box = [x1 - rx1, y1 - ry1, x2 - rx1, y2 - ry1]
    lx1, ly1, lx2, ly2 = local_box
    overlap = labels[ly1:ly2, lx1:lx2]
    counts = np.bincount(overlap.ravel(), minlength=len(stats))
    counts[0] = 0
    # OCR boxes may become completely white after inpainting or may cover
    # only a thin glyph.  Include components immediately around the box so a
    # valid balloon is still recovered instead of typesetting into the tiny
    # OCR rectangle.
    near_pad = max(10, int(max(box_w, box_h) * 0.65))
    nx1, ny1 = max(0, lx1 - near_pad), max(0, ly1 - near_pad)
    nx2 = min(labels.shape[1], lx2 + near_pad)
    ny2 = min(labels.shape[0], ly2 + near_pad)
    near_counts = np.bincount(
        labels[ny1:ny2, nx1:nx2].ravel(), minlength=len(stats)
    )
    near_counts[0] = 0
    local_cx = (lx1 + lx2) / 2
    local_cy = (ly1 + ly2) / 2
    original_area = max(1, box_w * box_h)

    candidate_labels = sorted(
        (
            int(label)
            for label in np.flatnonzero((counts > 0) | (near_counts > 0))
        ),
        key=lambda label: int(counts[label]) * 100 + int(near_counts[label]),
        reverse=True,
    )
    for label in candidate_labels[:6]:
        cx, cy, cw, ch, area = [int(v) for v in stats[label]]
        if cw * ch < original_area * 1.2 or area / max(1, cw * ch) < 0.38:
            continue
        touches = sum((cx <= 1, cy <= 1, cx + cw >= roi.shape[1] - 1,
                       cy + ch >= roi.shape[0] - 1))
        if touches >= 3:
            continue

        component = (labels[cy:cy + ch, cx:cx + cw] == label).astype(np.uint8)
        inset = max(2, int(min(cw, ch) * 0.035))
        safe = cv2.erode(
            component,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (inset * 2 + 1, inset * 2 + 1)
            ),
            iterations=1,
        )
        source_point = (local_cx - cx, local_cy - cy)
        # Greedy growth favours width and can stop at a short rectangle in a
        # tall oval. Prefer the largest safe interior when it owns this text;
        # retain anchored growth for connected neighbouring balloons.
        inner = _largest_inner_rectangle(safe)
        if inner is None or not (
            inner[0] <= source_point[0] < inner[2]
            and inner[1] <= source_point[1] < inner[3]
        ):
            inner = _inner_rectangle_near_point(safe, source_point)
        if inner is None:
            continue
        expanded = [
            rx1 + cx + inner[0],
            ry1 + cy + inner[1],
            rx1 + cx + inner[2],
            ry1 + cy + inner[3],
        ]
        expanded_area = max(0, expanded[2] - expanded[0]) * max(
            0, expanded[3] - expanded[1]
        )
        if expanded_area >= original_area * 1.15:
            return expanded
    return original


def prepare_render_regions(
    img_rgb: np.ndarray,
    textlines: list[dict],
) -> list[dict]:
    """Group OCR fragments that resolve to the same speech balloon.

    CTD may return several vertical columns/lines for one balloon. Expanding
    every fragment to the balloon and rendering each independently creates
    oversized overlapping paragraphs. This pass keeps their existing order,
    joins their translations, and typesets the balloon exactly once.
    """
    candidates: list[dict] = []
    # A few dense pages contain large connected white artwork/backgrounds.
    # Re-evaluating their component geometry for every OCR fragment can pin a
    # CPU core indefinitely. Detector boxes are already safe render bounds, so
    # use a small bounded expansion for dense pages instead of bubble tracing.
    dense_page = len(textlines) >= 13
    if dense_page:
        log.info(
            "Dense page (%s regions): using bounded detector-box rendering",
            len(textlines),
        )
    # Dense pages never consume component maps below, so do not build them.
    # On pages containing large connected white artwork, connected-components
    # analysis itself can pin a CPU core long enough for the browser request to
    # time out even though translation already finished.
    if dense_page:
        strict_component_data = None
        permissive_component_data = None
        page_gray = None
    else:
        # A stricter pass only accepts very bright inpainting residue. The old
        # lower threshold admitted gray artwork/background into a balloon and
        # was the main source of paragraphs rendered across characters/panels.
        strict_component_data = _build_bubble_components(
            img_rgb, light_threshold=225
        )
        permissive_component_data = _build_bubble_components(img_rgb)
        page_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    for textline in textlines:
        text = sanitize_text(textline.get("translated_text", ""))
        if not text:
            continue

        original_bbox = [int(v) for v in textline.get("bbox", [0, 0, 0, 0])]
        image_h, image_w = img_rgb.shape[:2]
        if dense_page:
            box_w = max(1, original_bbox[2] - original_bbox[0])
            box_h = max(1, original_bbox[3] - original_bbox[1])
            pad_x = min(28, max(5, int(box_w * 0.10)))
            pad_y = min(28, max(5, int(box_h * 0.10)))
            bounded = [
                max(0, original_bbox[0] - pad_x),
                max(0, original_bbox[1] - pad_y),
                min(image_w, original_bbox[2] + pad_x),
                min(image_h, original_bbox[3] + pad_y),
            ]
            possible_bboxes = [original_bbox, bounded]
            # Vertical Japanese columns need the balloon's width for Spanish.
            # Recover only narrow columns locally, preserving the dense-page
            # shortcut for normal boxes and avoiding full-page component maps.
            if box_h > box_w * 1.5 or len(text) <= 48:
                recovered = _expand_bbox_locally(img_rgb, original_bbox)
                if recovered != original_bbox:
                    possible_bboxes[1] = recovered
        else:
            possible_bboxes = [
                original_bbox,
                expand_bbox_to_bubble(
                    img_rgb, original_bbox, strict_component_data
                ),
                _expand_bbox_locally(img_rgb, original_bbox),
                expand_bbox_to_bubble(
                    img_rgb, original_bbox, permissive_component_data
                ),
            ]
        ocx = (original_bbox[0] + original_bbox[2]) / 2
        ocy = (original_bbox[1] + original_bbox[3]) / 2

        def _safe_balloon(candidate: list[int]) -> bool:
            x1, y1, x2, y2 = candidate
            width, height = x2 - x1, y2 - y1
            if width <= 0 or height <= 0:
                return False
            if not (x1 <= ocx <= x2 and y1 <= ocy <= y2):
                return False
            # Reject page/panel-sized white components. A genuine dialogue
            # balloon may be tall, but it should not consume most of a page.
            geometry_ok = (
                width <= image_w * 0.52
                and height <= image_h * 0.62
                and width * height <= image_w * image_h * 0.22
            )
            if not geometry_ok:
                return False

            # When the OCR source is on a light balloon, an expanded box must
            # remain predominantly light.  This rejects rectangles that cross
            # a panel border, a face or dark artwork even though their overall
            # dimensions look plausible.
            ox1, oy1, ox2, oy2 = original_bbox
            original_crop = page_gray[
                max(0, oy1):min(image_h, oy2),
                max(0, ox1):min(image_w, ox2),
            ]
            candidate_crop = page_gray[y1:y2, x1:x2]
            if original_crop.size and candidate_crop.size:
                original_light = float(np.mean(original_crop >= 205))
                candidate_light = float(np.mean(candidate_crop >= 205))
                if original_light >= 0.58 and candidate != original_bbox:
                    if candidate_light < max(0.64, original_light - 0.22):
                        return False
            return True

        # Prefer strict/local recovery.  The permissive component is only a
        # last resort because gray artwork can otherwise be mistaken for part
        # of a balloon and produce text outside its outline.
        if dense_page:
            preferred = [possible_bboxes[1]]
            fallback = []
        else:
            preferred = [
                box for box in possible_bboxes[:3]
                if box != original_bbox and _safe_balloon(box)
            ]
            permissive = possible_bboxes[3]
            fallback = (
                [permissive]
                if permissive != original_bbox and _safe_balloon(permissive)
                else []
            )
        render_bbox = max(
            preferred or fallback or [original_bbox],
            key=lambda box: max(0, box[2] - box[0]) * max(0, box[3] - box[1]),
        )
        was_expanded = render_bbox != original_bbox

        candidates.append({
            "bbox": render_bbox,
            "_original_bbox": original_bbox,
            "translated_text": text,
            "_expanded": was_expanded,
            "_fragment_count": 1,
        })

    prepared: list[dict] = []
    for item in candidates:
        matching = None
        for existing in prepared:
            same_expanded_box = (
                item["_expanded"]
                and existing.get("_expanded")
                and sum(
                    abs(a - b)
                    for a, b in zip(item["bbox"], existing["bbox"])
                ) <= 12
            )

            # OCR frequently leaves a tiny column close to the balloon edge.
            # That fragment may fail expansion even though its centre is still
            # inside the safely recovered balloon of a sibling.  Attach it to
            # that sibling instead of drawing a second microscopic paragraph.
            def _center_inside(inner: list[int], outer: list[int]) -> bool:
                cx = (inner[0] + inner[2]) / 2
                cy = (inner[1] + inner[3]) / 2
                return (
                    outer[0] - 2 <= cx <= outer[2] + 2
                    and outer[1] - 2 <= cy <= outer[3] + 2
                )

            edge_fragment_of_existing = (
                existing.get("_expanded")
                and _center_inside(item["_original_bbox"], existing["bbox"])
            )
            existing_fragment_of_item = (
                item["_expanded"]
                and _center_inside(existing["_original_bbox"], item["bbox"])
            )
            if (
                same_expanded_box
                or edge_fragment_of_existing
                or existing_fragment_of_item
            ):
                matching = existing
                break
        if matching is None:
            prepared.append(item)
            continue
        matching["translated_text"] = (
            matching["translated_text"].rstrip()
            + " "
            + item["translated_text"].lstrip()
        )
        matching["_fragment_count"] += item["_fragment_count"]

    for item in prepared:
        item.pop("_original_bbox", None)
    return prepared


def render_text_regions(
    img_rgb: np.ndarray,
    textlines: list[dict],
    font_path: str = "BarlowSemiCondensed-Medium.ttf",
    font_size: int = 0,
    font_color: tuple = (0, 0, 0),
    text_shadow: bool = False,
) -> np.ndarray:
    """
    Render translated text onto the image, guaranteeing text stays inside the bounding box.
    Uses dynamic font sizing, word wrapping and contrast-aware coloring.
    """
    # Convert to PIL for text rendering
    pil_img = Image.fromarray(img_rgb)

    resolved_font_path = find_font(font_path)
    if resolved_font_path is None:
        log.warning(f"Font '{font_path}' not found, using default.")
        resolved_font_path = ""

    requested_font_path = resolved_font_path
    for tl in prepare_render_regions(img_rgb, textlines):
        resolved_font_path = requested_font_path
        text = tl["translated_text"]
        bbox = tl["bbox"]
        image_w, image_h = pil_img.size
        x1 = max(0, min(image_w, int(bbox[0])))
        y1 = max(0, min(image_h, int(bbox[1])))
        x2 = max(0, min(image_w, int(bbox[2])))
        y2 = max(0, min(image_h, int(bbox[3])))
        bw = x2 - x1
        bh = y2 - y1

        if bw <= 10 or bh <= 10:
            continue

        # Leave breathing room between dialogue and the balloon outline.
        pad = max(3, int(min(bw, bh) * 0.04))
        usable_w = bw - pad * 2
        usable_h = bh - pad * 2

        if usable_w <= 0 or usable_h <= 0:
            continue

        # Dynamic font color selection based on background brightness
        current_fill = font_color
        current_stroke = (255, 255, 255)
        needs_outline = text_shadow
        try:
            h_img, w_img = img_rgb.shape[:2]
            rx1 = max(0, min(w_img - 1, x1))
            ry1 = max(0, min(h_img - 1, y1))
            rx2 = max(0, min(w_img - 1, x2))
            ry2 = max(0, min(h_img - 1, y2))
            if rx2 > rx1 and ry2 > ry1:
                crop = img_rgb[ry1:ry2, rx1:rx2]
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                avg_brightness = np.mean(gray_crop)
                if avg_brightness < 100:  # Dark background
                    current_fill = (255, 255, 255)
                    current_stroke = (0, 0, 0)
                    needs_outline = True
                else:  # Light background
                    current_fill = (0, 0, 0)
                    current_stroke = (255, 255, 255)
                    # Clean inpainted balloons need no halo. Retain a subtle
                    # outline only over visibly textured narration/artwork.
                    needs_outline = needs_outline or float(np.std(gray_crop)) > 45
        except Exception as e:
            log.warning(f"Brightness check failed for bbox {bbox}: {e}")

        # Render on a bbox-sized surface. Pillow otherwise lets glyphs extend
        # beyond a detector box, which placed long Spanish words over manga
        # characters and neighboring panels. The local surface is a final
        # hard clipping boundary even if a font has unusual bearings.
        region_layer = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        region_draw = ImageDraw.Draw(region_layer)

        # --- Auto-fit font size ---
        # A configured size is an upper bound, not permission to overflow.
        # Always auto-fit it to the actual balloon. This also prevents two
        # adjacent balloons from painting over one another.
        word_count = max(1, len(text.split()))
        # Match dialogue size to the page's displayed width. The old 17% of
        # balloon width ceiling made short phrases tiny in tall balloons even
        # when their entire text could fit comfortably at a larger size.
        page_dialogue_size = max(26, min(64, round(image_w * 0.032)))
        short_text_cap = round(page_dialogue_size * (1.10 if word_count <= 4 else 1.0))
        automatic_max = min(
            short_text_cap,
            max(24, int(min(bw, bh) * 0.50)),
        )
        max_fs = min(font_size, 48) if font_size > 0 else automatic_max
        # Seven-pixel glyphs were technically inside the box but unreadable.
        # Keep a practical floor; recovering the balloon above supplies the
        # extra area needed for short dialogue such as "¡Listo!".
        visible_chars = len("".join(text.split()))
        min_dim = min(bw, bh)
        page_span = max(image_w, image_h)
        if visible_chars <= 12:
            resolution_floor = min(28, max(16, int(page_span * 0.0105)))
            min_fs = max(resolution_floor, min(26, int(min_dim * 0.18)))
        elif visible_chars <= 32:
            resolution_floor = min(22, max(14, int(page_span * 0.008)))
            min_fs = max(resolution_floor, min(20, int(min_dim * 0.14)))
        else:
            min_fs = max(11, min(15, int(min_dim * 0.12)))
        # An explicit size remains a ceiling, including in narrow balloons.
        if font_size > 0:
            min_fs = min(min_fs, max_fs)
        max_fs = max(min_fs, max_fs)
        # Small/narrow default dialogue needs stronger strokes, not a halo.
        # This face has the same condensed proportions but survives downscaling
        # better. Preserve explicitly selected custom fonts.
        if (os.path.basename(font_path) == "BarlowSemiCondensed-Medium.ttf"
                and (bh > bw * 1.4 or bw < image_w * 0.14 or max_fs <= 24)):
            readable_face = find_font("BarlowSemiCondensed-SemiBold.ttf")
            if readable_face:
                resolved_font_path = readable_face
        fsize = max_fs
        lines = [text]

        if resolved_font_path:
            # Spanish contains substantially wider words than CJK text. Let a
            # line be condensed moderately before shrinking every glyph to an
            # unreadable size.
            max_line_condense = 1.10
            layout_w = max(usable_w, int(usable_w * max_line_condense))

            def _fits(fs: int) -> bool:
                candidate_font = get_font(resolved_font_path, fs)
                candidate_lines = wrap_text(
                    text, candidate_font, layout_w, break_long_words=False
                )
                candidate_line_h = max(fs + 2, int(fs * 1.08))
                if len(candidate_lines) * candidate_line_h > usable_h:
                    return False
                return all(
                    candidate_font.getbbox(line)[2]
                    - candidate_font.getbbox(line)[0]
                    <= layout_w * (1.20 if len(line.split()) == 1 else 1.0)
                    for line in candidate_lines
                )

            lo, hi = min_fs, max_fs
            best_fs: int | None = None
            while lo <= hi:
                mid = (lo + hi) // 2
                if _fits(mid):
                    best_fs = mid
                    lo = mid + 1
                else:
                    hi = mid - 1

            # If no intact-word layout fits, use the minimum readable size and
            # wrap exceptional long words instead of shrinking all glyphs.
            # Prefer a slightly smaller intact word to a stack of arbitrary
            # syllables. Width-only fallback below handles truly narrow boxes.
            if best_fs is None:
                for fs in range(min_fs - 1, max(10, int(min_fs * 0.70)) - 1, -1):
                    if _fits(fs):
                        best_fs = fs
                        break
            fsize = best_fs if best_fs is not None else min_fs
            font = get_font(resolved_font_path, fsize)
            lines = wrap_text(
                text,
                font,
                layout_w,
                break_long_words=False,
            )
            line_h = max(fsize + 2, int(fsize * 1.08))
            total_h = len(lines) * line_h
        else:
            font = ImageFont.load_default()
            lines = wrap_text(text, font, usable_w)
            fsize = min_fs
            line_h = max(fsize + 2, int(fsize * 1.08))
            total_h = len(lines) * line_h

        font = get_font(resolved_font_path, fsize) if resolved_font_path else ImageFont.load_default()
        line_h = max(fsize + 2, int(fsize * 1.08))
        total_h = len(lines) * line_h

        # At the readable floor, widen the layout before sacrificing vertical
        # glyph size. Spanish lines can be condensed moderately and remain far
        # more legible than a paragraph scaled down in both dimensions.
        if resolved_font_path and total_h > usable_h:
            wide_layout = max(usable_w, int(usable_w * 1.25))
            lines = wrap_text(text, font, wide_layout, break_long_words=False)
            total_h = len(lines) * line_h

        # A malformed OCR response can contain far more prose than a balloon
        # can hold. Preserve a readable floor and clip only this pathological
        # tail with an ellipsis; never shrink the complete paragraph into
        # microscopic text.
        max_lines = max(1, usable_h // line_h)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            last = lines[-1].rstrip(" .…") + "…"
            lines[-1] = last
            total_h = len(lines) * line_h

        stroke_w = (1 if fsize < 18 else 2) if needs_outline else 0
        block_h = max(1, total_h)
        text_block = Image.new("RGBA", (usable_w, block_h), (0, 0, 0, 0))
        block_draw = ImageDraw.Draw(text_block)

        for i, line in enumerate(lines):
            ly = i * line_h

            lbbox = font.getbbox(line)
            lw = lbbox[2] - lbbox[0]
            glyph_y = max(0, (line_h - (lbbox[3] - lbbox[1])) // 2) - lbbox[1]

            # Horizontally center, clamped within x1+pad .. x2-pad
            lx = max(0, (usable_w - lw) // 2)

            if resolved_font_path and lw > usable_w:
                # Condense one line only. This supports ordinary multi-word
                # Spanish lines too, reducing line count without shrinking the
                # paragraph vertically.
                margin = stroke_w + 2
                natural_w = max(1, lw + margin * 2)
                natural_h = line_h
                line_layer = Image.new(
                    "RGBA", (natural_w, natural_h), (0, 0, 0, 0)
                )
                line_draw = ImageDraw.Draw(line_layer)
                line_draw.text(
                    (margin - lbbox[0], glyph_y), line, font=font,
                    fill=current_fill,
                    stroke_width=stroke_w,
                    stroke_fill=current_stroke,
                )
                target_w = max(1, usable_w)
                line_layer = line_layer.resize(
                    (target_w, min(line_h, natural_h)), Image.Resampling.LANCZOS
                )
                text_block.alpha_composite(line_layer, (0, ly))
            elif resolved_font_path:
                block_draw.text(
                    (lx - lbbox[0], ly + glyph_y), line, font=font,
                    fill=current_fill,
                    stroke_width=stroke_w,
                    stroke_fill=current_stroke
                )
            else:
                block_draw.text((lx, ly), line, font=font, fill=current_fill)

        block_y = pad + max(0, (usable_h - text_block.height) // 2)
        region_layer.alpha_composite(text_block, (pad, block_y))

        # Composite coverage once. Masked paste onto a transparent intermediate
        # would apply the antialiasing alpha twice and thin small glyph edges.
        pil_img.paste(Image.alpha_composite(
            pil_img.crop((x1, y1, x2, y2)).convert("RGBA"), region_layer
        ).convert(pil_img.mode), (x1, y1))

    return np.array(pil_img)

def clear_font_cache():
    """Clear font cache."""
    _font_cache.clear()
