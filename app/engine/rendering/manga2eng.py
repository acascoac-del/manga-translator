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


def find_font(font_name: str = "anime_ace_3.ttf") -> Optional[str]:
    """Find font file by name, falling back to standard system fonts if not found."""
    for d in FONT_DIRS:
        path = os.path.join(d, font_name)
        if os.path.exists(path):
            return path
        # Search recursively
        for root, dirs, files in os.walk(d):
            if font_name in files:
                return os.path.join(root, font_name)
                
    # Fallback chain: prefer DejaVuSans for readability and full Spanish character coverage
    fallbacks = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for fb in fallbacks:
        if os.path.exists(fb):
            log.info(f"Requested font '{font_name}' not found. Using fallback: {fb}")
            return fb
            
    return None


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


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: float) -> list[str]:
    """Wrap text to fit within max_width, wrapping by words to prevent breaking syllables."""
    words = text.split()
    if not words:
        return [text]
        
    lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + " " + word if current_line else word
        # Calculate line width using pillow bbox
        bbox = font.getbbox(test_line)
        w = bbox[2] - bbox[0]
        
        if w > max_width:
            if current_line:
                lines.append(current_line)
                current_line = word
            else:
                # Force append single long word
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
        # Spanish inverted punctuation: AnimeAce font lacks these glyphs
        # convert them to standard punctuation so they render correctly
        "\u00A1": "!",       # ¡ -> !
        "\u00BF": "?",       # ¿ -> ?
        "¡": "!",
        "¿": "?",
    }
    
    for uni_char, ascii_char in replacements.items():
        text = text.replace(uni_char, ascii_char)
        
    text = text.replace("\n", " ").replace("\r", " ")
    
    # Whitelist printable characters: letters, numbers, basic punctuation, Spanish accents
    allowed_chars = set(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        ".,;:!?'\"()[]<>+-*/=%$@&_"
        "áéíóúüñÁÉÍÓÚÜÑ"
        " "
    )
    text = "".join(ch for ch in text if ch in allowed_chars)
    
    # Clean up double spaces
    words = text.split()
    return " ".join(words)


def render_text_regions(
    img_rgb: np.ndarray,
    textlines: list[dict],
    font_path: str = "DejaVuSans.ttf",
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
    draw = ImageDraw.Draw(pil_img)

    resolved_font_path = find_font(font_path)
    if resolved_font_path is None:
        log.warning(f"Font '{font_path}' not found, using default.")
        resolved_font_path = ""

    for tl in textlines:
        text = tl.get("translated_text", "")
        text = sanitize_text(text)
        if not text:
            continue

        bbox = tl.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        bw = x2 - x1
        bh = y2 - y1

        if bw <= 10 or bh <= 10:
            continue

        # Usable area with a smaller padding to maximize space in narrow bubbles
        pad = max(3, int(min(bw, bh) * 0.04))
        usable_w = bw - pad * 2
        usable_h = bh - pad * 2

        if usable_w <= 0 or usable_h <= 0:
            continue

        # Dynamic font color selection based on background brightness
        current_fill = font_color
        current_stroke = (255, 255, 255)
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
                else:  # Light background
                    current_fill = (0, 0, 0)
                    current_stroke = (255, 255, 255)
        except Exception as e:
            log.warning(f"Brightness check failed for bbox {bbox}: {e}")

        # --- Auto-fit font size ---
        if font_size > 0:
            fsize = font_size
            font = get_font(resolved_font_path, fsize) if resolved_font_path else ImageFont.load_default()
            lines = wrap_text(text, font, usable_w)
            line_h = int(fsize * 1.35)
            total_h = len(lines) * line_h
        else:
            # Set a more balanced size range for dialog readability: max 22px, min 11px
            max_fs = min(22, max(14, int(bh * 0.28)))
            min_fs = 11
            fsize = max_fs
            lines = [text]

            for fs in range(max_fs, min_fs - 1, -1):
                fsize = fs
                if not resolved_font_path:
                    break
                font = get_font(resolved_font_path, fsize)
                lines = wrap_text(text, font, usable_w)
                line_h = int(fsize * 1.35)
                total_h = len(lines) * line_h
                
                # Check horizontal fit for all lines
                fits_horizontally = True
                for line in lines:
                    lbbox = font.getbbox(line)
                    lw = lbbox[2] - lbbox[0]
                    if lw > usable_w:
                        fits_horizontally = False
                        break
                
                if total_h <= usable_h and fits_horizontally:
                    break  # found a size that fits both vertically and horizontally

        font = get_font(resolved_font_path, fsize) if resolved_font_path else ImageFont.load_default()
        line_h = int(fsize * 1.35)
        total_h = len(lines) * line_h

        # Vertically center within the bbox (clamped so it never exits)
        start_y = y1 + pad + max(0, (usable_h - total_h) // 2)
        # Hard clamp: bottom of last line must not exceed y2 - pad
        max_start_y = y2 - pad - total_h
        start_y = max(y1 + pad, min(start_y, max_start_y if max_start_y > y1 + pad else y1 + pad))

        stroke_w = 1 if fsize < 14 else 2

        for i, line in enumerate(lines):
            ly = start_y + i * line_h
            # Hard clip: don't draw lines that would go below bbox
            if ly + fsize > y2 - pad:
                break

            lbbox = font.getbbox(line)
            lw = lbbox[2] - lbbox[0]

            # Horizontally center, clamped within x1+pad .. x2-pad
            lx = x1 + pad + max(0, (usable_w - lw) // 2)

            if resolved_font_path:
                draw.text(
                    (lx, ly), line, font=font,
                    fill=current_fill,
                    stroke_width=stroke_w,
                    stroke_fill=current_stroke
                )
            else:
                draw.text((lx, ly), line, font=font, fill=current_fill)

    return np.array(pil_img)

def clear_font_cache():
    """Clear font cache."""
    _font_cache.clear()
