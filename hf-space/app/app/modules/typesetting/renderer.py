"""
Typesetting Module - Dynamic text rendering on manga bubbles.
Supports: horizontal and vertical text, auto font size, text wrapping.
"""

from typing import Optional
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.core.config import Settings


class Renderer:
    """
    Dynamic text renderer for manga bubbles.
    Handles vertical/horizontal text, auto-sizing, and wrapping.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.font_path: Optional[str] = None
        self.default_font_size = settings.FONT_SIZE_MAX

    async def load(self):
        """Load fonts."""
        font_path = Path(self.settings.DEFAULT_FONT)
        if font_path.exists():
            self.font_path = str(font_path)
            print(f"[Renderer] Using font: {font_path}")
        else:
            # Try to find any TTF font
            import glob
            fonts = glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
            if fonts:
                self.font_path = fonts[0]
                print(f"[Renderer] Using system font: {self.font_path}")
            else:
                print("[Renderer] No font found, using default")

    def render_text(
        self,
        canvas: Image.Image,
        text: str,
        bbox: dict,
        orientation: str = "horizontal",
        font_color: str = "#000000",
        text_shadow: bool = False,
    ):
        """
        Render translated text onto the canvas at the specified bbox.
        """
        if not text or not text.strip():
            return

        draw = ImageDraw.Draw(canvas)

        # Calculate optimal font size
        font_size = self._calc_font_size(draw, text, bbox)

        # Load font
        try:
            if self.font_path:
                font = ImageFont.truetype(self.font_path, font_size)
            else:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        # Wrap text to fit bbox
        lines = self._wrap_text(draw, text, bbox, font)

        # Calculate line height
        line_height = int(font_size * 1.3)
        total_height = len(lines) * line_height

        # Center vertically
        start_y = bbox["y"] + (bbox["height"] - total_height) // 2

        if orientation == "vertical":
            self._render_vertical(draw, lines, bbox, font, font_size, line_height, font_color)
        else:
            self._render_horizontal(draw, lines, bbox, font, font_size, line_height, font_color, text_shadow)

    def _render_horizontal(
        self, draw, lines, bbox, font, font_size, line_height, font_color, text_shadow
    ):
        """Render text horizontally centered in bbox."""
        start_y = bbox["y"] + (bbox["height"] - len(lines) * line_height) // 2

        for i, line in enumerate(lines):
            bbox_text = draw.textbbox((0, 0), line, font=font)
            text_width = bbox_text[2] - bbox_text[0]
            x = bbox["x"] + (bbox["width"] - text_width) // 2
            y = start_y + i * line_height

            if text_shadow:
                # Draw shadow
                draw.text((x + 1, y + 1), line, font=font, fill="#ffffff")
                draw.text((x - 1, y - 1), line, font=font, fill="#ffffff")

            draw.text((x, y), line, font=font, fill=font_color)

    def _render_vertical(self, draw, lines, bbox, font, font_size, line_height, font_color):
        """Render text vertically (right-to-left for Japanese)."""
        total_width = len(lines) * line_height
        start_x = bbox["x"] + (bbox["width"] + total_width) // 2 - line_height // 2

        for i, line in enumerate(lines):
            chars = list(line)
            x = start_x - i * line_height
            start_y = bbox["y"] + (bbox["height"] - len(chars) * font_size) // 2

            for j, char in enumerate(chars):
                y = start_y + j * font_size
                draw.text((x - font_size // 2, y), char, font=font, fill=font_color)

    def _calc_font_size(self, draw, text: str, bbox: dict) -> int:
        """Calculate optimal font size to fit text in bbox."""
        max_w = bbox["width"] * 0.9
        max_h = bbox["height"] * 0.9

        font_size = self.default_font_size
        min_size = self.settings.FONT_SIZE_MIN
        max_size = self.settings.FONT_SIZE_MAX

        while font_size > min_size:
            try:
                if self.font_path:
                    font = ImageFont.truetype(self.font_path, font_size)
                else:
                    font = ImageFont.load_default()

                lines = self._wrap_text(draw, text, bbox, font)
                line_height = font_size * 1.3
                total_h = len(lines) * line_height

                if total_h <= max_h:
                    # Check if any line is too wide
                    all_fit = all(
                        draw.textbbox((0, 0), l, font=font)[2] - draw.textbbox((0, 0), l, font=font)[0] <= max_w
                        for l in lines
                    )
                    if all_fit:
                        return font_size

            except Exception:
                pass

            font_size -= 1

        return max(min_size, font_size)

    def _wrap_text(self, draw, text: str, bbox: dict, font) -> list[str]:
        """Wrap text to fit within bbox width."""
        max_w = bbox["width"] * 0.9
        lines = []
        current = ""

        for char in text:
            test = current + char
            try:
                bbox_text = draw.textbbox((0, 0), test, font=font)
                width = bbox_text[2] - bbox_text[0]
            except Exception:
                width = len(test) * 10

            if width > max_w and current:
                lines.append(current)
                current = char
            else:
                current = test

        if current:
            lines.append(current)

        return lines if lines else [text]
