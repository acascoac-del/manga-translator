import io
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
from fastapi import UploadFile

from app import main
from app.engine.ocr import dispatcher


class SourceLanguageTests(unittest.IsolatedAsyncioTestCase):
    async def test_json_and_batch_preserve_explicit_language_and_default(self):
        result = SimpleNamespace(success=True, processing_time_ms=0,
                                 translated_b64="", regions=[])
        for endpoint in (main.translate_json, main.translate_batch):
            for language in ("ja", "auto", None):
                body = {"image": "eA==", "images": ["eA=="]}
                if language is not None:
                    body["source_lang"] = language
                request = SimpleNamespace(json=AsyncMock(return_value=body))
                with patch.object(main, "_run_pipeline_serialized", AsyncMock(return_value=result)) as run:
                    with patch.object(main, "_format_result", return_value={}):
                        await endpoint(request)
                    self.assertEqual(run.call_args.args[1].source_lang, language or "auto")

    async def test_multipart_preserves_japanese(self):
        with patch.object(main, "_run_pipeline_serialized", AsyncMock()) as run:
            with patch.object(main, "_format_result", return_value={}):
                await main.translate_multipart(
                    image=UploadFile(filename="page.png", file=io.BytesIO(b"image")),
                    source_lang="ja",
                )
            self.assertEqual(run.call_args.args[1].source_lang, "ja")

    async def test_japanese_skips_all_easyocr_readers_even_on_color_pages(self):
        regions = [{"text": "砂巨人"}]
        for color in ((255, 255, 255), (255, 0, 0)):
            image = np.full((32, 32, 3), color, dtype=np.uint8)
            with patch.object(dispatcher.manga_ocr, "load_model", AsyncMock()):
                with patch.object(dispatcher.manga_ocr, "recognize", AsyncMock(return_value=regions)) as manga:
                    with patch.object(dispatcher.easy_ocr, "recognize", AsyncMock()) as other:
                        actual = await dispatcher.recognize_regions(image, regions, "ja", "cpu")
                        self.assertEqual(actual, regions)
                        manga.assert_awaited_once()
                        other.assert_not_awaited()
