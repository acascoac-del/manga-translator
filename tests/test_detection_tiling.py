import unittest

from app.engine.detection.ctd import _needs_tiled_detection


class DetectionTilingTests(unittest.TestCase):
    def test_regular_high_resolution_manga_page_is_not_tiled(self):
        self.assertFalse(_needs_tiled_detection(2508, 1763))

    def test_long_webtoon_strip_is_tiled(self):
        self.assertTrue(_needs_tiled_detection(8000, 1200))

    def test_extremely_wide_banner_is_tiled(self):
        self.assertTrue(_needs_tiled_detection(318, 1156))


if __name__ == "__main__":
    unittest.main()
