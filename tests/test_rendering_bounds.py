import unittest
from unittest.mock import patch as mock_patch

import numpy as np

from app.engine.rendering.manga2eng import (
    _largest_inner_rectangle,
    expand_bbox_to_bubble,
    find_font,
    get_font,
    prepare_render_regions,
    render_text_regions,
    wrap_text,
)


class RenderingBoundsTests(unittest.TestCase):
    def test_local_tall_oval_uses_height_instead_of_greedy_square(self):
        import cv2
        from app.engine.rendering.manga2eng import _expand_bbox_locally
        image = np.full((360, 600, 3), 110, dtype=np.uint8)
        cv2.ellipse(image, (100, 180), (66, 145), 0, 0, 360, (255, 255, 255), -1)
        box = _expand_bbox_locally(image, [75, 100, 125, 260])
        self.assertGreater(box[3] - box[1], 170)
        self.assertGreater(box[2] - box[0], 65)
        self.assertTrue(np.all(image[box[1]:box[3], box[0]:box[2]] == 255))

    def test_small_default_dialogue_uses_semibold_but_custom_font_is_preserved(self):
        image = np.full((300, 700, 3), 255, dtype=np.uint8)
        regions = [{"bbox": [50, 40, 135, 245], "translated_text": "Que la luz cure las heridas."}]
        for selected, expected in [("BarlowSemiCondensed-Medium.ttf", "BarlowSemiCondensed-SemiBold.ttf"),
                                   ("anime_ace_3.ttf", "anime_ace_3.ttf")]:
            with mock_patch("app.engine.rendering.manga2eng.get_font", wraps=get_font) as fonts:
                render_text_regions(image, regions, font_path=selected)
            self.assertTrue(fonts.call_args.args[0].endswith(expected))

    def test_dense_short_reaction_recovers_balloon_not_just_tall_columns(self):
        import cv2
        image = np.full((800, 1400, 3), 100, dtype=np.uint8)
        cv2.ellipse(image, (170, 400), (92, 230), 0, 0, 360, (255, 255, 255), -1)
        regions = [{"bbox": [155, 370, 185, 420], "translated_text": "¿El Rey Demonio?"}]
        regions.extend({"bbox": [400 + i*65, 40, 435 + i*65, 80],
                        "translated_text": "Texto"} for i in range(12))
        prepared = prepare_render_regions(image, regions)
        self.assertEqual(len(prepared), 13)
        self.assertGreater(prepared[0]["bbox"][2] - prepared[0]["bbox"][0], 85)

    def test_dense_page_recovers_width_of_vertical_dialogue(self):
        import cv2
        image = np.full((700, 1200, 3), 100, dtype=np.uint8)
        cv2.ellipse(image, (130, 340), (75, 230), 0, 0, 360, (255, 255, 255), -1)
        regions = [{"bbox": [120, 220, 140, 460], "translated_text": "Todos... gracias."}]
        regions.extend({"bbox": [350 + i*55, 40, 385 + i*55, 80],
                        "translated_text": "Hola"} for i in range(12))
        prepared = prepare_render_regions(image, regions)
        self.assertEqual(len(prepared), 13)
        self.assertGreater(prepared[0]["bbox"][2] - prepared[0]["bbox"][0], 80)
        self.assertGreaterEqual(prepared[0]["bbox"][0], 55)
        self.assertLessEqual(prepared[0]["bbox"][2], 205)

    def test_narrow_dialogue_does_not_add_hyphens(self):
        from app.engine.rendering import manga2eng
        image = np.full((420, 300, 3), 255, dtype=np.uint8)
        with mock_patch.object(manga2eng, "wrap_text", wraps=wrap_text) as wrapping:
            render_text_regions(image, [{
                "bbox": [100, 50, 152, 360],
                "translated_text": "Yo también. Todos... gracias.",
            }])
        self.assertTrue(wrapping.call_args_list)
        for call in wrapping.call_args_list:
            self.assertFalse(call.kwargs.get("break_long_words", True))

    def test_large_component_geometry_is_bounded_and_mapped_back(self):
        mask = np.zeros((1800, 1400), dtype=np.uint8)
        mask[120:1680, 180:1220] = 1

        inner = _largest_inner_rectangle(mask)

        self.assertIsNotNone(inner)
        self.assertLessEqual(abs(inner[0] - 180), 6)
        self.assertLessEqual(abs(inner[1] - 120), 6)
        self.assertLessEqual(abs(inner[2] - 1220), 6)
        self.assertLessEqual(abs(inner[3] - 1680), 6)

    def test_dense_page_uses_bounded_detector_boxes(self):
        image = np.full((1800, 1400, 3), 255, dtype=np.uint8)
        regions = [
            {
                "bbox": [80 + index * 50, 100, 115 + index * 50, 170],
                "translated_text": f"Texto {index}",
            }
            for index in range(13)
        ]

        from app.engine.rendering.manga2eng import _build_bubble_components

        def bounded_components(crop, *args, **kwargs):
            self.assertLessEqual(max(crop.shape[:2]), 384)
            return _build_bubble_components(crop, *args, **kwargs)

        with mock_patch(
            "app.engine.rendering.manga2eng._build_bubble_components",
            side_effect=bounded_components,
        ):
            prepared = prepare_render_regions(image, regions)

        self.assertEqual(len(prepared), 13)
        for source, output in zip(regions, prepared):
            self.assertLessEqual(output["bbox"][0], source["bbox"][0])
            self.assertGreaterEqual(output["bbox"][2], source["bbox"][2])
            self.assertLessEqual(output["bbox"][2] - output["bbox"][0], 91)

    def test_tiny_text_box_expands_to_closed_light_bubble(self):
        image = np.full((400, 400, 3), 120, dtype=np.uint8)
        # A closed black outline around a white speech balloon.
        import cv2
        cv2.ellipse(image, (200, 190), (62, 82), 0, 0, 360, (0, 0, 0), 7)
        cv2.ellipse(image, (200, 190), (55, 75), 0, 0, 360, (255, 255, 255), -1)

        expanded = expand_bbox_to_bubble(image, [192, 177, 208, 202])

        self.assertLess(expanded[0], 170)
        self.assertGreater(expanded[2], 230)
        self.assertLess(expanded[1], 150)
        self.assertGreater(expanded[3], 230)
        # Every corner belongs to the white interior, unlike the old outer
        # component bounding box whose corners escaped curved balloons.
        for x, y in (
            (expanded[0], expanded[1]),
            (expanded[2] - 1, expanded[1]),
            (expanded[0], expanded[3] - 1),
            (expanded[2] - 1, expanded[3] - 1),
        ):
            self.assertTrue(np.all(image[y, x] == 255))

    def test_short_translation_uses_large_balloon_not_tiny_ocr_box(self):
        image = np.full((420, 420, 3), 120, dtype=np.uint8)
        import cv2
        cv2.ellipse(image, (210, 205), (82, 135), 0, 0, 360, (0, 0, 0), 7)
        cv2.ellipse(image, (210, 205), (74, 127), 0, 0, 360, (255, 255, 255), -1)
        # This source glyph box is more than 8x smaller than the balloon,
        # matching the unreadably tiny one-word regions seen in real pages.
        source_bbox = [204, 196, 216, 214]

        expanded = expand_bbox_to_bubble(image, source_bbox)
        self.assertGreater(expanded[2] - expanded[0], 70)
        self.assertGreater(expanded[3] - expanded[1], 100)

        result = render_text_regions(
            image,
            [{"bbox": source_bbox, "translated_text": "¡Listo!"}],
            font_path="DejaVuSans.ttf",
        )
        changed = np.any(result != image, axis=2)
        ys, xs = np.where(changed)
        self.assertGreater(xs.max() - xs.min(), 45)
        self.assertGreater(ys.max() - ys.min(), 15)

    def test_prepare_regions_prefers_largest_safe_balloon_area(self):
        image = np.full((420, 420, 3), 120, dtype=np.uint8)
        import cv2
        cv2.ellipse(image, (210, 205), (82, 135), 0, 0, 360, (0, 0, 0), 7)
        cv2.ellipse(image, (210, 205), (74, 127), 0, 0, 360, (255, 255, 255), -1)

        prepared = prepare_render_regions(image, [{
            "bbox": [204, 196, 216, 214],
            "translated_text": "¡Listo!",
        }])

        self.assertEqual(len(prepared), 1)
        self.assertGreater(prepared[0]["bbox"][2] - prepared[0]["bbox"][0], 70)
        self.assertGreater(prepared[0]["bbox"][3] - prepared[0]["bbox"][1], 100)

    def test_open_page_background_is_not_used_as_a_bubble(self):
        image = np.full((220, 260, 3), 255, dtype=np.uint8)
        bbox = [122, 92, 138, 117]

        self.assertEqual(expand_bbox_to_bubble(image, bbox), bbox)

    def test_fragments_in_same_bubble_are_rendered_once(self):
        image = np.full((400, 400, 3), 120, dtype=np.uint8)
        import cv2
        cv2.ellipse(image, (200, 190), (62, 82), 0, 0, 360, (0, 0, 0), 7)
        cv2.ellipse(image, (200, 190), (55, 75), 0, 0, 360, (255, 255, 255), -1)
        regions = [
            {"bbox": [185, 157, 200, 190], "translated_text": "Primera parte"},
            {"bbox": [202, 193, 217, 225], "translated_text": "segunda parte"},
        ]

        prepared = prepare_render_regions(image, regions)

        self.assertEqual(len(prepared), 1)
        self.assertEqual(
            prepared[0]["translated_text"],
            "Primera parte segunda parte",
        )
        self.assertEqual(prepared[0]["_fragment_count"], 2)

    def test_edge_fragment_is_attached_to_expanded_sibling_balloon(self):
        image = np.full((400, 400, 3), 120, dtype=np.uint8)
        import cv2
        cv2.ellipse(image, (200, 190), (70, 95), 0, 0, 360, (0, 0, 0), 7)
        cv2.ellipse(image, (200, 190), (63, 88), 0, 0, 360, (255, 255, 255), -1)
        regions = [
            {"bbox": [187, 150, 215, 215], "translated_text": "Frase"},
            {"bbox": [151, 180, 159, 193], "translated_text": "corta"},
        ]

        prepared = prepare_render_regions(image, regions)

        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared[0]["translated_text"], "Frase corta")
        self.assertEqual(prepared[0]["_fragment_count"], 2)

    def test_neighbouring_bubbles_keep_separate_inner_rectangles(self):
        image = np.full((360, 500, 3), 90, dtype=np.uint8)
        import cv2
        cv2.ellipse(image, (180, 180), (72, 105), 0, 0, 360, (0, 0, 0), 7)
        cv2.ellipse(image, (180, 180), (64, 97), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (330, 180), (72, 105), 0, 0, 360, (0, 0, 0), 7)
        cv2.ellipse(image, (330, 180), (64, 97), 0, 0, 360, (255, 255, 255), -1)

        prepared = prepare_render_regions(image, [
            {"bbox": [170, 150, 190, 200], "translated_text": "Primero"},
            {"bbox": [320, 150, 340, 200], "translated_text": "Segundo"},
        ])

        self.assertEqual(len(prepared), 2)
        self.assertLessEqual(prepared[0]["bbox"][2], prepared[1]["bbox"][0])

    def test_long_spanish_word_is_split_to_region_width(self):
        font_path = find_font("anime_ace_3.ttf")
        self.assertIsNotNone(font_path)
        font = get_font(font_path, 18)

        lines = wrap_text("DESAFORTUNADAMENTE", font, 72)

        self.assertGreater(len(lines), 1)
        for line in lines:
            bounds = font.getbbox(line)
            self.assertLessEqual(bounds[2] - bounds[0], 72)

    def test_long_words_can_stay_intact_while_finding_font_size(self):
        font_path = find_font("DejaVuSans.ttf")
        self.assertIsNotNone(font_path)
        font = get_font(font_path, 7)

        lines = wrap_text(
            "PRIMERO GRACIAS", font, 42, break_long_words=False
        )

        self.assertEqual(lines, ["PRIMERO", "GRACIAS"])
        self.assertFalse(any("-" in line for line in lines))

    def test_rendering_cannot_paint_outside_detector_bbox(self):
        image = np.full((180, 240, 3), 255, dtype=np.uint8)
        bbox = [80, 35, 132, 155]
        result = render_text_regions(
            image,
            [{
                "bbox": bbox,
                "translated_text": (
                    "Desafortunadamente no hay habitaciones disponibles"
                ),
            }],
            font_path="anime_ace_3.ttf",
        )

        outside = np.ones(result.shape[:2], dtype=bool)
        outside[bbox[1]:bbox[3], bbox[0]:bbox[2]] = False
        self.assertTrue(np.all(result[outside] == 255))
        self.assertTrue(np.any(result[bbox[1]:bbox[3], bbox[0]:bbox[2]] < 255))

    def test_fixed_font_size_is_reduced_to_fit_region(self):
        image = np.full((220, 260, 3), 255, dtype=np.uint8)
        bbox = [75, 25, 185, 195]
        result = render_text_regions(
            image,
            [{
                "bbox": bbox,
                "translated_text": (
                    "También hay una forma de devolver ese favor sin invadir "
                    "el globo vecino"
                ),
            }],
            font_path="DejaVuSans.ttf",
            font_size=42,
        )

        outside = np.ones(result.shape[:2], dtype=bool)
        outside[bbox[1]:bbox[3], bbox[0]:bbox[2]] = False
        self.assertTrue(np.all(result[outside] == 255))
        self.assertTrue(np.any(result[bbox[1]:bbox[3], bbox[0]:bbox[2]] < 255))

    def test_clean_light_region_does_not_add_white_text_halo(self):
        image = np.full((160, 220, 3), 200, dtype=np.uint8)
        result = render_text_regions(
            image,
            [{"bbox": [50, 30, 170, 130], "translated_text": "Texto limpio"}],
            font_path="DejaVuSans.ttf",
        )

        self.assertLessEqual(int(result.max()), 200)


if __name__ == "__main__":
    unittest.main()
