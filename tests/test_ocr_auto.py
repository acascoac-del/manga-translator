import unittest

from app.engine.ocr.dispatcher import (
    _choose_auto_language,
    _english_overrides_japanese,
    _english_page_is_confident,
    _english_page_is_plausible,
    _image_has_meaningful_color,
    _japanese_fast_path_is_confident,
    _japanese_page_is_confident,
    _korean_page_is_confident,
    _chinese_page_is_confident,
    _script_evidence,
)
from app.engine.ocr.easy_ocr import (
    _assign_english_lines,
    _english_candidate_quality,
    _english_needs_crop_retry,
    _select_english_crop_retries,
    _english_text_score,
    _normalize_english_ocr,
    _recover_unassigned_english_regions,
)
from app.engine.pipeline import _flatten_uniform_english_regions, _is_valid_ocr_text
import numpy as np


class AutomaticOcrRoutingTests(unittest.TestCase):
    def test_real_english_dialogue_uses_fast_path(self):
        regions = [
            {"text": "NOW, THEN..."},
            {"text": "THE MATCH WILL BEGIN!"},
        ]
        self.assertTrue(_english_page_is_confident(regions))

    def test_short_latin_artifacts_do_not_select_english(self):
        regions = [{"text": "% TOP"}, {"text": '"LO'}, {"text": "W"}]
        self.assertFalse(_english_page_is_confident(regions))

    def test_stylized_english_is_plausible_even_when_fast_path_is_strict(self):
        regions = [
            {"text": "There should be a hot spring therell"},
            {"text": "The place that showep up in that Movie We Watqed"},
        ]
        self.assertTrue(_english_page_is_plausible(regions))

    def test_digit_heavy_latin_hallucination_is_not_plausible_english(self):
        regions = [
            {"text": "J3 ;6 efxe: 6"},
            {"text": "2 l 78706 . J5,7ii rt ;o5xiz?617837"},
        ]
        self.assertFalse(_english_page_is_plausible(regions))

    def test_digit_heavy_japanese_hallucination_does_not_select_english(self):
        regions = [
            {"text": "J3 ;6 efxe: 6"},
            {"text": "2 l 78706 . J5,7ii rt ;o5xiz?617837"},
            {"text": "U: bf Gltw wt8 Le756 bgewi372.5 If? If328"},
        ]
        self.assertFalse(_english_page_is_confident(regions))

    def test_stylized_english_overrides_manga_ocr_hallucination(self):
        regions = [
            {"text": "Dark elf training began outside"},
            {"text": "There should be a hot spring nearby"},
        ]
        scores = {"en": 90, "ja": 316, "ko": 0, "zh": 0}
        evidence = {"latin": 90, "kana": 71, "han_ja": 32}
        self.assertTrue(
            _english_overrides_japanese(regions, scores, evidence)
        )

    def test_real_japanese_page_is_not_overridden_by_latin_noise(self):
        regions = [
            {"text": "Stage clear fantasy story"},
            {"text": "Knight level status"},
        ]
        scores = {"en": 190, "ja": 1157, "ko": 0, "zh": 0}
        evidence = {"latin": 190, "kana": 240, "han_ja": 197}
        self.assertFalse(
            _english_overrides_japanese(regions, scores, evidence)
        )

    def test_manga_ocr_kana_selects_japanese(self):
        regions = [
            {"text": "それ以外はアルトラ達を使い"},
            {"text": "必ず生きて逃げ切り"},
        ]
        self.assertTrue(_japanese_page_is_confident(regions))

    def test_real_japanese_page_can_skip_other_cpu_readers(self):
        regions = [
            {"text": "瘴気を放つ遺跡はここから少し先の海底にあった"},
            {"text": "海底に遺跡があるなんて"},
        ]
        self.assertTrue(_japanese_fast_path_is_confident(regions))

    def test_stray_kana_does_not_skip_korean_comparison(self):
        regions = [{"text": "가나다라 の"}, {"text": "한국어 대화"}]
        self.assertFalse(_japanese_fast_path_is_confident(regions))

    def test_colour_webtoon_page_requires_native_reader_comparison(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[:, :100] = [225, 155, 120]
        image[:, 100:] = [80, 140, 210]
        self.assertTrue(_image_has_meaningful_color(image))
        self.assertFalse(
            _image_has_meaningful_color(np.full((200, 200, 3), 180, dtype=np.uint8))
        )

    def test_short_real_korean_balloon_beats_fake_kana(self):
        regions = [{"text": "사람들이 왜 쳐다보지?"}]
        evidence = {"kana": 3, "hangul": 9, "ko_len": 11}
        self.assertTrue(_korean_page_is_confident(regions, evidence))

    def test_sparse_hangul_noise_does_not_select_korean(self):
        regions = [{"text": "가 abcdefghij"}]
        evidence = {"kana": 0, "hangul": 1, "ko_len": 11}
        self.assertFalse(_korean_page_is_confident(regions, evidence))

    def test_dense_chinese_pass_can_override_fake_japanese(self):
        regions = [{"text": "人们为什么都盯着我看"}]
        scores = {"en": 0, "ja": 18, "ko": 0, "zh": 30}
        evidence = {"kana": 3, "han_zh": 10}
        self.assertTrue(_chinese_page_is_confident(regions, scores, evidence))

    def test_easyocr_hangul_hallucinations_cannot_override_kana(self):
        scores = {"en": 1, "ja": 8, "ko": 16, "zh": 18}
        evidence = {
            "kana": 2,
            "hangul": 9,
            "han_ja": 8,
            "han_zh": 10,
            "latin": 6,
        }
        self.assertEqual(_choose_auto_language(scores, evidence), "ja")

    def test_script_evidence_counts_real_kana(self):
        evidence = _script_evidence({
            "en": [],
            "ja": [{"text": "その為なら私は"}],
            "ko": [{"text": ""}],
        })
        self.assertEqual(evidence["kana"], 5)
        self.assertEqual(evidence["han_ja"], 2)

    def test_english_score_rejects_digit_heavy_hallucination(self):
        self.assertGreater(
            _english_text_score("WHEN YOU PUSH YOURSELF AND FOCUS"),
            _english_text_score("42I 3o4 QLtg 83 914Jotz 4Z444"),
        )

    def test_stylized_english_garbage_requests_crop_retry(self):
        self.assertTrue(_english_needs_crop_retry("Thhegly Finew"))
        self.assertTrue(_english_needs_crop_retry("Uncle's effemder Buevee and"))
        self.assertFalse(_english_needs_crop_retry("He's totally fine!!"))
        self.assertGreater(
            _english_candidate_quality("He's totally fined"),
            _english_candidate_quality("Thhegly Finew"),
        )

    def test_english_crop_retries_are_bounded_and_keep_empty_candidate(self):
        regions = [
            {"bbox": [0, 0, 100 + index * 10, 80]}
            for index in range(7)
        ]
        texts = [
            "Thhegly Finew",
            "Uncle's effemder Buevee and",
            "Strangely Wrogn text",
            "Another stylized sentnce",
            "More unknown lettters here",
            "Badly scannedd dialogue",
            "",
        ]
        selected = _select_english_crop_retries(regions, texts, limit=4)

        self.assertEqual(len(selected), 4)
        self.assertTrue(any(index == 6 for index, _, _ in selected))

    def test_english_validation_rejects_isolated_detector_noise(self):
        for text in ("F", "1", "X", 'Zh" "'):
            self.assertFalse(_is_valid_ocr_text(text, "en"))
        self.assertTrue(_is_valid_ocr_text("I", "en"))
        self.assertTrue(_is_valid_ocr_text("NO", "en"))

    def test_english_validation_does_not_change_other_scripts(self):
        self.assertTrue(_is_valid_ocr_text("何?", "ja"))
        self.assertTrue(_is_valid_ocr_text("안녕", "ko"))

    def test_japanese_validation_keeps_fragmented_short_dialogue(self):
        for text in ("は…？", "ル", "イ", "ん"):
            self.assertTrue(_is_valid_ocr_text(text, "ja"), text)

    def test_uniform_english_background_cleanup_removes_residual_glyphs(self):
        image = np.full((100, 140, 3), 248, dtype=np.uint8)
        image[35:65, 45:95] = 20
        cleaned = _flatten_uniform_english_regions(
            image,
            [{
                "bbox": [40, 30, 100, 70],
                "ocr_polygons": [[[45, 35], [95, 35], [95, 65], [45, 65]]],
            }],
        )
        self.assertGreater(float(cleaned[35:65, 45:95].mean()), 230)

    def test_nonuniform_background_is_not_flattened(self):
        x = np.arange(140, dtype=np.uint8)
        image = np.repeat(x[None, :, None], 100, axis=0)
        image = np.repeat(image, 3, axis=2)
        cleaned = _flatten_uniform_english_regions(
            image,
            [{
                "bbox": [40, 30, 100, 70],
                "ocr_polygons": [[[45, 35], [95, 35], [95, 65], [45, 65]]],
            }],
        )
        np.testing.assert_array_equal(cleaned, image)

    def test_page_english_ocr_lines_are_assigned_in_reading_order(self):
        regions = [{"bbox": [0, 0, 200, 120]}, {"bbox": [250, 0, 400, 100]}]
        recognized = [
            ([[10, 50], [100, 50], [100, 70], [10, 70]], "SECOND", 0.9),
            ([[10, 10], [100, 10], [100, 30], [10, 30]], "FIRST", 0.9),
            ([[270, 10], [380, 10], [380, 30], [270, 30]], "OTHER", 0.9),
        ]
        self.assertEqual(
            _assign_english_lines(regions, recognized),
            ["FIRST SECOND", "OTHER"],
        )
        self.assertEqual(len(regions[0]["ocr_polygons"]), 2)
        self.assertEqual(len(regions[1]["ocr_polygons"]), 1)

    def test_unassigned_english_dialogue_is_recovered_and_grouped(self):
        regions = [{"bbox": [0, 0, 100, 80]}]
        recognized = [
            ([[150, 10], [260, 10], [260, 30], [150, 30]], "HE'S TOTALLY", 0.9),
            ([[165, 34], [245, 34], [245, 54], [165, 54]], "FINE!!", 0.88),
        ]
        texts, unassigned = _assign_english_lines(
            regions, recognized, return_unassigned=True
        )
        recovered = _recover_unassigned_english_regions(
            unassigned, (200, 400, 3)
        )
        self.assertEqual(texts, [""])
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0]["text"], "He's totally fine!!")
        self.assertTrue(recovered[0]["from_opencv"])
        self.assertEqual(len(recovered[0]["ocr_polygons"]), 2)

    def test_unassigned_english_noise_and_watermarks_are_not_recovered(self):
        unassigned = [
            (10, 10, 30, 30, "NGH", 0.2),
            (40, 40, 180, 60, "KAYNSCAN.COM", 0.95),
        ]
        self.assertEqual(
            _recover_unassigned_english_regions(unassigned, (200, 400, 3)),
            [],
        )

    def test_english_ocr_normalization_repairs_font_confusions(self):
        self.assertEqual(
            _normalize_english_ocr(
                "Zhell 0F couzse. The Stronger Wels Feel- INGS Are;"
            ),
            "Well of course. The stronger Wels feelings are;",
        )
        self.assertEqual(
            _normalize_english_ocr("MY Bodys About T0 FALL Apart:"),
            "My body's about to fall apart:",
        )
        self.assertEqual(
            _normalize_english_ocr(
                "Hes totally Finew. So this is. The Hepo of mankindj"
            ),
            "He's totally fine. So this is the hero of mankind",
        )
        self.assertEqual(
            _normalize_english_ocr("Mana haroening!? A human can Do thati?"),
            "Mana hardening!? A human can do that?",
        )
        self.assertEqual(
            _normalize_english_ocr(
                "Aojusteo Meia? That coulo ve 'klled Mel"
            ),
            "Adjusted me!? That could've killed me!",
        )
        self.assertEqual(
            _normalize_english_ocr(
                'Aojusted Meia? That co"ld y klleo Mel'
            ),
            "Adjusted me!? That could've killed me!",
        )
        self.assertEqual(
            _normalize_english_ocr(
                "Howwas I 5u9poseo to ficugejhat Outl?"
            ),
            "How was I supposed to figure that out!?",
        )
        self.assertEqual(
            _normalize_english_ocr(
                "Beng able to Yill a magic beast without the holy shord"
            ),
            "Being able to kill a magic beast without the holy sword",
        )
        self.assertEqual(
            _normalize_english_ocr("Iov Wepent Gonna die eumavi"),
            "You weren't gonna die anyway!",
        )
        self.assertEqual(
            _normalize_english_ocr("Sls? 5e9l"),
            "Elsie! Elsie!",
        )
        self.assertEqual(
            _normalize_english_ocr("Uncle's getting bitten over and over but. Oee"),
            "Uncle's getting bitten over and over but.",
        )
        self.assertEqual(
            _normalize_english_ocr(
                "Mana hacoeningi? Uncle's getting bitten over and Nuvee"
            ),
            "Mana hardening? Uncle's getting bitten over and over",
        )
        self.assertEqual(
            _normalize_english_ocr("Lyou really grew. Mana Ordlnary. Mamay"),
            "You really grew. Mana ordinary. Mama!!",
        )
        self.assertEqual(
            _normalize_english_ocr(
                "Uncles getting bitten over and over but. Cee"
            ),
            "Uncle's getting bitten over and over but.",
        )
        self.assertEqual(
            _normalize_english_ocr("Mana ox Normal. Mana 07\" Ordinary"),
            "Mana 0% Normal. Mana 0% Ordinary",
        )

    def test_english_normalization_does_not_process_other_scripts(self):
        # The helper is only called by EasyOCR's explicit English branch and
        # leaves non-Latin scripts byte-for-byte intact if invoked directly.
        self.assertEqual(_normalize_english_ocr("その為なら私は"), "その為なら私は")


if __name__ == "__main__":
    unittest.main()
