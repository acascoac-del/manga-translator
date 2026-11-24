"""Shared prompt and output-quality helpers for LLM translators."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher


SYSTEM_MESSAGE = "You are a precise professional manga translator."

# Shared Groq-tuned sampling defaults for every LLM translator.
LLM_TEMPERATURE = 0.1
LLM_TOP_P = 0.85
LLM_REPEAT_PENALTY = 1.1


def chat_messages(prompt: str) -> list[dict[str, str]]:
    """Build the identical chat envelope used by every LLM translator."""
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]


def openai_chat_payload(model: str, prompt: str, **extra) -> dict:
    """Build an OpenAI-compatible chat payload (Groq and similar APIs)."""
    return {
        "model": model,
        "messages": chat_messages(prompt),
        "temperature": LLM_TEMPERATURE,
        "top_p": LLM_TOP_P,
        **extra,
    }


def llamacpp_completion_kwargs(prompt: str, *, max_tokens: int = 256) -> dict:
    """Build llama.cpp create_chat_completion kwargs aligned with Groq."""
    return {
        "messages": chat_messages(prompt),
        "temperature": LLM_TEMPERATURE,
        "top_p": LLM_TOP_P,
        "repeat_penalty": LLM_REPEAT_PENALTY,
        "max_tokens": max_tokens,
    }


_LANGUAGE_NAMES = {
    "auto": "the automatically detected source language",
    "en": "English",
    "eng_latn": "English",
    "es": "Spanish",
    "spa_latn": "Spanish",
    "ja": "Japanese",
    "jpn_jpan": "Japanese",
    "ko": "Korean",
    "kor_hang": "Korean",
    "zh": "Chinese",
    "zh-cn": "Simplified Chinese",
    "ch_sim": "Simplified Chinese",
    "zho_hans": "Simplified Chinese",
}

_ENGLISH_COMMON = {
    "a", "about", "again", "and", "are", "as", "at", "be", "but", "do",
    "does", "dont", "for", "from", "has", "have", "he", "her", "him", "his",
    "i", "if", "in", "is", "it", "me", "my", "never", "no", "not", "now",
    "of", "on", "or", "our", "she", "so", "that", "the", "then", "there",
    "they", "this", "to", "was", "we", "what", "when", "where", "who", "why",
    "will", "with", "you", "your",
}

# One of these in a purported Spanish result is enough to flag a mixed output.
_ENGLISH_DISTINCTIVE = {
    "absolute", "agreement", "awkward", "blabbing", "brother", "chance",
    "completely", "earth", "happen", "happens", "marry", "miracle", "nonsense",
    "permit", "princess", "ridiculous", "setting", "sister", "stakes", "talking",
    "thanks", "understand", "move", "moves",
}

# Words that are legitimately spelled the same in English and Spanish. They
# must not trigger the retained-source-word check below.
_SHARED_EN_ES = {
    "actor", "animal", "bar", "base", "capital", "central", "club", "color",
    "criminal", "doctor", "error", "fatal", "final", "general", "hospital",
    "hotel", "ideal", "legal", "local", "metal", "moral", "normal", "original",
    "mana", "popular", "radio", "real", "regular", "similar", "social", "total",
}

_SPANISH_COMMON = {
    "a", "al", "ahora", "como", "con", "de", "del", "el", "ella", "en", "entonces",
    "es", "esta", "este", "gracias", "ha", "hay", "la", "las", "lo", "los",
    "me", "mi", "no", "nos", "para", "pero", "por", "que", "se", "si", "sin",
    "su", "sus", "te", "tu", "un", "una", "usted", "y", "ya", "yo",
}

_MODEL_REFUSAL_PATTERNS = (
    r"\bno se (?:ha )?proporcion(?:o|ado) (?:un )?texto\b",
    r"\bno (?:hay|se detecto) texto (?:legible|para traducir)\b",
    r"\b(?:unable|cannot|can't) (?:to )?translat(?:e|ion)\b",
    r"\bno readable text\b",
    r"\bplease provide (?:the )?(?:text|dialogue)\b",
)


def _is_model_refusal(text: str) -> bool:
    folded = "".join(
        char for char in unicodedata.normalize("NFKD", text.lower())
        if not unicodedata.combining(char)
    )
    return any(re.search(pattern, folded) for pattern in _MODEL_REFUSAL_PATTERNS)


def language_name(lang: str | None) -> str:
    normalized = (lang or "auto").lower().strip()
    return _LANGUAGE_NAMES.get(normalized, lang or _LANGUAGE_NAMES["auto"])


def clean_model_output(text: str | None) -> str:
    value = (text or "").strip()
    value = re.sub(
        r"^(?:translation|translated text|traducci[oó]n)\s*:\s*",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()
    return value


def parse_numbered_output(raw: str | None, expected_count: int) -> dict[int, str]:
    """Parse a numbered LLM response without shifting speech-bubble indices.

    Accepts common forms such as ``1.``, ``1:``, ``[1]`` and multiline
    translations. Only explicit, in-range indices are returned.
    """
    parsed_parts: dict[int, list[str]] = {}
    current: int | None = None
    pattern = re.compile(r"^\s*\[?(\d+)\]?\s*(?:[.):-]|-\s)?\s+(.+?)\s*$")

    for raw_line in (raw or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if match:
            index = int(match.group(1)) - 1
            if 0 <= index < expected_count:
                current = index
                parsed_parts.setdefault(index, []).append(match.group(2).strip())
            else:
                current = None
            continue
        if current is not None:
            parsed_parts[current].append(line)

    return {
        index: clean_model_output(" ".join(parts))
        for index, parts in parsed_parts.items()
        if clean_model_output(" ".join(parts))
    }


async def repair_failed_outputs(
    results: list[str],
    indexed: list[tuple[int, str]],
    source_lang: str,
    target_lang: str,
    engine_name: str,
    logger,
) -> list[str]:
    """Apply the same quality gate and Google repair to every LLM engine.

    Repairs use Google's batch path so one bad page cannot create dozens of
    simultaneous fallback clients.  Besides being faster, this honours the
    shared circuit breaker when Google is unavailable.
    """
    from . import google
    pending = [
        (index, original)
        for index, original in indexed
        if needs_google_fallback(original, results[index], source_lang, target_lang)
    ]
    if not pending:
        return results

    try:
        fallbacks = await google.translate_batch(
            [original for _, original in pending], source_lang, target_lang
        )
    except Exception as exc:
        logger.warning("Google batch repair failed for %s: %s", engine_name, exc)
        return results

    for (index, original), fallback in zip(pending, fallbacks):
        rejected = results[index]
        fallback = clean_model_output(fallback)
        if fallback and not needs_google_fallback(
            original, fallback, source_lang, target_lang
        ):
            logger.info(
                "Google repaired %s region %s: source=%r rejected=%r repaired=%r",
                engine_name, index, original, rejected, fallback,
            )
            results[index] = fallback
        else:
            logger.warning(
                "Google fallback returned an invalid/untranslated result for %s "
                "region %s: %r",
                engine_name, index, fallback,
            )
    return results


def contains_cjk(text: str) -> bool:
    return any(
        "\u3040" <= char <= "\u30ff"
        or "\u3400" <= char <= "\u9fff"
        or "\uac00" <= char <= "\ud7af"
        for char in text
    )


def _words(text: str) -> list[str]:
    folded = "".join(
        char for char in unicodedata.normalize("NFKD", text.lower())
        if not unicodedata.combining(char)
    )
    return re.findall(r"[a-z]+", folded)


def _normalized(text: str) -> str:
    return " ".join(_words(text))


def needs_google_fallback(
    source: str,
    translated: str,
    source_lang: str = "auto",
    target_lang: str = "es",
) -> bool:
    """Reject untranslated, copied, CJK, or English/Spanish mixed LLM output."""
    source = (source or "").strip()
    translated = clean_model_output(translated)
    if not translated:
        return True
    if _is_model_refusal(translated):
        return True

    target = (target_lang or "es").lower().strip()
    source_norm = _normalized(source)
    translated_norm = _normalized(translated)

    if source_norm and source_norm == translated_norm:
        return True
    if len(source_norm) >= 10 and len(translated_norm) >= 10:
        if SequenceMatcher(None, source_norm, translated_norm).ratio() >= 0.82:
            return True

    if target not in {"ja", "ko", "zh", "zh-cn", "ch_sim"} and contains_cjk(translated):
        return True

    if target in {"es", "spa_latn"}:
        words = _words(translated)
        source_words_list = _words(source)
        english_common = sum(word in _ENGLISH_COMMON for word in words)
        spanish_common = sum(word in _SPANISH_COMMON for word in words)
        if any(word in _ENGLISH_DISTINCTIVE for word in words):
            return True
        if english_common >= 2 and english_common > spanish_common:
            return True

        # Reject retained *English* words, not every word shared with the
        # source. Manga dialogue legitimately preserves names (Rimu, Wataru),
        # honorifics (-chan, -san), interjections and loanwords such as video.
        # The former broad intersection rejected valid Spanish from both Groq
        # and Google, causing the pipeline to restore the original English.
        source_words = set(source_words_list)
        retained = {
            word for word in words
            if (
                len(word) >= 4
                and word in source_words
                and word not in _SHARED_EN_ES
                and word not in _SPANISH_COMMON
                and word in _ENGLISH_COMMON
            )
        }
        if retained:
            return True

        # Small local models sometimes translate the surrounding question
        # while silently dropping its essential fantasy term.
        if "mana" in source_words and "mana" not in words:
            return True

    return False


def single_prompt(source_lang: str, target_lang: str, text: str) -> str:
    return (
        f"Translate this manga dialogue from {language_name(source_lang)} to "
        f"{language_name(target_lang)}. The input may contain minor OCR spacing, casing, "
        "or character-recognition errors; silently correct only obvious OCR mistakes before "
        "translating. Return the complete dialogue entirely in the target language. Never "
        "copy source-language phrases into the result. Keep names unchanged and use natural, "
        "concise speech-bubble dialogue. Output only the translation, without quotes, labels, "
        "notes, or explanations.\n\n"
        f"Text to translate:\n{text}"
    )


def batch_prompt(source_lang: str, target_lang: str, numbered: str) -> str:
    source_notes = (
        "Japanese notes: short reactions such as なに/何 mean 'what?', not 'because'. "
        "A question may continue in another balloon (何 ... 言ってるんだ = "
        "'what ... are you saying?'). Japanese often omits pronouns: do not automatically "
        "add 'I' to every verb or invent possessives for relatives.\n"
        if source_lang.lower() in {"ja", "jpn", "japanese", "jpn_jpan"} else ""
    )
    return (
        f"Translate every numbered manga dialogue from {language_name(source_lang)} to "
        f"{language_name(target_lang)}. The inputs may contain minor OCR spacing, casing, or "
        "character-recognition errors; silently correct only obvious OCR mistakes.\n"
        "Rules:\n"
        "- Return every translation entirely in the target language; never leave English or "
        "other source-language phrases mixed into it.\n"
        "- Output exactly one numbered translation per input, using the same numbers.\n"
        "- Never merge, skip, reorder, or invent lines.\n"
        "- Keep names unchanged and use natural, concise manga dialogue.\n"
        "- Use the other lines as context to resolve omitted subjects and sentence fragments, "
        "but keep each fragment's meaning under its original number. Input order may differ "
        "from reading order. Preserve questions, negation, and who performs each action; "
        "do not turn an ambiguous fragment into a confident invented statement.\n"
        "- Output no explanations, labels, notes, or extra lines.\n\n"
        f"{source_notes}Lines to translate:\n{numbered}"
    )
