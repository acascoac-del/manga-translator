"""
Translation using native llama.cpp (GGUF model).
Runs Tencent HY-MT1.5 1.8B, a translation-specialized model, locally.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import re
import threading
import time
from huggingface_hub import hf_hub_download

from .quality import (
    clean_model_output,
    language_name,
    needs_google_fallback,
    parse_numbered_output,
)

log = logging.getLogger("translation.llamacpp")

# We will lazily load llama_cpp to avoid crashing if it's not installed yet
_llama_model = None
_thread_lock = threading.Lock()

# Default model info. Q6 retains substantially more translation accuracy than
# Q4 while remaining smaller than the previous Qwen2.5 3B Q4 model.
DEFAULT_REPO = "tencent/HY-MT1.5-1.8B-GGUF"
DEFAULT_FILES = (
    "HY-MT1.5-1.8B-Q6_K.gguf",
)
WEIGHTS_DIR = "/app/models/weights"

# Inference settings recommended by the model publisher. HY-MT is a dedicated
# translation model and intentionally has no system prompt.
# HY-MT's published decoding configuration.  Very low temperature makes this
# translation-specialized model fall into a copy-the-source mode, especially
# for English -> Spanish batches.
HY_TEMPERATURE = 0.7
HY_TOP_K = 20
HY_TOP_P = 0.6
HY_REPEAT_PENALTY = 1.05


def _generation_timeout_seconds() -> float:
    try:
        return max(5.0, float(os.getenv("LLAMACPP_MAX_GENERATION_SECONDS", "15")))
    except ValueError:
        return 15.0


def _translation_prompt(
    target_lang: str, text: str, source_lang: str = "auto"
) -> str:
    """Use HY-MT's native prompt rather than a general chatbot envelope."""
    source = (source_lang or "auto").lower().strip()
    source_hint = ""
    if source in {"ko", "kor", "korean"}:
        source_hint = " from Korean"
    elif source in {"zh", "zh-cn", "zh-tw", "ch_sim", "ch_tra", "chinese"}:
        source_hint = " from Chinese"
    return (
        f"Translate the following segment{source_hint} into "
        f"{language_name(target_lang)}, "
        "without additional explanation.\n\n"
        f"{text}"
    )


def _batch_translation_prompt(
    target_lang: str, numbered: str, source_lang: str = "auto"
) -> str:
    """Translate a numbered page with HY-MT's exact native prompt.

    HY-MT is not an instruction-following chat model.  Adding several output
    rules to its translation instruction can make it echo the complete source.
    Numbering itself is enough to retain region alignment and is validated by
    the parser and shared quality gate afterwards.
    """
    return _translation_prompt(target_lang, numbered, source_lang)


def _completion_kwargs(prompt: str, max_tokens: int) -> dict:
    return {
        "prompt": prompt,
        "temperature": HY_TEMPERATURE,
        "top_k": HY_TOP_K,
        "top_p": HY_TOP_P,
        "repeat_penalty": HY_REPEAT_PENALTY,
        "max_tokens": max_tokens,
    }


def _stream_completion(prompt: str, max_tokens: int) -> dict:
    """Generate with a wall-clock limit without a logits callback.

    ``llama_cpp`` stopping criteria receive the complete logits array after
    every token.  With a CUDA model that forces a GPU/CPU synchronization and
    reduced this 1.8B model to roughly two tokens per second.  Streaming gives
    Python only the selected token, so the deadline remains enforceable without
    copying the vocabulary-sized logits buffer on every generation step.
    """
    started = time.perf_counter()
    deadline = started + _generation_timeout_seconds()
    pieces: list[str] = []
    completion_tokens = 0
    timed_out = False
    finish_reason = None
    chunks = _llama_model.create_completion(
        **_completion_kwargs(prompt, max_tokens=max_tokens),
        stream=True,
    )
    try:
        for chunk in chunks:
            choices = chunk.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            piece = choice.get("text", "")
            if piece:
                pieces.append(piece)
                completion_tokens += 1
            finish_reason = choice.get("finish_reason") or finish_reason
            if time.perf_counter() >= deadline and finish_reason is None:
                timed_out = True
                break
    finally:
        close = getattr(chunks, "close", None)
        if callable(close):
            close()

    return {
        "choices": [{"text": "".join(pieces), "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": "?", "completion_tokens": completion_tokens},
        "_elapsed_seconds": time.perf_counter() - started,
        "_timed_out": timed_out,
    }


def _fix_japanese_honorific(source: str, translated: str) -> str:
    """Prevent さん after a name from becoming a family/religious title.

    HY-MT occasionally localizes names such as バルトロメウスさん as
    ``padre Bartolomé``. The source only contains the neutral honorific さん,
    so correct that narrow, objectively invalid interpretation. Other regions
    mentioning an actual parent/guardian remain untouched.
    """
    if "さん" not in (source or ""):
        return translated
    value = re.sub(
        r"\bpadre(?=\s+[A-ZÁÉÍÓÚÑ])",
        "señor",
        translated,
        flags=re.IGNORECASE,
    )
    return re.sub(
        r"\bmadre(?=\s+[A-ZÁÉÍÓÚÑ])",
        "señora",
        value,
        flags=re.IGNORECASE,
    )


def _output_token_budget(indexed: list[tuple[int, str]]) -> int:
    """Bound generation to the amount of dialogue on the page.

    A fixed 1024-token ceiling lets malformed responses run far beyond what a
    manga page needs. Spanish can expand relative to CJK, so retain generous
    headroom while capping the common case much lower.
    """
    if not indexed:
        return 32
    source_chars = sum(len(text.strip()) for _, text in indexed)
    return min(640, max(64, source_chars * 2 + len(indexed) * 12))


async def _load_local_model():
    global _llama_model
    if _llama_model is not None:
        return _llama_model
        
    with _thread_lock:
        if _llama_model is not None:
            return _llama_model
            
        import llama_cpp

        supports_gpu_offload = bool(llama_cpp.llama_supports_gpu_offload())
        if not supports_gpu_offload:
            raise RuntimeError(
                "llama-cpp-python was installed without CUDA support. "
                "Rebuild the Docker image so the cu121 wheel is installed."
            )
        log.info(
            "llama.cpp %s: CUDA GPU offload available",
            getattr(llama_cpp, "__version__", "unknown"),
        )
        
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        model_path = os.path.join(WEIGHTS_DIR, DEFAULT_FILES[0])

        missing_files = [
            filename for filename in DEFAULT_FILES
            if not os.path.exists(os.path.join(WEIGHTS_DIR, filename))
        ]
        if missing_files:
            log.info(
                "Downloading %s HY-MT1.5 1.8B GGUF file(s) from HuggingFace...",
                len(missing_files),
            )
            loop = asyncio.get_event_loop()

            def download_model():
                paths = []
                for filename in missing_files:
                    paths.append(hf_hub_download(
                        repo_id=DEFAULT_REPO,
                        filename=filename,
                        local_dir=WEIGHTS_DIR,
                    ))
                return paths
            try:
                await loop.run_in_executor(None, download_model)
            except Exception as e:
                log.error(f"Failed to download model: {e}")
                raise
                
        log.info(f"Loading llama.cpp model from {model_path}...")
        
        # Load model in a thread
        loop = asyncio.get_event_loop()
        def load_model():
            # GPU inference should not occupy every CPU core: OCR and image
            # rendering need headroom or the whole PC becomes unresponsive.
            cpu_threads = max(2, min(4, os.cpu_count() or 4))
            model = llama_cpp.Llama(
                model_path=model_path,
                # The 1.8B Q6 model fits comfortably on the 6 GB GPU.
                n_gpu_layers=-1,
                n_ctx=1536,
                # Larger prompt batches reduce page-level prompt evaluation.
                # Flash Attention and K/Q/V offload keep the full translation
                # path on the RTX GPU and reduce memory traffic.
                n_batch=512,
                n_ubatch=256,
                n_threads=cpu_threads,
                n_threads_batch=cpu_threads,
                flash_attn=True,
                offload_kqv=True,
                op_offload=True,
                verbose=False
            )
            # CUDA kernels and the tokenizer are initialized lazily. Two
            # tiny generations move that one-time cost to application startup
            # instead of delaying the user's first translated pages.
            warmup_prompt = _translation_prompt("es", "Hello")
            for _ in range(2):
                model.create_completion(
                    **_completion_kwargs(warmup_prompt, max_tokens=1),
                )
            return model
            
        _llama_model = await loop.run_in_executor(None, load_model)
        log.info(
            "llama.cpp model loaded, all model layers offloaded and CUDA-warmed successfully"
        )
        return _llama_model

async def preload():
    """Preload llama.cpp model so first request doesn't wait."""
    try:
        await _load_local_model()
    except Exception as e:
        log.warning(f"Failed to preload llama.cpp: {e}")

async def translate_text(
    text: str,
    source_lang: str = "auto",
    target_lang: str = "es",
) -> str:
    """Translate a single text using llama.cpp."""
    if not text or not text.strip():
        return ""
        
    await _load_local_model()
    
    prompt = _translation_prompt(target_lang, text, source_lang)
    
    # We must run generation in a thread since llama.cpp is CPU-blocking
    loop = asyncio.get_event_loop()
    def do_generate():
        with _thread_lock:
            max_tokens = _output_token_budget([(0, text)])
            return _stream_completion(prompt, max_tokens)
        
    try:
        response = await loop.run_in_executor(None, do_generate)
        translated = clean_model_output(response["choices"][0]["text"])
        translated = _fix_japanese_honorific(text, translated)
        usage = response.get("usage", {})
        log.info(
            "llama.cpp single generation: %.0fms, prompt=%s, completion=%s tokens",
            response.get("_elapsed_seconds", 0) * 1000,
            usage.get("prompt_tokens", "?"),
            usage.get("completion_tokens", "?"),
        )
        if response.get("_timed_out"):
            log.warning("llama.cpp single generation reached its time limit")
        return translated if translated else text
    except Exception as e:
        log.error(f"llama.cpp translation failed: {e}")
        return text

async def translate_batch(
    texts: list[str],
    source_lang: str = "auto",
    target_lang: str = "es",
) -> list[str]:
    """Translate all texts using batch prompting with llama.cpp."""
    indexed = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
    results = list(texts)
    
    if not indexed:
        return results
        
    if len(indexed) == 1:
        i, t = indexed[0]
        results[i] = await translate_text(t, source_lang, target_lang)
        return results
        
    await _load_local_model()

    async def generate_batch(batch: list[tuple[int, str]]) -> dict[int, str]:
        numbered = "\n".join(
            f"{number + 1}. {text}" for number, (_, text) in enumerate(batch)
        )
        prompt = _batch_translation_prompt(target_lang, numbered, source_lang)
        max_tokens = _output_token_budget(batch)
        loop = asyncio.get_running_loop()

        def do_generate():
            with _thread_lock:
                return _stream_completion(prompt, max_tokens)

        response = await loop.run_in_executor(None, do_generate)
        usage = response.get("usage", {})
        log.info(
            "llama.cpp batch generation: %.0fms, regions=%s, budget=%s, "
            "prompt=%s, completion=%s tokens",
            response.get("_elapsed_seconds", 0) * 1000,
            len(batch),
            max_tokens,
            usage.get("prompt_tokens", "?"),
            usage.get("completion_tokens", "?"),
        )
        if response.get("_timed_out"):
            log.warning(
                "llama.cpp batch reached its %.0fs generation limit; "
                "missing regions will use the fallback translator",
                _generation_timeout_seconds(),
            )
        raw = response["choices"][0]["text"].strip()
        log.info("LLAMA.CPP RAW RESPONSE:\n%s", raw)
        parsed = parse_numbered_output(raw, len(batch))
        return {
            original_index: _fix_japanese_honorific(original_text, parsed[number])
            for number, (original_index, original_text) in enumerate(batch)
            if number in parsed
        }

    try:
        results_by_index = await generate_batch(indexed)
        for original_index, translated in results_by_index.items():
            results[original_index] = translated

        # Do not run the same local model for a second full deadline. The shared
        # dispatcher already repairs only these rejected regions with the fast
        # fallback provider; repeating llama.cpp here doubled page latency.
        invalid = [
            (orig_i, orig_t)
            for orig_i, orig_t in indexed
            if needs_google_fallback(
                orig_t, results[orig_i], source_lang, target_lang
            )
        ]
        if invalid:
            log.info(
                "llama.cpp deferring %s invalid/missing regions to dispatcher repair",
                len(invalid),
            )

        log.info(f"llama.cpp batch translated {len(indexed)} texts")
    except Exception as e:
        log.error(f"llama.cpp batch translation failed: {e}")
        # Preserve source text. The dispatcher may try one different provider;
        # repeating this same model region-by-region caused most of the stalls.

    return results

async def translate_regions(
    textlines: list[dict],
    source_lang: str = "auto",
    target_lang: str = "es",
) -> list[dict]:
    """Translate all text regions."""
    texts = [tl.get("text", "") for tl in textlines]
    translations = await translate_batch(texts, source_lang, target_lang)

    for tl, trans in zip(textlines, translations):
        tl["translated_text"] = trans

    log.info(f"llama.cpp translated {len(textlines)} regions ({source_lang} → {target_lang}).")
    return textlines

async def cleanup():
    """Close the native context and return its VRAM to the system."""
    global _llama_model
    had_model = False
    with _thread_lock:
        model = _llama_model
        _llama_model = None
        if model is not None:
            had_model = True
            try:
                model.close()
            except Exception as exc:
                log.warning("Could not explicitly close llama.cpp: %s", exc)
    if had_model:
        del model
        gc.collect()
        log.info("llama.cpp model unloaded and VRAM released")


async def release_if_loaded():
    """Unload the local model only when it is currently resident."""
    if _llama_model is not None:
        await cleanup()
