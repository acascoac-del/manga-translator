"""Standalone GPU entrypoint. No changes or imports of the local FastAPI app."""
import asyncio
import base64
import io
import os
import uuid
from urllib.parse import urlparse
import httpx
from PIL import Image

CONTROL_URL = os.environ.get('COMMERCIAL_CONTROL_URL', '').rstrip('/')
MAX_TRANSFER = 16 * 1024 * 1024


def validate_input(data):
    job_id = str(uuid.UUID(data['job_id']))
    token = data['token']
    if not isinstance(token, str) or not 32 <= len(token) <= 100:
        raise ValueError('Invalid transfer token')
    source, target = data.get('source', 'auto'), data.get('target', 'es')
    if source not in {'auto', 'en', 'ja', 'ko', 'zh'} or target != 'es':
        raise ValueError('Unsupported language')
    parsed = urlparse(CONTROL_URL)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.query or parsed.fragment:
        raise ValueError('COMMERCIAL_CONTROL_URL must be a fixed HTTPS origin')
    return job_id, token, source, target


async def local_only_translation(textlines, source_lang='auto', target_lang='es', **kwargs):
    # Do not use unofficial free Google endpoints or silently invoke another
    # paid provider. The commercial package includes only local translation.
    from app.engine.translation import llamacpp
    from app.engine.translation.quality import needs_google_fallback
    regions = await llamacpp.translate_regions(textlines, source_lang=source_lang, target_lang=target_lang)
    if any(needs_google_fallback(r.get('text', ''), r.get('translated_text', ''), source_lang, target_lang) for r in regions):
        raise RuntimeError('Commercial translation incomplete; refusing to charge')
    return regions


async def process(data):
    job_id, token, source, target = validate_input(data)
    from app.engine.pipeline import PipelineConfig, run_pipeline
    from app.engine.translation import dispatcher
    dispatcher.translate_regions = local_only_translation  # scoped to this separate worker process
    headers = {'Authorization': f'Bearer {token}'}
    async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
        async with client.stream('GET', f'{CONTROL_URL}/internal/jobs/{job_id}/input/', headers=headers) as response:
            response.raise_for_status()
            raw = bytearray()
            async for chunk in response.aiter_bytes():
                raw.extend(chunk)
                if len(raw) > MAX_TRANSFER:
                    raise ValueError('Input too large')
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > 4_000_000 or max(image.size) > 4096:
                raise ValueError('Input dimensions exceeded')
        result = await run_pipeline(base64.b64encode(raw).decode(), PipelineConfig(
            source_lang=source, target_lang=target, translator='llamacpp'))
        if not result.success:
            # Native CUDA failure can poison subsequent requests; process exits
            # and RunPod replaces it. API reconciles provider FAILED/TIMED_OUT.
            if 'cuda' in (result.error or '').lower():
                os._exit(1)
            return {'success': False, 'job_id': job_id, 'error': 'pipeline_failed'}
        from app.engine.translation.quality import needs_google_fallback
        if not result.regions or any(needs_google_fallback(r.get('source_text', ''), r.get('translated_text', ''),
                                    result.detected_source_lang, target) for r in result.regions):
            return {'success': False, 'job_id': job_id, 'error': 'untranslated_or_empty'}
        translated = base64.b64decode(result.translated_b64)
        if len(translated) > MAX_TRANSFER:
            raise ValueError('Output too large')
        upload = await client.post(f'{CONTROL_URL}/internal/jobs/{job_id}/output/', headers=headers,
                                    content=translated)
        upload.raise_for_status()
        return {'success': True, 'job_id': job_id, 'sha256': upload.json()['sha256'],
                'processing_ms': result.processing_time_ms, 'region_count': len(result.regions)}


loop = asyncio.new_event_loop()


def handler(event):
    return loop.run_until_complete(process(event['input']))


if __name__ == '__main__':
    import runpod
    # Deploy with one concurrent job per worker; the pipeline shares GPU state.
    runpod.serverless.start({'handler': handler})
