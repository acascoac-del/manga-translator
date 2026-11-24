"""Run the commercial GPU queue inside a dedicated RunPod Pod."""
import asyncio
import base64
import logging
import time
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import close_old_connections, transaction
from django.utils import timezone

from billing.dispatch import prune
from billing.models import Job
from billing.services import finish_job, normalize_image


class Command(BaseCommand):
    help = 'Consume the private Pod translation queue with one GPU process.'

    def handle(self, *args, **options):
        from app.engine.pipeline import PipelineConfig, run_pipeline
        from app.engine.translation import dispatcher
        from runpod_worker.handler import local_only_translation

        dispatcher.translate_regions = local_only_translation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # This command runs as a single supervised process per Pod.
        for job_id in Job.objects.filter(state='pod_running').values_list('id', flat=True):
            finish_job(job_id, False, 'El procesamiento se interrumpió. Crédito devuelto.')
        last_cleanup = 0
        while True:
            close_old_connections()
            if time.monotonic() - last_cleanup > 600:
                prune()
                last_cleanup = time.monotonic()
            job = self.claim() if settings.TRANSLATIONS_ENABLED else None
            if job is None:
                time.sleep(2)
                continue
            try:
                raw = (settings.DATA_DIR / job.input_path).read_bytes()
                result = loop.run_until_complete(run_pipeline(
                    base64.b64encode(raw).decode(),
                    PipelineConfig(source_lang=job.source, target_lang=job.target),
                ))
                if not result.success or not result.regions:
                    raise RuntimeError(result.error or 'No se detectó texto traducible')
                output = normalize_image(base64.b64decode(result.translated_b64), max_bytes=16 * 1024 * 1024)
                job.output_path = f'images/{job.id}.output.png'
                path = settings.DATA_DIR / job.output_path
                path.write_bytes(output)
                job.save(update_fields=['output_path'])
                finish_job(job.pk, True, processing_ms=result.processing_time_ms, region_count=len(result.regions))
            except Exception as exc:
                logging.error('Pod job %s failed (%s)', job.pk, type(exc).__name__)
                finish_job(job.pk, False, 'No se pudo completar la traducción. Crédito devuelto.')
                if 'cuda' in str(exc).lower():
                    raise SystemExit(1) from None

    @staticmethod
    @transaction.atomic
    def claim():
        job = Job.objects.select_for_update(skip_locked=True).filter(state='queued').order_by('created_at').first()
        if job:
            job.state = 'pod_running'
            job.save(update_fields=['state', 'updated_at'])
        return job
