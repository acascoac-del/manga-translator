"""Durable outbox consumer. No automatic retry of ambiguous paid submissions."""
import logging
from datetime import timedelta
import httpx
from django.conf import settings
from django.db import transaction
from django.utils import timezone
from .models import Job
from .services import ACTIVE, finish_job

log = logging.getLogger(__name__)


def tick():
    if not settings.TRANSLATIONS_ENABLED:
        return
    if not settings.RUNPOD_API_KEY or not settings.RUNPOD_ENDPOINT_ID:
        raise RuntimeError('RunPod is enabled but credentials are missing')
    base = f'https://api.runpod.ai/v2/{settings.RUNPOD_ENDPOINT_ID}'
    headers = {'Authorization': f'Bearer {settings.RUNPOD_API_KEY}'}
    # A crash after /run was accepted must not cause a second paid GPU job.
    Job.objects.filter(state='submitting', updated_at__lt=timezone.now() - timedelta(minutes=2)).update(
        state='dispatch_unknown', error='Envío pendiente de conciliación; no se volverá a enviar automáticamente.')
    with httpx.Client(timeout=20, headers=headers) as client:
        # Reconcile first so long queues do not delay finished credits.
        for job in Job.objects.filter(state='submitted').order_by('updated_at')[:100]:
            try:
                response = client.get(f'{base}/status/{job.remote_id}')
                response.raise_for_status()
                status = response.json()
                state = status.get('status')
                if state == 'COMPLETED':
                    result = status.get('output') or {}
                    job.refresh_from_db()
                    valid = (isinstance(result, dict) and result.get('success') is True and
                             result.get('job_id') == str(job.pk) and bool(job.output_digest) and
                             result.get('sha256') == job.output_digest)
                    finish_job(job.pk, valid, '' if valid else 'No se pudo completar la traducción. Crédito devuelto.',
                        result.get('processing_ms', 0) if isinstance(result, dict) else 0,
                        status.get('executionTime', 0) or 0,
                        result.get('region_count', 0) if isinstance(result, dict) else 0)
                elif state in {'FAILED', 'CANCELLED', 'TIMED_OUT'}:
                    finish_job(job.pk, False, 'El procesamiento falló. Crédito devuelto.')
            except (httpx.HTTPError, ValueError, TypeError):
                log.warning('Could not reconcile job %s; keeping reservation', job.pk)
        with transaction.atomic():
            job = Job.objects.select_for_update(skip_locked=True).filter(state='queued').order_by('created_at').first()
            if not job:
                return
            job.state = 'submitting'
            job.transfer_expires = timezone.now() + timedelta(hours=3)
            job.save()
        payload = {'input': {'job_id': str(job.pk), 'token': job.transfer_token,
                            'source': job.source, 'target': job.target},
                   'policy': {'executionTimeout': settings.RUNPOD_EXECUTION_TIMEOUT_MS,
                              'ttl': settings.RUNPOD_JOB_TTL_MS}}
        try:
            response = client.post(f'{base}/run', json=payload)
            # Even an HTTP error may be produced after remote acceptance. Fail
            # closed: operator reconciliation, not blind resubmission.
            response.raise_for_status()
            remote_id = response.json()['id']
            if not isinstance(remote_id, str) or not remote_id or len(remote_id) > 160:
                raise ValueError('Invalid remote id')
            Job.objects.filter(pk=job.pk, state='submitting').update(state='submitted', remote_id=remote_id, updated_at=timezone.now())
        except (httpx.HTTPError, ValueError, KeyError):
            Job.objects.filter(pk=job.pk, state='submitting').update(state='dispatch_unknown',
                error='No se pudo confirmar el envío. El saldo queda reservado para revisión.', updated_at=timezone.now())


def prune():
    """Remove only expired image artifacts, never financial history or active jobs."""
    cutoff = timezone.now() - timedelta(hours=settings.RETENTION_HOURS)
    for job in Job.objects.filter(state__in=['completed', 'failed'], updated_at__lt=cutoff):
        for name in [job.input_path, job.output_path]:
            if name:
                path = (settings.DATA_DIR / name).resolve()
                root = (settings.DATA_DIR / 'images').resolve()
                if path.parent == root:
                    path.unlink(missing_ok=True)
    # Files written before a failed SQL commit can be orphans; purge only old,
    # known image suffixes in the private images directory.
    root = settings.DATA_DIR / 'images'
    if root.exists():
        for path in root.glob('*.png'):
            if path.stat().st_mtime < cutoff.timestamp():
                job_id = path.name.split('.')[0]
                if not Job.objects.filter(pk=job_id, state__in=ACTIVE).exists():
                    path.unlink(missing_ok=True)
