import base64
import hashlib
import io
import secrets
import uuid
import warnings
from datetime import timedelta
from PIL import Image
from django.conf import settings
from django.contrib.auth.models import User
from django.db import transaction
from django.utils import timezone
from .models import Job, Ledger, Order, Wallet
from .security import allow

ACTIVE = ['queued', 'submitting', 'dispatch_unknown', 'submitted', 'pod_running']


def normalize_image(raw, max_bytes=None):
    if not raw or len(raw) > (max_bytes or settings.MAX_IMAGE_BYTES):
        raise ValueError('La imagen debe pesar hasta 5 MB.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format not in {'PNG', 'JPEG', 'WEBP'} or getattr(image, 'n_frames', 1) != 1:
                    raise ValueError('Usá una imagen estática PNG, JPEG o WebP.')
                if image.width * image.height > settings.MAX_IMAGE_PIXELS or max(image.size) > 4096:
                    raise ValueError('Máximo 4 megapíxeles y 4096 píxeles por lado.')
                image.load()
                output = io.BytesIO()
                image.convert('RGB').save(output, format='PNG')
                return output.getvalue()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning, OSError) as exc:
        raise ValueError('Imagen inválida o demasiado grande.') from exc


def image_from_base64(value):
    if not isinstance(value, str) or len(value) > 7_000_000:
        raise ValueError('Imagen inválida o demasiado grande.')
    try:
        return normalize_image(base64.b64decode(value.split(',', 1)[-1], validate=True))
    except (ValueError, TypeError) as exc:
        raise ValueError(str(exc) or 'Base64 inválido.') from exc


@transaction.atomic
def verify_user(user_id):
    user = User.objects.select_for_update().get(pk=user_id)
    wallet, _ = Wallet.objects.get_or_create(user=user)
    if user.is_active:
        return user
    if Ledger.objects.filter(key=f'trial:{user.pk}').exists():
        raise ValueError('La cuenta requiere revisión.')
    if not allow('trial-grants', settings.DAILY_TRIAL_LIMIT, 86400):
        raise ValueError('Se alcanzó el cupo diario de pruebas. Intentá mañana.')
    user.is_active = True
    user.save(update_fields=['is_active'])
    wallet.available += settings.TRIAL_CREDITS
    wallet.save()
    Ledger.objects.create(user=user, key=f'trial:{user.pk}', kind='trial', available_delta=settings.TRIAL_CREDITS)
    return user


@transaction.atomic
def reserve_job(user, raw, source, target, request_key):
    if source not in {'auto', 'en', 'ja', 'ko', 'zh'} or target != 'es':
        raise ValueError('Idioma no admitido; esta beta traduce al español.')
    request_key = uuid.UUID(str(request_key))
    wallet = Wallet.objects.select_for_update().get(user=user)
    fingerprint = hashlib.sha256(raw + f'|{source}|{target}|pipeline-v1'.encode()).hexdigest()
    previous = Job.objects.filter(user=user, request_key=request_key).first()
    if previous:
        if previous.fingerprint != fingerprint:
            raise ValueError('La clave de solicitud ya corresponde a otra imagen.')
        return previous
    cached = Job.objects.filter(user=user, fingerprint=fingerprint, state='completed',
                                created_at__gt=timezone.now() - timedelta(hours=settings.RETENTION_HOURS)).first()
    if cached and (settings.DATA_DIR / cached.output_path).is_file():
        return cached
    if wallet.blocked:
        raise ValueError('Cuenta en revisión. Contactá soporte.')
    if wallet.available < 1:
        raise ValueError('No quedan créditos disponibles.')
    if Job.objects.filter(user=user, state__in=ACTIVE).count() >= settings.MAX_ACTIVE_PER_USER:
        raise ValueError('Ya tenés tres imágenes pendientes. Esperá a que terminen.')
    if not allow('jobs-global', settings.DAILY_JOBS_LIMIT, 86400):
        raise ValueError('Límite diario del servicio alcanzado. No se cobró esta imagen.')
    job = Job(user=user, source=source, target=target, fingerprint=fingerprint, request_key=request_key,
              transfer_token=secrets.token_urlsafe(48), transfer_expires=timezone.now() + timedelta(hours=3))
    job.input_path = f'images/{job.id}.input.png'
    path = settings.DATA_DIR / job.input_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    job.save()
    wallet.available -= 1
    wallet.reserved += 1
    wallet.save()
    Ledger.objects.create(user=user, key=f'reserve:{job.id}', kind='reserve', available_delta=-1, reserved_delta=1)
    return job


@transaction.atomic
def finish_job(job_id, success, error='', processing_ms=0, execution_ms=0, region_count=0):
    owner = Job.objects.only('user_id').get(pk=job_id).user_id
    wallet = Wallet.objects.select_for_update().get(user_id=owner)
    job = Job.objects.select_for_update().get(pk=job_id)
    if job.state not in ACTIVE:
        return job
    if success and (not job.output_path or not (settings.DATA_DIR / job.output_path).is_file()):
        raise ValueError('Falta el archivo de salida; no se puede cobrar.')
    wallet.reserved -= 1
    if not success:
        wallet.available += 1
    wallet.save()
    Ledger.objects.create(user_id=owner, key=f'finish:{job.id}', kind='consume' if success else 'release',
                          available_delta=0 if success else 1, reserved_delta=-1)
    job.state = 'completed' if success else 'failed'
    job.error = error[:240]
    job.processing_ms = max(0, int(processing_ms))
    job.execution_ms = max(0, int(execution_ms))
    job.region_count = max(0, int(region_count))
    job.save()
    return job


@transaction.atomic
def credit_order(order_id, amount, currency, payment_id):
    owner = Order.objects.only('user_id').get(pk=order_id).user_id
    wallet = Wallet.objects.select_for_update().get(user_id=owner)
    order = Order.objects.select_for_update().get(pk=order_id)
    if order.amount != amount or order.currency != currency or not payment_id:
        raise ValueError('El pago no coincide con la orden.')
    if order.state != 'pending':
        return
    order.state = 'paid'
    order.payment_id = payment_id
    order.save()
    wallet.available += order.credits
    wallet.save()
    Ledger.objects.create(user_id=owner, key=f'purchase:{order.id}', kind='purchase', available_delta=order.credits)
