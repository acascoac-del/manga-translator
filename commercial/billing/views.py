import base64
import hashlib
import io
import json
import secrets
from datetime import timedelta
from pathlib import Path
import httpx
import stripe
from PIL import Image
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import views as auth_views
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.core import signing
from django.core.mail import send_mail
from django.db import IntegrityError, transaction
from django.http import FileResponse, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST
from .forms import RegistrationForm
from .models import AccessToken, Job, Ledger, Order, PaymentEvent, Wallet
from .security import allow, bearer, digest
from .services import ACTIVE, credit_order, image_from_base64, normalize_image, reserve_job, verify_user


def home(request):
    return render(request, 'home.html', {'payments_enabled': settings.PAYMENTS_ENABLED})


def policy(request, page):
    return render(request, 'policy.html', {'page': page})


def register(request):
    form = RegistrationForm(request.POST or None)
    if request.method == 'POST':
        # Proxy IP is not accepted from the public client. A global limit also
        # prevents unlimited email dispatch when all clients share a proxy IP.
        if not allow('register:' + digest(request.META.get('REMOTE_ADDR', '')), 10):
            return HttpResponse('Demasiados intentos. Intentá más tarde.', status=429)
        if form.is_valid():
            try:
                with transaction.atomic():
                    user = form.save()
                    Wallet.objects.create(user=user)
            except IntegrityError:
                form.add_error('email', 'No se pudo registrar el correo.')
            else:
                _send_verification(user)
                return render(request, 'notice.html', {'message': 'Revisá tu correo para activar la cuenta y recibir tus 50 créditos.'})
    return render(request, 'form.html', {'form': form, 'title': 'Empezá con 50 imágenes gratis', 'button': 'Crear cuenta'})


def _send_verification(user):
    token = signing.dumps({'user': user.pk, 'email': user.email}, salt='verify-email')
    url = f'{settings.PUBLIC_URL}/verify/{token}/'
    send_mail('Activá MangaTranslator', f'Confirmá tu correo (válido por 24 horas):\n{url}',
              settings.DEFAULT_FROM_EMAIL, [user.email])


@require_POST
def resend_verification(request):
    if not allow('resend:' + digest(request.META.get('REMOTE_ADDR', '')), 5):
        return HttpResponse(status=429)
    email = request.POST.get('email', '').strip().lower()
    user = User.objects.filter(username=email, is_active=False).first()
    if user:
        _send_verification(user)
    return render(request, 'notice.html', {'message': 'Si el correo tiene una cuenta pendiente, enviamos un nuevo enlace.'})


def verify(request, token):
    try:
        payload = signing.loads(token, salt='verify-email', max_age=86400)
        user = User.objects.get(pk=payload['user'], email=payload['email'])
        verify_user(user.pk)
    except (signing.BadSignature, User.DoesNotExist, ValueError, KeyError) as exc:
        return render(request, 'notice.html', {'message': 'No se pudo activar la cuenta: enlace vencido o cupo diario completo. Solicitá otro enlace o intentá mañana.'}, status=400)
    messages.success(request, 'Cuenta verificada. Ya podés iniciar sesión.')
    return redirect('login')


class LoginView(auth_views.LoginView):
    template_name = 'form.html'
    extra_context = {'title': 'Entrá a tu cuenta', 'button': 'Ingresar'}

    def post(self, request, *args, **kwargs):
        identity = request.POST.get('username', '').lower().strip()
        if (not allow('login-ip:' + digest(request.META.get('REMOTE_ADDR', '')), 30, 900)
                or not allow('login-user:' + digest(identity), 10, 900)):
            return HttpResponse('Demasiados intentos. Esperá 15 minutos.', status=429)
        return super().post(request, *args, **kwargs)


class PasswordResetView(auth_views.PasswordResetView):
    template_name = 'form.html'
    email_template_name = 'registration/password_reset_email.html'
    extra_context = {'title': 'Recuperar contraseña', 'button': 'Enviar enlace'}

    def post(self, request, *args, **kwargs):
        if not allow('reset:' + digest(request.META.get('REMOTE_ADDR', '')), 5):
            return HttpResponse(status=429)
        return super().post(request, *args, **kwargs)


@login_required
def account(request):
    wallet, _ = Wallet.objects.get_or_create(user=request.user)
    return render(request, 'account.html', {'wallet': wallet, 'jobs': Job.objects.filter(user=request.user).order_by('-created_at')[:30],
        'ledger': Ledger.objects.filter(user=request.user).order_by('-created_at')[:15],
        'payments_enabled': settings.PAYMENTS_ENABLED, 'translations_enabled': settings.TRANSLATIONS_ENABLED})


@login_required
@require_POST
def issue_token(request):
    if not allow(f'token:{request.user.pk}', 10):
        return HttpResponse(status=429)
    token = secrets.token_urlsafe(48)
    AccessToken.objects.filter(user=request.user).delete()
    AccessToken.objects.create(user=request.user, digest=digest(token), password_stamp=digest(request.user.password),
                                expires_at=timezone.now() + timedelta(days=7))
    return render(request, 'token.html', {'token': token, 'origin': settings.PUBLIC_URL})


@login_required
@require_POST
def revoke_tokens(request):
    AccessToken.objects.filter(user=request.user).delete()
    messages.success(request, 'Acceso de la extensión revocado.')
    return redirect('account')


@require_GET
def health(request):
    return JsonResponse({'status': 'ok', 'service': 'manga-commercial', 'translations_enabled': settings.TRANSLATIONS_ENABLED,
                         'payments_enabled': settings.PAYMENTS_ENABLED})


@bearer
@require_GET
def balance(request):
    wallet = Wallet.objects.get(user=request.api_user)
    return JsonResponse({'available': wallet.available, 'reserved': wallet.reserved, 'blocked': wallet.blocked})


@csrf_exempt  # Cookie sessions are NOT accepted here; bearer token required.
@bearer
@require_POST
def create_job(request):
    if not settings.TRANSLATIONS_ENABLED or (
        not settings.POD_MODE and (not settings.RUNPOD_API_KEY or not settings.RUNPOD_ENDPOINT_ID)
    ):
        return JsonResponse({'error': 'Traducción comercial todavía no habilitada. No se descontaron créditos.'}, status=503)
    if not allow(f'job-request:{request.api_user.pk}', 30, 60):
        return JsonResponse({'error': 'Demasiadas solicitudes.'}, status=429)
    try:
        body = json.loads(request.body)
        raw = image_from_base64(body['image'])
        job = reserve_job(request.api_user, raw, body.get('source', 'auto'), body.get('target', 'es'), body['request_key'])
    except (ValueError, KeyError, TypeError) as exc:
        return JsonResponse({'error': str(exc)[:240]}, status=400)
    return JsonResponse({'id': str(job.pk), 'state': job.state}, status=202)


@bearer
@require_GET
def job_status(request, job_id):
    job = get_object_or_404(Job, pk=job_id, user=request.api_user)
    data = {'id': str(job.pk), 'state': job.state, 'error': job.error, 'processing_time_ms': job.processing_ms,
            'region_count': job.region_count}
    if job.state == 'completed':
        path = settings.DATA_DIR / job.output_path
        if path.is_file():
            data['translated_image'] = base64.b64encode(path.read_bytes()).decode()
        else:
            data['error'] = 'El resultado venció. Los archivos se conservan 24 horas.'
    return JsonResponse(data)


@login_required
@require_GET
def download(request, job_id):
    job = get_object_or_404(Job, pk=job_id, user=request.user, state='completed')
    path = settings.DATA_DIR / job.output_path
    if not path.is_file():
        return HttpResponse('Archivo vencido.', status=410)
    return FileResponse(path.open('rb'), content_type='image/png', as_attachment=True, filename='traducida.png')


def _transfer_job(request, job_id):
    job = get_object_or_404(Job, pk=job_id)
    token = request.headers.get('Authorization', '').removeprefix('Bearer ')
    if (not secrets.compare_digest(token, job.transfer_token) or job.transfer_expires < timezone.now()
            or job.state not in ACTIVE):
        return None
    return job


@require_GET
def worker_input(request, job_id):
    job = _transfer_job(request, job_id)
    if not job:
        return HttpResponse(status=403)
    return FileResponse((settings.DATA_DIR / job.input_path).open('rb'), content_type='image/png')


@csrf_exempt
@require_POST
def worker_output(request, job_id):
    job = _transfer_job(request, job_id)
    if not job:
        return HttpResponse(status=403)
    try:
        raw = normalize_image(request.body, max_bytes=16 * 1024 * 1024)
        with Image.open(settings.DATA_DIR / job.input_path) as original, Image.open(io.BytesIO(raw)) as output:
            if original.size != output.size:
                raise ValueError('Dimensiones de salida incorrectas.')
    except ValueError as exc:
        return JsonResponse({'error': str(exc)}, status=400)
    with transaction.atomic():
        locked = Job.objects.select_for_update().get(pk=job.pk)
        if locked.state not in ACTIVE:
            return HttpResponse(status=409)
        if not locked.output_path:
            locked.output_path = f'images/{locked.id}.output.png'
            (settings.DATA_DIR / locked.output_path).write_bytes(raw)
            locked.output_digest = hashlib.sha256(raw).hexdigest()
            locked.save()
    return JsonResponse({'stored': True, 'sha256': locked.output_digest})


@login_required
@require_POST
def checkout(request):
    if not settings.PAYMENTS_ENABLED or not settings.STRIPE_SECRET_KEY or not settings.STRIPE_WEBHOOK_SECRET:
        return render(request, 'notice.html', {'message': 'Los cobros aún no están habilitados.'}, status=503)
    if not allow(f'checkout:{request.user.pk}', 5):
        return HttpResponse(status=429)
    if Wallet.objects.get(user=request.user).blocked:
        return HttpResponse('Cuenta en revisión.', status=403)
    order = Order.objects.create(user=request.user, amount=settings.PACK_CENTS, credits=settings.PACK_CREDITS, currency=settings.PACK_CURRENCY)
    session = stripe.checkout.Session.create(api_key=settings.STRIPE_SECRET_KEY,
        mode='payment', client_reference_id=str(order.pk), metadata={'order_id': str(order.pk)},
        line_items=[{'price_data': {'currency': order.currency, 'unit_amount': order.amount,
            'product_data': {'name': '1000 imágenes · MangaTranslator'}}, 'quantity': 1}],
        success_url=settings.PUBLIC_URL + '/account/?payment=processing', cancel_url=settings.PUBLIC_URL + '/account/',
        idempotency_key=f'checkout:{order.pk}')
    order.checkout_id = session.id
    order.save()
    return redirect(session.url)


@csrf_exempt
@require_POST
def stripe_webhook(request):
    if not settings.STRIPE_WEBHOOK_SECRET:
        return HttpResponse(status=503)
    try:
        event = stripe.Webhook.construct_event(request.body, request.headers.get('Stripe-Signature', ''), settings.STRIPE_WEBHOOK_SECRET)
    except (ValueError, stripe.SignatureVerificationError):
        return HttpResponse(status=400)
    kind = event['type']
    obj = event['data']['object']
    with transaction.atomic():
        _, created = PaymentEvent.objects.get_or_create(provider_id=event['id'], defaults={'kind': kind})
        if not created:
            return HttpResponse(status=200)
        if kind in {'checkout.session.completed', 'checkout.session.async_payment_succeeded'} and obj.get('payment_status') == 'paid':
            order_id = obj.get('metadata', {}).get('order_id')
            if not order_id:
                return HttpResponse(status=200)  # unrelated product in the same Stripe account
            order = Order.objects.get(pk=order_id)
            if obj.get('mode') != 'payment' or (order.checkout_id and order.checkout_id != obj['id']):
                raise ValueError('Checkout incorrecto')
            credit_order(order.pk, obj['amount_total'], obj['currency'], obj['payment_intent'])
        elif kind in {'charge.refunded', 'charge.dispute.created'}:
            # Freeze rather than silently subtract from unrelated purchases.
            # Staff must reconcile refunds/consumed credits before unblocking.
            for order in Order.objects.filter(payment_id=obj.get('payment_intent', '')):
                Wallet.objects.filter(user=order.user).update(blocked=True)
                order.state = 'review'
                order.save()
    return HttpResponse(status=200)
