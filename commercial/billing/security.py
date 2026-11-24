import hashlib
from datetime import timedelta
from functools import wraps
from django.db import transaction
from django.http import JsonResponse
from django.utils import timezone
from .models import AccessToken, Counter


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def allow(key, limit, seconds=3600):
    """Shared DB fixed-window limiter. Never trust client-supplied forwarding headers."""
    now = timezone.now()
    bucket = int(now.timestamp()) // seconds
    with transaction.atomic():
        row, _ = Counter.objects.get_or_create(key=f'{key}:{bucket}', defaults={'expires_at': now + timedelta(seconds=seconds * 2)})
        row = Counter.objects.select_for_update().get(pk=row.pk)
        if row.value >= limit:
            return False
        row.value += 1
        row.save(update_fields=['value'])
        return True


def bearer(view):
    @wraps(view)
    def wrapped(request, *args, **kwargs):
        header = request.headers.get('Authorization', '')
        if not header.startswith('Bearer ') or len(header) > 200:
            return JsonResponse({'error': 'Iniciá sesión en la extensión.'}, status=401)
        token = AccessToken.objects.select_related('user').filter(digest=digest(header[7:]), expires_at__gt=timezone.now()).first()
        if not token or not token.user.is_active or token.password_stamp != digest(token.user.password):
            return JsonResponse({'error': 'Sesión vencida o revocada.'}, status=401)
        request.api_user = token.user
        return view(request, *args, **kwargs)
    return wrapped


class SecurityHeaders:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        response['Cache-Control'] = 'no-store'
        # No remote scripts, no inline scripts. Django admin uses inline styling.
        response['Content-Security-Policy'] = "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        return response
