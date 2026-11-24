"""Independent control plane. Never imports the local GPU service."""
import os
from pathlib import Path
from urllib.parse import urlparse

BASE_DIR = Path(__file__).resolve().parent.parent
DEBUG = os.environ.get('COMMERCIAL_DEBUG', '0') == '1'
SECRET_KEY = os.environ.get('COMMERCIAL_SECRET_KEY', '')
if not SECRET_KEY:
    if not DEBUG:
        raise RuntimeError('Set COMMERCIAL_SECRET_KEY (do not reuse the local .env)')
    SECRET_KEY = 'development-only-never-use-this-key-in-production'
PUBLIC_URL = os.environ.get('COMMERCIAL_PUBLIC_URL', 'http://127.0.0.1:8010').rstrip('/')
ALLOWED_HOSTS = [urlparse(PUBLIC_URL).hostname, 'localhost', '127.0.0.1']
CSRF_TRUSTED_ORIGINS = [PUBLIC_URL]
if not DEBUG and (len(SECRET_KEY) < 50 or not PUBLIC_URL.startswith('https://')):
    raise RuntimeError('Production requires HTTPS and a secret of at least 50 characters')
INSTALLED_APPS = ['django.contrib.admin', 'django.contrib.auth', 'django.contrib.contenttypes',
                  'django.contrib.sessions', 'django.contrib.messages', 'django.contrib.staticfiles', 'billing']
MIDDLEWARE = ['django.middleware.security.SecurityMiddleware', 'whitenoise.middleware.WhiteNoiseMiddleware',
              'django.contrib.sessions.middleware.SessionMiddleware', 'django.middleware.common.CommonMiddleware',
              'django.middleware.csrf.CsrfViewMiddleware', 'django.contrib.auth.middleware.AuthenticationMiddleware',
              'django.contrib.messages.middleware.MessageMiddleware', 'django.middleware.clickjacking.XFrameOptionsMiddleware',
              'billing.security.SecurityHeaders']
ROOT_URLCONF = 'config.urls'
WSGI_APPLICATION = 'config.wsgi.application'
TEMPLATES = [{'BACKEND': 'django.template.backends.django.DjangoTemplates', 'DIRS': [BASE_DIR / 'templates'],
              'APP_DIRS': True, 'OPTIONS': {'context_processors': ['django.template.context_processors.request',
              'django.contrib.auth.context_processors.auth', 'django.contrib.messages.context_processors.messages']}}]
DATA_DIR = Path(os.environ.get('COMMERCIAL_DATA_DIR', BASE_DIR / 'data'))
DATA_DIR.mkdir(parents=True, exist_ok=True)
if os.environ.get('POSTGRES_HOST'):
    DATABASES = {'default': {'ENGINE': 'django.db.backends.postgresql', 'HOST': os.environ['POSTGRES_HOST'],
        'NAME': os.environ.get('POSTGRES_DB', 'manga'), 'USER': os.environ.get('POSTGRES_USER', 'manga'),
        'PASSWORD': os.environ['POSTGRES_PASSWORD'], 'CONN_MAX_AGE': 60}}
else:
    if not DEBUG:
        raise RuntimeError('Production requires PostgreSQL')
    DATABASES = {'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': DATA_DIR / 'dev.sqlite3'}}
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator', 'OPTIONS': {'min_length': 12}},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]
LANGUAGE_CODE = 'es-ar'
TIME_ZONE = 'UTC'
USE_TZ = True
STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT = BASE_DIR / 'staticfiles'
LOGIN_URL = '/login/'
LOGIN_REDIRECT_URL = '/account/'
LOGOUT_REDIRECT_URL = '/'
SESSION_COOKIE_NAME = 'manga_commercial_session'
CSRF_COOKIE_NAME = 'manga_commercial_csrf'
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SECURE = not DEBUG
CSRF_COOKIE_SECURE = not DEBUG
SESSION_COOKIE_SAMESITE = 'Lax'
SECURE_SSL_REDIRECT = not DEBUG
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')  # only Caddy may reach production web
SECURE_HSTS_SECONDS = 31536000 if not DEBUG else 0
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_REFERRER_POLICY = 'same-origin'
DATA_UPLOAD_MAX_MEMORY_SIZE = 17 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 1024 * 1024
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend' if DEBUG else 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = os.environ.get('SMTP_HOST', '')
EMAIL_PORT = int(os.environ.get('SMTP_PORT', '587'))
EMAIL_USE_TLS = os.environ.get('SMTP_TLS', '1') == '1'
EMAIL_HOST_USER = os.environ.get('SMTP_USER', '')
EMAIL_HOST_PASSWORD = os.environ.get('SMTP_PASSWORD', '')
EMAIL_TIMEOUT = 10
DEFAULT_FROM_EMAIL = os.environ.get('MAIL_FROM', 'noreply@example.invalid')
RUNPOD_API_KEY = os.environ.get('RUNPOD_API_KEY', '')
RUNPOD_ENDPOINT_ID = os.environ.get('RUNPOD_ENDPOINT_ID', '')
POD_MODE = os.environ.get('COMMERCIAL_POD_MODE', '0') == '1'
TRANSLATIONS_ENABLED = os.environ.get('TRANSLATIONS_ENABLED', '0') == '1'
PAYMENTS_ENABLED = os.environ.get('PAYMENTS_ENABLED', '0') == '1'
STRIPE_SECRET_KEY = os.environ.get('STRIPE_SECRET_KEY', '')
STRIPE_WEBHOOK_SECRET = os.environ.get('STRIPE_WEBHOOK_SECRET', '')
TRIAL_CREDITS = 50
PACK_CREDITS = 1000
PACK_CENTS = 600
PACK_CURRENCY = 'usd'
DAILY_JOBS_LIMIT = int(os.environ.get('DAILY_JOBS_LIMIT', '500'))
DAILY_TRIAL_LIMIT = int(os.environ.get('DAILY_TRIAL_LIMIT', '20'))
MAX_ACTIVE_PER_USER = 3
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 4_000_000
RETENTION_HOURS = 24
RUNPOD_EXECUTION_TIMEOUT_MS = 120_000
RUNPOD_JOB_TTL_MS = 3_600_000
