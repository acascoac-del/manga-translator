import os

from django.core.asgi import get_asgi_application
from fastapi import FastAPI

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django_app = get_asgi_application()
app = FastAPI(title='MangaTranslator', docs_url=None, redoc_url=None, openapi_url=None)
app.mount('/', django_app)
