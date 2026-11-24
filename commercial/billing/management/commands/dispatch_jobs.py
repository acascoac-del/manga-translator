import logging
import time
from django.core.management.base import BaseCommand
from django.db import close_old_connections
from billing.dispatch import tick, prune
from billing.models import AccessToken, Counter
from django.utils import timezone


class Command(BaseCommand):
    help = 'Process the commercial outbox. Never calls localhost or the local GPU.'

    def add_arguments(self, parser):
        parser.add_argument('--once', action='store_true')

    def handle(self, *args, **options):
        n = 0
        while True:
            close_old_connections()
            try:
                tick()
                if n % 300 == 0:
                    prune()
                    Counter.objects.filter(expires_at__lt=timezone.now()).delete()
                    AccessToken.objects.filter(expires_at__lt=timezone.now()).delete()
            except Exception:
                logging.exception('Commercial dispatcher iteration failed')
            if options['once']:
                return
            n += 1
            time.sleep(2)
