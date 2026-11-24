import uuid
from django.conf import settings
from django.db import models


class Wallet(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.PROTECT)
    available = models.PositiveIntegerField(default=0)
    reserved = models.PositiveIntegerField(default=0)
    blocked = models.BooleanField(default=False)


class Ledger(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT)
    key = models.CharField(max_length=160, unique=True)
    kind = models.CharField(max_length=24)
    available_delta = models.IntegerField()
    reserved_delta = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)


class AccessToken(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    digest = models.CharField(max_length=64, unique=True)
    password_stamp = models.CharField(max_length=64)
    expires_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)


class Job(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT)
    request_key = models.UUIDField()
    fingerprint = models.CharField(max_length=64)
    source = models.CharField(max_length=8)
    target = models.CharField(max_length=8, default='es')
    state = models.CharField(max_length=24, default='queued', db_index=True)
    remote_id = models.CharField(max_length=160, blank=True)
    transfer_token = models.CharField(max_length=100)
    transfer_expires = models.DateTimeField()
    input_path = models.CharField(max_length=200)
    output_path = models.CharField(max_length=200, blank=True)
    output_digest = models.CharField(max_length=64, blank=True)
    error = models.CharField(max_length=240, blank=True)
    processing_ms = models.PositiveIntegerField(default=0)
    execution_ms = models.PositiveIntegerField(default=0)
    region_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [models.UniqueConstraint(fields=['user', 'request_key'], name='one_user_request')]


class Order(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT)
    amount = models.PositiveIntegerField(default=600)
    credits = models.PositiveIntegerField(default=1000)
    currency = models.CharField(max_length=3, default='usd')
    state = models.CharField(max_length=24, default='pending')
    checkout_id = models.CharField(max_length=180, blank=True)
    payment_id = models.CharField(max_length=180, blank=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)


class PaymentEvent(models.Model):
    provider_id = models.CharField(max_length=180, unique=True)
    kind = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)


class Counter(models.Model):
    key = models.CharField(max_length=180, primary_key=True)
    value = models.PositiveIntegerField(default=0)
    expires_at = models.DateTimeField()
