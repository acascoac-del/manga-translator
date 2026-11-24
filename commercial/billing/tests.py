import base64
import hashlib
import hmac
import io
import json
import tempfile
import time
import uuid
from datetime import timedelta
from unittest.mock import patch
import httpx
from PIL import Image
from django.conf import settings
from django.contrib.auth.models import User
from django.core import mail, signing
from django.test import TestCase, Client, override_settings
from django.utils import timezone
from .models import AccessToken, Job, Ledger, Order, Wallet
from .security import digest
from .services import reserve_job, finish_job, credit_order, normalize_image, verify_user
from .dispatch import tick


def picture():
    output = io.BytesIO()
    Image.new('RGB', (64, 64), 'white').save(output, 'PNG')
    return output.getvalue()


class CommercialTests(TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        from pathlib import Path
        self.settings_override = override_settings(DATA_DIR=Path(self.directory.name),
            ALLOWED_HOSTS=['testserver'], EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend')
        self.settings_override.enable()
        self.addCleanup(self.settings_override.disable)
        self.addCleanup(self.directory.cleanup)
        self.user = User.objects.create_user('reader@example.com', email='reader@example.com', password='SecureTest-Passphrase97', is_active=False)
        Wallet.objects.create(user=self.user)
        verify_user(self.user.pk)
        self.user.refresh_from_db()
        self.raw = picture()
        self.token = 'test-token-' + 'a' * 48
        AccessToken.objects.create(user=self.user, digest=digest(self.token), password_stamp=digest(self.user.password),
                                    expires_at=timezone.now() + timedelta(days=1))
        self.headers = {'HTTP_AUTHORIZATION': 'Bearer ' + self.token}

    def job(self):
        return reserve_job(self.user, self.raw, 'en', 'es', uuid.uuid4())

    def test_trial_only_once(self):
        verify_user(self.user.pk)
        self.assertEqual(Wallet.objects.get(user=self.user).available, 50)
        self.assertEqual(Ledger.objects.filter(kind='trial').count(), 1)

    def test_verification_required_before_login(self):
        user = User.objects.create_user('waiting@example.com', password='SecureTest-Passphrase97', is_active=False)
        self.assertFalse(self.client.login(username=user.username, password='SecureTest-Passphrase97'))

    def test_registration_sends_verification_not_credits(self):
        response = self.client.post('/register/', {'email': 'NEW@example.com', 'password1': 'SecureSecond-Passphrase37', 'password2': 'SecureSecond-Passphrase37'})
        self.assertEqual(response.status_code, 200)
        user = User.objects.get(username='new@example.com')
        self.assertFalse(user.is_active)
        self.assertEqual(user.wallet.available, 0)
        self.assertEqual(len(mail.outbox), 1)

    def test_signed_verification(self):
        user = User.objects.create_user('verify@example.com', email='verify@example.com', is_active=False)
        token = signing.dumps({'user': user.pk, 'email': user.email}, salt='verify-email')
        self.assertEqual(self.client.get('/verify/' + token + '/').status_code, 302)
        self.assertEqual(Wallet.objects.get(user=user).available, 50)

    def test_reserve_idempotent_and_no_overspend(self):
        key = uuid.uuid4()
        first = reserve_job(self.user, self.raw, 'en', 'es', key)
        second = reserve_job(self.user, self.raw, 'en', 'es', key)
        self.assertEqual(first.pk, second.pk)
        wallet = Wallet.objects.get(user=self.user)
        self.assertEqual((wallet.available, wallet.reserved), (49, 1))
        Wallet.objects.filter(user=self.user).update(available=0)
        with self.assertRaises(ValueError): self.job()

    def test_idempotency_cannot_change_input(self):
        job = self.job()
        with self.assertRaises(ValueError): reserve_job(self.user, self.raw, 'ja', 'es', job.request_key)

    def test_failure_releases_once(self):
        job = self.job()
        finish_job(job.pk, False)
        finish_job(job.pk, False)
        wallet = Wallet.objects.get(user=self.user)
        self.assertEqual((wallet.available, wallet.reserved), (50, 0))

    def test_success_requires_file_then_captures_once(self):
        job = self.job()
        with self.assertRaises(ValueError): finish_job(job.pk, True)
        job.output_path = f'images/{job.pk}.output.png'
        (settings.DATA_DIR / job.output_path).write_bytes(self.raw)
        job.save()
        finish_job(job.pk, True)
        finish_job(job.pk, True)
        wallet = Wallet.objects.get(user=self.user)
        self.assertEqual((wallet.available, wallet.reserved), (49, 0))

    def test_private_job(self):
        other = User.objects.create_user('other')
        Wallet.objects.create(user=other, available=3)
        job = reserve_job(other, self.raw, 'en', 'es', uuid.uuid4())
        self.assertEqual(self.client.get(f'/api/jobs/{job.pk}/', **self.headers).status_code, 404)

    def test_cookie_cannot_bypass_bearer_api(self):
        self.client.force_login(self.user)
        self.assertEqual(self.client.get('/api/balance/').status_code, 401)

    def test_password_change_invalidates_extension_token(self):
        self.user.set_password('NewPassword47-safe'); self.user.save()
        self.assertEqual(self.client.get('/api/balance/', **self.headers).status_code, 401)

    def test_disabled_cloud_never_reserves_or_calls_provider(self):
        with patch('httpx.Client.post') as network:
            response = self.client.post('/api/jobs/', json.dumps({'image':base64.b64encode(self.raw).decode(), 'request_key':str(uuid.uuid4())}), content_type='application/json', **self.headers)
            self.assertEqual(response.status_code, 503)
            network.assert_not_called()
        self.assertEqual(Job.objects.count(), 0)

    def test_image_limits(self):
        with self.assertRaises(ValueError): normalize_image(b'not-an-image')
        with self.assertRaises(ValueError): normalize_image(b'a' * (5 * 1024 * 1024 + 1))
        large = io.BytesIO(); Image.new('RGB', (2500, 2000)).save(large, 'PNG')
        with self.assertRaises(ValueError): normalize_image(large.getvalue())

    def test_worker_upload_private_and_not_enough_to_charge(self):
        job = self.job()
        url = f'/internal/jobs/{job.pk}/output/'
        self.assertEqual(self.client.post(url, self.raw, content_type='image/png').status_code, 403)
        response = self.client.post(url, self.raw, content_type='image/png', HTTP_AUTHORIZATION='Bearer ' + job.transfer_token)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(Wallet.objects.get(user=self.user).reserved, 1)

    def test_purchase_idempotent_and_validates_amount(self):
        order = Order.objects.create(user=self.user)
        with self.assertRaises(ValueError): credit_order(order.pk, 1, 'usd', 'pi_example')
        credit_order(order.pk, 600, 'usd', 'pi_example')
        credit_order(order.pk, 600, 'usd', 'pi_example')
        self.assertEqual(Wallet.objects.get(user=self.user).available, 1050)

    @override_settings(STRIPE_WEBHOOK_SECRET='whsec_test_not_real')
    def test_signed_payment_webhook_and_duplicate(self):
        order = Order.objects.create(user=self.user, checkout_id='cs_test')
        payload = json.dumps({'id':'evt_paid', 'type':'checkout.session.completed', 'data':{'object':{
            'id':'cs_test', 'mode':'payment', 'metadata':{'order_id':str(order.pk)}, 'amount_total':600,
            'currency':'usd', 'payment_intent':'pi_test', 'payment_status':'paid'}}})
        timestamp = int(time.time())
        signature = hmac.new(b'whsec_test_not_real', f'{timestamp}.{payload}'.encode(), hashlib.sha256).hexdigest()
        for _ in range(2):
            response = self.client.post('/webhooks/stripe/', payload, content_type='application/json', HTTP_STRIPE_SIGNATURE=f't={timestamp},v1={signature}')
            self.assertEqual(response.status_code, 200)
        self.assertEqual(Wallet.objects.get(user=self.user).available, 1050)
        self.assertEqual(self.client.post('/webhooks/stripe/', payload, content_type='application/json').status_code, 400)

    def test_checkout_disabled(self):
        self.client.force_login(self.user)
        self.assertEqual(self.client.post('/checkout/').status_code, 503)
        self.assertEqual(Order.objects.count(), 0)

    @override_settings(TRANSLATIONS_ENABLED=True, RUNPOD_API_KEY='test', RUNPOD_ENDPOINT_ID='test')
    def test_ambiguous_submission_not_automatically_retried(self):
        job = self.job()
        with patch('httpx.Client.post', side_effect=httpx.ReadTimeout('simulated')) as submit:
            tick(); tick()
            self.assertEqual(submit.call_count, 1)
        job.refresh_from_db()
        self.assertEqual(job.state, 'dispatch_unknown')
        self.assertEqual(Wallet.objects.get(user=self.user).reserved, 1)

    @override_settings(TRANSLATIONS_ENABLED=True, RUNPOD_API_KEY='test', RUNPOD_ENDPOINT_ID='test')
    def test_provider_completion_captures_uploaded_result(self):
        job = self.job()
        self.client.post(f'/internal/jobs/{job.pk}/output/', self.raw, content_type='image/png', HTTP_AUTHORIZATION='Bearer ' + job.transfer_token)
        job.refresh_from_db(); job.state='submitted'; job.remote_id='remote-test'; job.save()
        payload = {'status':'COMPLETED','executionTime':3000,'output':{'success':True,'job_id':str(job.pk), 'sha256':job.output_digest,'region_count':2}}
        response = httpx.Response(200, json=payload, request=httpx.Request('GET','https://api.runpod.ai/test'))
        with patch('httpx.Client.get', return_value=response): tick(); tick()
        job.refresh_from_db()
        self.assertEqual(job.state, 'completed')
        self.assertEqual((Wallet.objects.get(user=self.user).available, Wallet.objects.get(user=self.user).reserved), (49,0))

    def test_csrf_required_on_web_token_creation(self):
        client = Client(enforce_csrf_checks=True); client.force_login(self.user)
        self.assertEqual(client.post('/account/token/').status_code, 403)

    def test_pages_render(self):
        for url in ['/', '/login/', '/register/', '/privacy/', '/terms/', '/reset/']:
            self.assertEqual(self.client.get(url).status_code, 200, url)
        self.client.force_login(self.user)
        self.assertEqual(self.client.get('/account/').status_code, 200)
