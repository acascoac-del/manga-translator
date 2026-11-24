from django.contrib import admin
from django.contrib.auth import views as auth
from django.urls import path
from billing import views as v

urlpatterns = [
    path('', v.home, name='home'), path('health/', v.health),
    path('register/', v.register, name='register'), path('verify/<str:token>/', v.verify),
    path('resend/', v.resend_verification), path('login/', v.LoginView.as_view(), name='login'),
    path('logout/', auth.LogoutView.as_view(), name='logout'),
    path('reset/', v.PasswordResetView.as_view(), name='password_reset'),
    path('reset/sent/', auth.PasswordResetDoneView.as_view(template_name='notice.html', extra_context={'message': 'Si la cuenta existe y está activa, recibirás un enlace por correo.'}), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', auth.PasswordResetConfirmView.as_view(template_name='form.html', extra_context={'title': 'Nueva contraseña', 'button': 'Guardar'}), name='password_reset_confirm'),
    path('reset/complete/', auth.PasswordResetCompleteView.as_view(template_name='notice.html', extra_context={'message': 'Contraseña actualizada. Iniciá sesión otra vez.'}), name='password_reset_complete'),
    path('account/', v.account, name='account'), path('account/token/', v.issue_token),
    path('account/revoke/', v.revoke_tokens), path('account/download/<uuid:job_id>/', v.download),
    path('checkout/', v.checkout), path('webhooks/stripe/', v.stripe_webhook),
    path('privacy/', v.policy, {'page': 'privacy'}), path('terms/', v.policy, {'page': 'terms'}),
    path('api/balance/', v.balance), path('api/jobs/', v.create_job), path('api/jobs/<uuid:job_id>/', v.job_status),
    path('internal/jobs/<uuid:job_id>/input/', v.worker_input),
    path('internal/jobs/<uuid:job_id>/output/', v.worker_output), path('admin/', admin.site.urls),
]
