from django.contrib import admin
from .models import Job, Ledger, Order, Wallet


@admin.register(Wallet)
class WalletAdmin(admin.ModelAdmin):
    list_display = ['user', 'available', 'reserved', 'blocked']
    readonly_fields = ['user', 'available', 'reserved']

    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


class AuditAdmin(admin.ModelAdmin):
    def get_readonly_fields(self, request, obj=None):
        return [f.name for f in self.model._meta.fields]

    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(Job)
class JobAdmin(AuditAdmin):
    list_display = ['id', 'user', 'state', 'remote_id', 'created_at']
    exclude = ['transfer_token']


@admin.register(Order)
class OrderAdmin(AuditAdmin):
    list_display = ['id', 'user', 'state', 'amount', 'created_at']


@admin.register(Ledger)
class LedgerAdmin(AuditAdmin):
    list_display = ['user', 'kind', 'available_delta', 'reserved_delta', 'created_at']
