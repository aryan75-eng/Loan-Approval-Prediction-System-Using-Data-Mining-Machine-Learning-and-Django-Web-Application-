from django.contrib import admin

from .models import Dataset


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display  = ["id", "dataset_name", "user", "total_records", "status", "upload_date"]
    list_filter   = ["status", "upload_date"]
    search_fields = ["dataset_name", "user__username"]
    readonly_fields = ["upload_date", "total_records", "total_columns"]
