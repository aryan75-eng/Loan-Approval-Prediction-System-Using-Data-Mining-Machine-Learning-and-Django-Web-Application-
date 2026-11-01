from django.contrib import admin

from .models import ModelPerformance, PredictionHistory


@admin.register(ModelPerformance)
class ModelPerformanceAdmin(admin.ModelAdmin):
    list_display  = ["model_name", "accuracy", "precision", "recall", "f1_score", "is_selected", "training_date"]
    list_filter   = ["is_selected", "model_name"]
    readonly_fields = ["training_date"]


@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    list_display  = ["user", "predicted_result", "approved_probability", "prediction_probability", "created_at"]
    list_filter   = ["predicted_result", "created_at"]
    search_fields = ["user__username"]
    readonly_fields = ["created_at", "input_data"]
