"""
dashboard/views.py — Aggregated stats for the home dashboard.
"""

from datetime import timedelta

from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.utils import timezone

from datasets.models import Dataset
from predictions.models import ModelPerformance, PredictionHistory


@login_required
def home_view(request):
    """Render dashboard with 8 stat cards and recent prediction history."""
    user = request.user

    # ── Dataset stats (user's latest uploaded & validated dataset) ────────────
    latest_dataset = (
        Dataset.objects.filter(user=user, status=Dataset.STATUS_VALIDATED)
        .order_by("-upload_date")
        .first()
    )

    total_records  = latest_dataset.total_records if latest_dataset else 0
    total_features = latest_dataset.total_columns if latest_dataset else 0

    # Missing + duplicates require re-reading the file for accuracy.
    # We show 0 if no dataset yet (user must view preview for detail).
    missing_values = 0
    duplicate_count = 0

    # ── Model stats ───────────────────────────────────────────────────────────
    selected_model = ModelPerformance.objects.filter(is_selected=True).first()
    model_accuracy = f"{selected_model.accuracy_pct()}%" if selected_model else "N/A"

    # ── Prediction stats ─────────────────────────────────────────────────────
    user_predictions = PredictionHistory.objects.filter(user=user)
    approved_count   = user_predictions.filter(predicted_result="Approved").count()
    rejected_count   = user_predictions.filter(predicted_result="Rejected").count()

    seven_days_ago   = timezone.now() - timedelta(days=7)
    recent_count     = user_predictions.filter(created_at__gte=seven_days_ago).count()

    # ── Recent predictions table (last 10) ────────────────────────────────────
    recent_predictions = user_predictions.order_by("-created_at")[:10]

    # ── User's datasets list (last 5) ─────────────────────────────────────────
    recent_datasets = Dataset.objects.filter(user=user).order_by("-upload_date")[:5]

    context = {
        # Stat cards
        "total_records":   total_records,
        "total_features":  total_features,
        "missing_values":  missing_values,
        "duplicate_count": duplicate_count,
        "model_accuracy":  model_accuracy,
        "approved_count":  approved_count,
        "rejected_count":  rejected_count,
        "recent_count":    recent_count,
        # Tables
        "recent_predictions": recent_predictions,
        "recent_datasets":    recent_datasets,
        "selected_model":     selected_model,
        "latest_dataset":     latest_dataset,
    }

    return render(request, "dashboard/home.html", context)
