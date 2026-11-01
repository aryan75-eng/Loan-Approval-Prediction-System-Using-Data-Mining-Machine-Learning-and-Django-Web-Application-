"""
predictions/views.py — Predict, result, history, and model performance views.
"""

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.shortcuts import get_object_or_404, redirect, render

from . import ml_service
from .forms import PredictionForm
from .models import ModelPerformance, PredictionHistory


@login_required
def predict_view(request):
    """
    GET  → render blank prediction form
    POST → validate, run ml_service.predict(), save history, redirect to result
    """
    if not ml_service.is_model_loaded():
        messages.error(
            request,
            "The prediction model has not been trained yet. "
            "Please ask the administrator to run the training pipeline."
        )
        return render(request, "predictions/predict_form.html", {"form": PredictionForm(), "model_missing": True})

    if request.method == "POST":
        form = PredictionForm(request.POST)
        if form.is_valid():
            try:
                input_dict = form.to_prediction_dict()
                label, pred_proba, app_proba = ml_service.predict(input_dict)

                # Save to prediction history
                history = PredictionHistory.objects.create(
                    user                   = request.user,
                    input_data             = input_dict,
                    predicted_result       = label,
                    prediction_probability = round(pred_proba, 4),
                    approved_probability   = round(app_proba, 4),
                )

                return redirect("predictions:result", pk=history.pk)

            except FileNotFoundError:
                messages.error(request, "Model file not found. Please contact the administrator.")
            except Exception as e:
                messages.error(request, f"Prediction error: {e}")
        else:
            messages.error(request, "Please correct the errors in the form.")

    else:
        form = PredictionForm()

    return render(request, "predictions/predict_form.html", {"form": form})


@login_required
def result_view(request, pk):
    """Display the result of a single prediction."""
    prediction = get_object_or_404(PredictionHistory, pk=pk, user=request.user)
    return render(request, "predictions/result.html", {"prediction": prediction})


@login_required
def history_view(request):
    """Paginated prediction history for the logged-in user."""
    qs = PredictionHistory.objects.filter(user=request.user).order_by("-created_at")
    paginator = Paginator(qs, 15)
    page = request.GET.get("page", 1)
    predictions = paginator.get_page(page)
    return render(request, "predictions/history.html", {"predictions": predictions})


@login_required
def model_performance_view(request):
    """Display model comparison table and the selected model's metrics."""
    performances = ModelPerformance.objects.all().order_by("-f1_score")
    selected = ModelPerformance.objects.filter(is_selected=True).first()
    model_info = ml_service.get_model_info()

    context = {
        "performances": performances,
        "selected":     selected,
        "model_info":   model_info,
    }
    return render(request, "predictions/model_performance.html", context)
