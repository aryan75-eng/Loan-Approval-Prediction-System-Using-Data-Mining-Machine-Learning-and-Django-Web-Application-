"""
predictions/models.py — ModelPerformance and PredictionHistory.
"""

from django.contrib.auth.models import User
from django.db import models


class ModelPerformance(models.Model):
    """Stores ML model evaluation metrics. Populated by load_model_metrics command."""

    dataset     = models.ForeignKey(
        "datasets.Dataset",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="model_performances",
    )
    model_name    = models.CharField(max_length=50)
    accuracy      = models.FloatField()
    precision     = models.FloatField()
    recall        = models.FloatField()
    f1_score      = models.FloatField()
    cv_mean       = models.FloatField(default=0.0)
    cv_std        = models.FloatField(default=0.0)
    training_date = models.DateTimeField(auto_now_add=True)
    is_selected   = models.BooleanField(default=False)

    class Meta:
        verbose_name        = "Model Performance"
        verbose_name_plural = "Model Performances"
        unique_together     = [("dataset", "model_name")]
        indexes = [
            models.Index(fields=["is_selected"]),
        ]

    def __str__(self):
        selected = " ★" if self.is_selected else ""
        return f"{self.model_name}{selected} (F1={self.f1_score:.4f})"

    def accuracy_pct(self):
        return round(self.accuracy * 100, 2)

    def precision_pct(self):
        return round(self.precision * 100, 2)

    def recall_pct(self):
        return round(self.recall * 100, 2)

    def f1_pct(self):
        return round(self.f1_score * 100, 2)


class PredictionHistory(models.Model):
    """Records each loan prediction made via the web form."""

    user                   = models.ForeignKey(User, on_delete=models.CASCADE, related_name="predictions")
    input_data             = models.JSONField()
    predicted_result       = models.CharField(max_length=10)        # "Approved" / "Rejected"
    prediction_probability = models.FloatField()                    # probability of predicted class
    approved_probability   = models.FloatField()                    # probability of Y (approval)
    created_at             = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name        = "Prediction History"
        verbose_name_plural = "Prediction Histories"
        ordering            = ["-created_at"]
        indexes = [
            models.Index(fields=["-created_at"]),
            models.Index(fields=["user"]),
        ]

    def __str__(self):
        return f"{self.user.username} — {self.predicted_result} ({self.created_at.date()})"

    def is_approved(self):
        return self.predicted_result == "Approved"

    def approved_pct(self):
        return round(self.approved_probability * 100, 1)

    def prediction_pct(self):
        return round(self.prediction_probability * 100, 1)
