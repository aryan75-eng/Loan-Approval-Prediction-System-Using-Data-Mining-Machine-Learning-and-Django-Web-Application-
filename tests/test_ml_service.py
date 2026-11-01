"""
test_ml_service.py — Tests for the Django ml_service module.

Tests T14, T15 from the test matrix.
"""

import sys
from pathlib import Path

import django
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DJANGO_APP   = PROJECT_ROOT / "django_app"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DJANGO_APP))

import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from django.contrib.auth.models import User
from django.test import TestCase


SAMPLE_INPUT = {
    "ApplicantIncome":   5000.0,
    "CoapplicantIncome": 1500.0,
    "LoanAmount":        128.0,
    "Loan_Amount_Term":  360.0,
    "Credit_History":    1.0,
    "Gender":            "Male",
    "Married":           "Yes",
    "Dependents":        "0",
    "Education":         "Graduate",
    "Self_Employed":     "No",
    "Property_Area":     "Urban",
}


class TestMLService(TestCase):
    def test_is_model_loaded_returns_bool(self):
        """ml_service.is_model_loaded() must return a bool."""
        from predictions import ml_service
        result = ml_service.is_model_loaded()
        self.assertIsInstance(result, bool)

    def test_predict_requires_model(self):
        """If model does not exist, predict() should raise FileNotFoundError."""
        from django.conf import settings
        from predictions import ml_service

        if not ml_service.is_model_loaded():
            with self.assertRaises(FileNotFoundError):
                ml_service.predict(SAMPLE_INPUT)
        else:
            # Model exists — test it returns correct types
            label, pred_proba, app_proba = ml_service.predict(SAMPLE_INPUT)
            self.assertIn(label, ("Approved", "Rejected"))
            self.assertGreaterEqual(pred_proba, 0.0)
            self.assertLessEqual(pred_proba, 1.0)
            self.assertGreaterEqual(app_proba, 0.0)
            self.assertLessEqual(app_proba, 1.0)


class TestPredictionHistory(TestCase):
    """T14 — History rows must be saved with correct input_data JSON."""

    def setUp(self):
        self.user = User.objects.create_user(username="histuser", password="pass12345")

    def test_T14_history_row_saved_correctly(self):
        """T14 — PredictionHistory stores input_data as JSON accurately."""
        from predictions.models import PredictionHistory

        history = PredictionHistory.objects.create(
            user=self.user,
            input_data=SAMPLE_INPUT,
            predicted_result="Approved",
            prediction_probability=0.85,
            approved_probability=0.85,
        )
        saved = PredictionHistory.objects.get(pk=history.pk)
        self.assertEqual(saved.predicted_result, "Approved")
        self.assertEqual(saved.input_data["ApplicantIncome"], 5000.0)
        self.assertAlmostEqual(saved.approved_probability, 0.85)

    def test_T14_history_is_approved_method(self):
        """T14 — PredictionHistory.is_approved() should return correct bool."""
        from predictions.models import PredictionHistory

        approved = PredictionHistory(predicted_result="Approved")
        rejected = PredictionHistory(predicted_result="Rejected")
        self.assertTrue(approved.is_approved())
        self.assertFalse(rejected.is_approved())


class TestModelPerformance(TestCase):
    """T15 — Model performance table must match imported CSV data."""

    def test_T15_load_and_retrieve_metrics(self):
        """T15 — After import, DB metrics must match CSV values exactly."""
        from src.config import MODEL_COMPARISON_PATH

        if not MODEL_COMPARISON_PATH.exists():
            pytest.skip("model_comparison.csv not found — run model_training.py first.")

        import csv
        from predictions.models import ModelPerformance

        # Re-import metrics
        from django.core.management import call_command
        call_command("load_model_metrics", verbosity=0)

        with open(MODEL_COMPARISON_PATH) as f:
            reader = csv.DictReader(f)
            for row in reader:
                obj = ModelPerformance.objects.filter(model_name=row["name"]).first()
                self.assertIsNotNone(obj, f"Model '{row['name']}' not found in DB")
                self.assertAlmostEqual(obj.f1_score, float(row["f1"]), places=4)
                self.assertAlmostEqual(obj.accuracy, float(row["accuracy"]), places=4)
