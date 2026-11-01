"""
test_views.py — Integration tests for Django views.

Tests T01, T10, T12, T13 from the test matrix.
"""

import io
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
from django.test import Client, TestCase
from django.urls import reverse


def make_mock_csv(content: bytes = None, name: str = "loan.csv"):
    from django.core.files.uploadedfile import SimpleUploadedFile
    if content is None:
        content = b"Loan_ID,Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area,Loan_Status\nLP001,Male,Yes,0,Graduate,No,5000,1500,128,360,1.0,Urban,Y\n"
    return SimpleUploadedFile(name, content, content_type="text/csv")


class TestAuthRequired(TestCase):
    """T12 — Unauthenticated access to protected views must redirect to login."""

    def test_T12_dashboard_redirects(self):
        response = self.client.get(reverse("dashboard:home"))
        self.assertEqual(response.status_code, 302)
        self.assertIn("/accounts/login/", response["Location"])

    def test_T12_predict_redirects(self):
        response = self.client.get(reverse("predictions:predict"))
        self.assertEqual(response.status_code, 302)

    def test_T12_datasets_redirects(self):
        response = self.client.get(reverse("datasets:list"))
        self.assertEqual(response.status_code, 302)

    def test_T12_analytics_redirects(self):
        response = self.client.get(reverse("analytics:overview"))
        self.assertEqual(response.status_code, 302)


class TestDatasetUploadView(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="testuser", password="pass12345")
        self.other = User.objects.create_user(username="other", password="pass12345")
        self.client.login(username="testuser", password="pass12345")

    def test_T01_valid_csv_upload(self):
        """T01 — Valid CSV upload should create Dataset and redirect to preview."""
        csv_file = make_mock_csv()
        response = self.client.post(
            reverse("datasets:upload"),
            {"dataset_name": "Test Dataset", "file": csv_file},
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn("/preview/", response["Location"])

        from datasets.models import Dataset
        self.assertTrue(
            Dataset.objects.filter(user=self.user, dataset_name="Test Dataset").exists()
        )

    def test_T13_other_user_cannot_access_dataset(self):
        """T13 — A user cannot preview another user's dataset."""
        from datasets.models import Dataset
        csv_file = make_mock_csv()
        # Upload as testuser
        self.client.post(
            reverse("datasets:upload"),
            {"dataset_name": "Private", "file": csv_file},
        )
        dataset = Dataset.objects.filter(user=self.user).first()
        if not dataset:
            self.skipTest("Dataset not created — check upload flow.")

        # Switch to other user
        self.client.logout()
        self.client.login(username="other", password="pass12345")
        response = self.client.get(reverse("datasets:preview", kwargs={"pk": dataset.pk}))
        self.assertIn(response.status_code, [403, 404])


class TestPredictionView(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="preduser", password="pass12345")
        self.client.login(username="preduser", password="pass12345")

    def test_T10_predict_view_get(self):
        """T10 — GET /predictions/predict/ should render form (or model_missing notice)."""
        response = self.client.get(reverse("predictions:predict"))
        self.assertEqual(response.status_code, 200)

    def test_T12_history_requires_login(self):
        """T12 — History view requires authentication."""
        self.client.logout()
        response = self.client.get(reverse("predictions:history"))
        self.assertEqual(response.status_code, 302)
