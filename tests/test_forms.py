"""
test_forms.py — Tests for Django forms.

Tests T03, T04, T11 from the test matrix.
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

from django.test import TestCase

from datasets.forms import DatasetUploadForm
from predictions.forms import PredictionForm


def make_mock_file(content: bytes, name: str):
    """Create a simple in-memory file object that mimics Django's UploadedFile."""
    from django.core.files.uploadedfile import InMemoryUploadedFile
    f = io.BytesIO(content)
    return InMemoryUploadedFile(f, "file", name, "text/plain", len(content), None)


# ── T03: Invalid extension rejected ──────────────────────────────────────────

class TestDatasetUploadForm(TestCase):
    def test_T03_txt_file_rejected(self):
        """T03 — .txt files must be rejected by the upload form."""
        file = make_mock_file(b"col1,col2\n1,2", "data.txt")
        form = DatasetUploadForm(
            data={"dataset_name": "test"},
            files={"file": file},
        )
        self.assertFalse(form.is_valid())
        self.assertIn("file", form.errors)

    def test_T03_csv_accepted(self):
        """T03 — Valid CSV files must pass form validation."""
        file = make_mock_file(b"Gender,Married\nMale,Yes", "data.csv")
        form = DatasetUploadForm(
            data={"dataset_name": "test dataset"},
            files={"file": file},
        )
        self.assertTrue(form.is_valid(), msg=str(form.errors))

    def test_T03_xlsx_accepted(self):
        """T03 — .xlsx extension must be accepted."""
        # Minimal XLSX magic bytes (ZIP header PK)
        xlsx_magic = b"PK\x03\x04" + b"\x00" * 100
        file = make_mock_file(xlsx_magic, "data.xlsx")
        form = DatasetUploadForm(
            data={"dataset_name": "xlsx test"},
            files={"file": file},
        )
        # Only extension check at form level — should pass
        self.assertTrue(form.is_valid(), msg=str(form.errors))

    def test_T04_oversized_file_rejected(self):
        """T04 — Files larger than MAX_UPLOAD_SIZE must be rejected."""
        from django.conf import settings
        oversized = b"x" * (settings.MAX_UPLOAD_SIZE + 1)
        file = make_mock_file(oversized, "big.csv")
        form = DatasetUploadForm(
            data={"dataset_name": "big file"},
            files={"file": file},
        )
        self.assertFalse(form.is_valid())
        self.assertIn("file", form.errors)


# ── T11: Prediction form — negative income rejected ───────────────────────────

class TestPredictionForm(TestCase):
    def _valid_data(self):
        return {
            "ApplicantIncome":    "5000",
            "CoapplicantIncome":  "1500",
            "LoanAmount":         "128",
            "Loan_Amount_Term":   "360",
            "Credit_History":     "1.0",
            "Gender":             "Male",
            "Married":            "Yes",
            "Dependents":         "0",
            "Education":          "Graduate",
            "Self_Employed":      "No",
            "Property_Area":      "Urban",
        }

    def test_valid_prediction_form(self):
        """Valid inputs should produce a valid form."""
        form = PredictionForm(data=self._valid_data())
        self.assertTrue(form.is_valid(), msg=str(form.errors))

    def test_T11_negative_income_rejected(self):
        """T11 — Negative ApplicantIncome must fail validation."""
        data = self._valid_data()
        data["ApplicantIncome"] = "-1000"
        form = PredictionForm(data=data)
        self.assertFalse(form.is_valid())
        self.assertIn("ApplicantIncome", form.errors)

    def test_T11_negative_loan_rejected(self):
        """T11 — LoanAmount below minimum must fail."""
        data = self._valid_data()
        data["LoanAmount"] = "0"
        form = PredictionForm(data=data)
        self.assertFalse(form.is_valid())
        self.assertIn("LoanAmount", form.errors)

    def test_T11_missing_required_field(self):
        """T11 — Missing required field must fail."""
        data = self._valid_data()
        del data["ApplicantIncome"]
        form = PredictionForm(data=data)
        self.assertFalse(form.is_valid())
