"""
datasets/models.py — Dataset model for uploaded CSV/XLSX files.
"""

import os

from django.contrib.auth.models import User
from django.db import models


def user_dataset_upload_path(instance, filename):
    """Store uploads under media/datasets/<user_id>/<filename> to prevent collisions."""
    import uuid
    from pathlib import Path

    ext = Path(filename).suffix.lower()
    safe_name = f"{uuid.uuid4().hex}{ext}"
    return os.path.join("datasets", str(instance.user.id), safe_name)


class Dataset(models.Model):
    """Represents an uploaded dataset file."""

    STATUS_UPLOADED  = "uploaded"
    STATUS_VALIDATED = "validated"
    STATUS_ERROR     = "error"

    STATUS_CHOICES = [
        (STATUS_UPLOADED,  "Uploaded"),
        (STATUS_VALIDATED, "Validated"),
        (STATUS_ERROR,     "Error"),
    ]

    user          = models.ForeignKey(User, on_delete=models.CASCADE, related_name="datasets")
    dataset_name  = models.CharField(max_length=255)
    uploaded_file = models.FileField(upload_to=user_dataset_upload_path, max_length=500)
    upload_date   = models.DateTimeField(auto_now_add=True)
    total_records = models.IntegerField(default=0)
    total_columns = models.IntegerField(default=0)
    status        = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_UPLOADED)
    error_message = models.TextField(blank=True, default="")

    class Meta:
        ordering = ["-upload_date"]
        verbose_name = "Dataset"
        verbose_name_plural = "Datasets"
        indexes = [
            models.Index(fields=["user"]),
            models.Index(fields=["-upload_date"]),
        ]

    def __str__(self):
        return f"{self.dataset_name} ({self.user.username})"

    def filename(self):
        """Return just the filename without the path."""
        return os.path.basename(self.uploaded_file.name)

    def file_size_kb(self):
        """Return file size in KB, or None if file doesn't exist."""
        try:
            return round(self.uploaded_file.size / 1024, 1)
        except Exception:
            return None
