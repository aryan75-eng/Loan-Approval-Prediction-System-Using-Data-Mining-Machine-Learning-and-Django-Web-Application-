"""
datasets/forms.py — Dataset upload form with server-side validation.
"""

from django import forms
from django.conf import settings

from .services import ALLOWED_EXTENSIONS, FileValidationError, validate_file_extension, validate_file_size


class DatasetUploadForm(forms.Form):
    dataset_name = forms.CharField(
        max_length=255,
        widget=forms.TextInput(attrs={
            "class": "form-control",
            "placeholder": "Give your dataset a descriptive name",
        }),
        label="Dataset Name",
    )
    file = forms.FileField(
        widget=forms.ClearableFileInput(attrs={
            "class": "form-control",
            "accept": ".csv,.xlsx,.xls",
        }),
        label="Upload File (CSV or XLSX)",
        help_text=f"Max size: {settings.MAX_UPLOAD_SIZE // (1024*1024)} MB. "
                  f"Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}",
    )

    def clean_file(self):
        f = self.cleaned_data.get("file")
        if not f:
            return f
        try:
            validate_file_extension(f.name)
            validate_file_size(f)
        except FileValidationError as e:
            raise forms.ValidationError(str(e))
        return f
