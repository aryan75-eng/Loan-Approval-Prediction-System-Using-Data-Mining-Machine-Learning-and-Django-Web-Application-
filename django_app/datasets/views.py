"""
datasets/views.py — Upload, preview, list, and download dataset views.
All views require login. Ownership enforced on preview/download.
"""

import os

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.http import FileResponse, Http404
from django.shortcuts import get_object_or_404, redirect, render

from .forms import DatasetUploadForm
from .models import Dataset
from .services import FileValidationError, read_dataframe, validate_and_profile


@login_required
def upload_view(request):
    """Upload a new dataset file."""
    if request.method == "POST":
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES["file"]
            dataset_name  = form.cleaned_data["dataset_name"]

            try:
                # Validate and profile the file
                profile = validate_and_profile(uploaded_file, uploaded_file.name)

                # Create Dataset record
                dataset = Dataset(
                    user         = request.user,
                    dataset_name = dataset_name,
                    total_records = profile["total_records"],
                    total_columns = profile["total_columns"],
                    status        = Dataset.STATUS_VALIDATED,
                )
                # Reset file pointer before saving
                uploaded_file.seek(0)
                dataset.uploaded_file = uploaded_file
                dataset.save()

                messages.success(request, f"'{dataset_name}' uploaded successfully ({profile['total_records']} rows).")
                return redirect("datasets:preview", pk=dataset.pk)

            except FileValidationError as e:
                messages.error(request, str(e))
                # Mark as error if record partially exists — here it doesn't yet
            except Exception as e:
                messages.error(request, f"Unexpected error during upload: {e}")

        else:
            messages.error(request, "Please fix the errors below.")
    else:
        form = DatasetUploadForm()

    return render(request, "datasets/upload.html", {"form": form, "expected_columns": [
        "Loan_ID", "Gender", "Married", "Dependents", "Education", "Self_Employed",
        "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
        "Credit_History", "Property_Area", "Loan_Status",
    ]})


@login_required
def preview_view(request, pk):
    """Preview first 10 rows, dtypes, and missing value summary."""
    dataset = get_object_or_404(Dataset, pk=pk, user=request.user)

    try:
        with open(dataset.uploaded_file.path, "rb") as f:
            df = read_dataframe(f, dataset.filename())
        rows    = df.head(10).fillna("").to_dict(orient="records")
        columns = list(df.columns)
        dtypes  = df.dtypes.astype(str).to_dict()
        missing = df.isnull().sum().to_dict()
        missing_total = sum(missing.values())
        duplicate_count = int(df.duplicated().sum())

        context = {
            "dataset":        dataset,
            "rows":           rows,
            "columns":        columns,
            "dtypes":         dtypes,
            "missing":        missing,
            "missing_total":  missing_total,
            "duplicate_count": duplicate_count,
            "total_records":  len(df),
        }
    except Exception as e:
        messages.error(request, f"Could not read file: {e}")
        context = {"dataset": dataset, "error": str(e)}

    return render(request, "datasets/preview.html", context)


@login_required
def list_view(request):
    """Paginated list of the user's uploaded datasets."""
    datasets_qs = Dataset.objects.filter(user=request.user).order_by("-upload_date")
    paginator = Paginator(datasets_qs, 10)
    page = request.GET.get("page", 1)
    datasets_page = paginator.get_page(page)
    return render(request, "datasets/list.html", {"datasets": datasets_page})


@login_required
def download_view(request, pk):
    """
    Serve dataset file as a download.
    Ownership-checked — uses FileResponse (not redirect to MEDIA_URL).
    """
    dataset = get_object_or_404(Dataset, pk=pk, user=request.user)
    try:
        file_handle = open(dataset.uploaded_file.path, "rb")
        response = FileResponse(
            file_handle,
            as_attachment=True,
            filename=dataset.filename(),
        )
        return response
    except FileNotFoundError:
        raise Http404("File not found on server.")
