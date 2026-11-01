"""
management/commands/load_model_metrics.py

Import reports/model_comparison.csv into the ModelPerformance table.
Idempotent: re-running updates existing rows (unique_together = dataset, model_name).

Usage:
  python manage.py load_model_metrics
  python manage.py load_model_metrics --dataset_id 3
"""

import csv
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from predictions.models import ModelPerformance


class Command(BaseCommand):
    help = "Import model comparison metrics from reports/model_comparison.csv into the database."

    def add_arguments(self, parser):
        parser.add_argument(
            "--dataset_id",
            type=int,
            default=None,
            help="Optional Dataset PK to associate with the imported metrics.",
        )

    def handle(self, *args, **options):
        csv_path = settings.ML_REPORTS_DIR / "model_comparison.csv"

        if not csv_path.exists():
            raise CommandError(
                f"Report not found at {csv_path}. "
                "Run 'python src/model_training.py' first."
            )

        dataset = None
        dataset_id = options.get("dataset_id")
        if dataset_id:
            from datasets.models import Dataset
            try:
                dataset = Dataset.objects.get(pk=dataset_id)
            except Dataset.DoesNotExist:
                raise CommandError(f"Dataset with id={dataset_id} does not exist.")

        # Reset is_selected flags before import
        ModelPerformance.objects.all().update(is_selected=False)

        imported = 0
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                is_selected = row.get("is_selected", "False").strip() == "True"
                obj, created = ModelPerformance.objects.update_or_create(
                    dataset    = dataset,
                    model_name = row["name"],
                    defaults={
                        "accuracy":    float(row["accuracy"]),
                        "precision":   float(row["precision"]),
                        "recall":      float(row["recall"]),
                        "f1_score":    float(row["f1"]),
                        "cv_mean":     float(row["cv_mean"]),
                        "cv_std":      float(row["cv_std"]),
                        "is_selected": is_selected,
                    },
                )
                action = "Created" if created else "Updated"
                self.stdout.write(f"  {action}: {row['name']} (F1={row['f1']})")
                imported += 1

        self.stdout.write(self.style.SUCCESS(
            f"\nSuccessfully imported {imported} model(s) from {csv_path}"
        ))
