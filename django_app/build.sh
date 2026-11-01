#!/usr/bin/env bash
# build.sh — Render build script for Loan Approval Prediction System
set -o errexit   # Exit on any error

# Install Python dependencies from project root
pip install -r ../requirements.txt

# Collect static files
python manage.py collectstatic --noinput

# Apply database migrations
python manage.py migrate --noinput
