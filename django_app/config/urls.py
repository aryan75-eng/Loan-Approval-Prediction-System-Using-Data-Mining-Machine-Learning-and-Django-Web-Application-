"""
config/urls.py — Root URL configuration.
"""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.shortcuts import redirect
from django.urls import include, path


def root_redirect(request):
    return redirect("dashboard:home")


urlpatterns = [
    # Root → dashboard
    path("", root_redirect, name="root"),

    # Admin
    path("admin/", admin.site.urls),

    # Apps
    path("accounts/",   include("accounts.urls",   namespace="accounts")),
    path("dashboard/",  include("dashboard.urls",   namespace="dashboard")),
    path("datasets/",   include("datasets.urls",    namespace="datasets")),
    path("analytics/",  include("analytics.urls",   namespace="analytics")),
    path("predictions/",include("predictions.urls", namespace="predictions")),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# Custom error handlers
handler404 = "config.views.error_404"
handler500 = "config.views.error_500"
