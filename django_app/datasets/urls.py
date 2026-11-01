from django.urls import path

from . import views

app_name = "datasets"

urlpatterns = [
    path("",                    views.list_view,     name="list"),
    path("upload/",             views.upload_view,   name="upload"),
    path("<int:pk>/preview/",   views.preview_view,  name="preview"),
    path("<int:pk>/download/",  views.download_view, name="download"),
]
