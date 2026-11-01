from django.urls import path

from . import views

app_name = "analytics"

urlpatterns = [
    path("overview/",                    views.overview_view,        name="overview"),
    path("preprocessing/",               views.preprocessing_view,   name="preprocessing"),
    path("visualizations/",              views.visualizations_view,  name="visualizations"),
    path("viz/<str:filename>",           views.viz_image_view,       name="viz_image"),
]
