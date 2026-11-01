from django.urls import path

from . import views

app_name = "predictions"

urlpatterns = [
    path("predict/",            views.predict_view,           name="predict"),
    path("predict/<int:pk>/",   views.result_view,            name="result"),
    path("history/",            views.history_view,           name="history"),
    path("performance/",        views.model_performance_view, name="performance"),
]
