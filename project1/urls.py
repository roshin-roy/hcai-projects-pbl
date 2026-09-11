from django.urls import path
from . import views

app_name = "project1"

urlpatterns = [
    path("", views.index, name="index"),
    path("upload/", views.upload, name="upload"),
    path("dataset/", views.dataset, name="dataset"),
    path("visualize/", views.visualize, name="visualize"),
    path("train/", views.train, name="train"),
]
