from django.urls import path
from . import views

app_name = "project3"

urlpatterns = [
    path("", views.index, name="index"),
    path("human/", views.human_loop, name="human_loop"),
    path("human/submit/", views.human_loop_submit, name="human_loop_submit"),
    path("human/reset/", views.human_loop_reset, name="human_loop_reset"),
]