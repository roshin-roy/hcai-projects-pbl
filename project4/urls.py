from django.urls import path
from . import views

app_name = "project4"

urlpatterns = [
    path("", views.index, name="index"),
    path("start/", views.start, name="start"),
    path("consent/", views.consent, name="consent"),
    path("task/", views.task, name="task"),
    path("pairwise/submit/", views.pairwise_submit, name="pairwise_submit"),
    path("ranking/submit/", views.ranking_submit, name="ranking_submit"),
    path("results/", views.results, name="results"),
    path("reset/", views.reset, name="reset"),
]