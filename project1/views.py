import os
import uuid

from django.conf import settings
from django.shortcuts import render, redirect

from . import ml


def index(request):
    return render(request, "project1/index.html")


def upload(request):
    if request.method != "POST" or "csv_file" not in request.FILES:
        return redirect("project1:index")

    upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    f = request.FILES["csv_file"]
    saved_name = f"{uuid.uuid4().hex[:8]}_{f.name}"
    saved_path = os.path.join(upload_dir, saved_name)
    with open(saved_path, "wb+") as dest:
        for chunk in f.chunks():
            dest.write(chunk)

    # Validate it parses, and remember the detected problem type.
    try:
        df = ml.read_csv(saved_path)
    except Exception as e:
        return render(request, "project1/index.html",
                      {"error": f"Could not read CSV: {e}"})

    request.session["dataset_path"] = saved_path
    request.session["dataset_name"] = f.name
    request.session["problem_type"] = ml.detect_problem_type(df)
    # A fresh upload should clear any old scatter plot.
    request.session.pop("scatter_url", None)
    return redirect("project1:dataset")


def _load_df(request):
    # reload from session path, or None
    path = request.session.get("dataset_path")
    if not path or not os.path.exists(path):
        return None
    return ml.read_csv(path)


def _dataset_context(request, df):
    problem_type = request.session.get("problem_type", "classification")
    target_numeric = ml.target_is_numeric(df)
    if not target_numeric:
        problem_type = "classification"
        request.session["problem_type"] = problem_type
    summary = ml.dataframe_summary(df)
    return {
        "dataset_name": request.session.get("dataset_name", "dataset.csv"),
        "problem_type": problem_type,
        "target_numeric": target_numeric,
        "summary": summary,
        "features": summary["features"],
        "models": ml.available_models(problem_type),
        "scores": ml.available_scores(problem_type),
        "scatter_url": request.session.get("scatter_url"),
    }


def dataset(request):
    df = _load_df(request)
    if df is None:
        return redirect("project1:index")
    return render(request, "project1/dataset.html", _dataset_context(request, df))


def visualize(request):
    df = _load_df(request)
    if df is None:
        return redirect("project1:index")

    if request.method == "POST":
        problem_type = request.session.get("problem_type", "classification")

        features = ml.feature_columns(df)
        feature_x = request.POST.get("feature_x", features[0])
        feature_y = request.POST.get("feature_y",
                                     features[1] if len(features) > 1 else features[0])
        try:
            request.session["scatter_url"] = ml.scatter_plot(
                df, feature_x, feature_y, problem_type
            )
        except Exception as e:
            ctx = _dataset_context(request, df)
            ctx["error"] = f"Could not plot: {e}"
            return render(request, "project1/dataset.html", ctx)

    return render(request, "project1/dataset.html", _dataset_context(request, df))


def train(request):
    df = _load_df(request)
    if df is None:
        return redirect("project1:index")
    if request.method != "POST":
        return redirect("project1:dataset")

    problem_type = request.POST.get("problem_type",
                                    request.session.get("problem_type"))
    request.session["problem_type"] = problem_type

    model_name = request.POST["model"]
    test_size = float(request.POST.get("test_size", 0.2))
    score_name = request.POST.get("score")

    try:
        result = ml.run_sweep(df, model_name, problem_type, test_size, score_name)
    except Exception as e:
        ctx = _dataset_context(request, df)
        ctx["error"] = f"Training failed: {e}"
        return render(request, "project1/dataset.html", ctx)

    return render(request, "project1/results.html", {
        "dataset_name": request.session.get("dataset_name"),
        "problem_type": problem_type,
        "result": result,
        "test_size": test_size,
    })
