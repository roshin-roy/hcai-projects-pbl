from django.shortcuts import render
from . import ml
from . import expert as ex
from . import defer as df
from . import active as al
from . import human_loop as hl
from django.shortcuts import redirect

_CACHE = {}


def _get_baseline():
    if "pipeline" not in _CACHE:
        pipeline, metrics = ml.get_baseline()
        _CACHE["pipeline"] = pipeline
        _CACHE["metrics"] = metrics
    return _CACHE["pipeline"], _CACHE["metrics"]


def _get_expert_metrics():
    if "expert_metrics" not in _CACHE:
        test_df = ml.get_test_set()
        _CACHE["expert_metrics"] = ex.evaluate_expert(test_df)
    return _CACHE["expert_metrics"]


def _get_comparison_chart():
    # regenerated each time (not cached)
    _, base_metrics = _get_baseline()
    expert_metrics = _get_expert_metrics()
    return ex.plot_comparison_chart(
        base_metrics["report"],
        expert_metrics["report"],
        base_metrics["class_names"],
    )


def _per_class_rows(metrics_report, class_names, competence=None):
    # build per-class rows for the template
    rows = []
    for i, cname in enumerate(class_names):
        r = metrics_report[cname]
        row = {
            "name": cname,
            "precision": round(r["precision"], 4),
            "recall": round(r["recall"], 4),
            "f1": round(r["f1-score"], 4),
            "support": int(r["support"]),
        }
        if competence is not None:
            row["competence"] = competence[i]
        rows.append(row)
    return rows


def index(request):
    _, base_metrics = _get_baseline()
    expert_metrics = _get_expert_metrics()

    return render(request, "project3/index.html", {
        # Baseline classifier
        "test_accuracy": round(base_metrics["test_accuracy"], 4),
        "n_train": base_metrics["n_train"],
        "n_test": base_metrics["n_test"],
        "fit_time": round(base_metrics["fit_time_seconds"], 1),
        "source": base_metrics.get("source", "unknown"),
        "defer_metrics": _get_defer_metrics(),
        "active_metrics": _get_active_metrics(),
        "per_class_rows": _per_class_rows(
            base_metrics["report"], base_metrics["class_names"]
        ),
        # Simulated expert
        "expert_accuracy": round(expert_metrics["overall_accuracy"], 4),
        "expert_rows": _per_class_rows(
            expert_metrics["report"], expert_metrics["class_names"],
            competence=expert_metrics["competence"],
        ),
        "comparison_chart_url": _get_comparison_chart(),
    })
def _get_defer_metrics():
    if "defer_metrics" not in _CACHE:
        _bundle, metrics = df.compute_defer_metrics()
        _CACHE["defer_metrics"] = metrics
    return _CACHE["defer_metrics"]

def _get_active_metrics():
    if "active_metrics" not in _CACHE:
        _CACHE["active_metrics"] = al.compute_active_metrics()
    return _CACHE["active_metrics"]

def human_loop(request):
    labeled = request.session.get("hl_labeled", [])
    answers = request.session.get("hl_answers", {})

    pos = hl.next_query_position(labeled)
    article_text = hl.get_article_text(pos) if pos is not None else None

    competence = hl.estimate_competence(labeled, answers)
    competence_rows = []

    for class_id, class_name in enumerate(ml.CLASS_NAMES):
        result = competence[class_id]

        competence_rows.append({
            "name": class_name,
            "accuracy": (
                round(result["accuracy"], 3)
                if result["accuracy"] is not None
                else None
            ),
            "correct": result["correct"],
            "total": result["total"],
        })

    overall_accuracy = hl.overall_human_accuracy(labeled, answers)

    return render(request, "project3/human_loop.html", {
        "n_labeled": len(labeled),
        "article_text": article_text,
        "query_position": pos,
        "class_choices": list(enumerate(ml.CLASS_NAMES)),
        "competence_rows": competence_rows,
        "overall_accuracy": (
            round(overall_accuracy, 3)
            if overall_accuracy is not None
            else None
        ),
    })


def human_loop_submit(request):
    if request.method != "POST":
        return redirect("project3:human_loop")

    labeled = request.session.get("hl_labeled", [])
    answers = request.session.get("hl_answers", {})

    try:
        submitted_position = int(request.POST.get("position", ""))
        submitted_label = int(request.POST.get("label", ""))
    except (TypeError, ValueError):
        return redirect("project3:human_loop")

    expected_position = hl.next_query_position(labeled)
    valid_labels = range(len(ml.CLASS_NAMES))

    if (
        submitted_position == expected_position
        and submitted_position not in labeled
        and submitted_label in valid_labels
    ):
        labeled.append(submitted_position)
        answers[str(submitted_position)] = submitted_label

        request.session["hl_labeled"] = labeled
        request.session["hl_answers"] = answers

    return redirect("project3:human_loop")


def human_loop_reset(request):
    request.session.pop("hl_labeled", None)
    request.session.pop("hl_answers", None)
    return redirect("project3:human_loop")