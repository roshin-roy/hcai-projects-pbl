import random
from django.shortcuts import render, redirect
from . import features as ft
from . import elicit as el
from . import study as st


def index(request):
    data = ft.get_feature_data()
    return render(request, "project4/index.html",
                  {"n_movies": len(data["df"]), "n_features": len(data["feature_names"])})


def start(request):
    seed = random.randint(0, 10000)
    pairs, rankings = st.build_pools(seed)
    first = random.choice(["pairwise", "ranking"])
    request.session["p4_pairs"] = pairs
    request.session["p4_rankings"] = rankings
    request.session["p4_order"] = [first, "ranking" if first == "pairwise" else "pairwise"]
    request.session["p4_pair_done"] = 0
    request.session["p4_rank_done"] = 0
    request.session["p4_pair_responses"] = []
    request.session["p4_rank_responses"] = []
    request.session["p4_stage_index"] = 0
    return redirect("project4:consent")


def consent(request):
    return render(request, "project4/consent.html",
                  {"order": request.session.get("p4_order", ["pairwise", "ranking"])})


def _current_design(request):
    order = request.session.get("p4_order")
    idx = request.session.get("p4_stage_index", 0)
    if not order or idx >= len(order):
        return None
    return order[idx]


def _advance(request):
    request.session["p4_stage_index"] = request.session.get("p4_stage_index", 0) + 1


def task(request):
    d = _current_design(request)
    if d is None:
        return redirect("project4:results")
    return _pairwise(request) if d == "pairwise" else _ranking(request)


def _pairwise(request):
    done = request.session.get("p4_pair_done", 0)
    pairs = request.session.get("p4_pairs", [])
    if done >= len(pairs):
        _advance(request)
        return redirect("project4:task")
    a, b = pairs[done]
    return render(request, "project4/pairwise.html",
                  {"movie_a": st.movie_display(a), "movie_b": st.movie_display(b),
                   "progress": done + 1, "total": len(pairs)})


def pairwise_submit(request):
    if request.method != "POST":
        return redirect("project4:task")

    if _current_design(request) != "pairwise":
        return redirect("project4:task")

    done = request.session.get("p4_pair_done", 0)
    pairs = request.session.get("p4_pairs", [])

    if done >= len(pairs):
        return redirect("project4:task")

    try:
        task_number = int(request.POST.get("task_number", ""))
        winner = int(request.POST.get("winner", ""))
    except (TypeError, ValueError):
        return redirect("project4:task")

    # Check that this is the current task
    if task_number != done + 1:
        return redirect("project4:task")

    a, b = pairs[done]

    if winner not in (a, b):
        return redirect("project4:task")

    loser = b if winner == a else a

    responses = request.session.get("p4_pair_responses", [])
    responses.append({
        "winner": winner,
        "loser": loser,
    })

    request.session["p4_pair_responses"] = responses
    request.session["p4_pair_done"] = done + 1

    return redirect("project4:task")


def _render_ranking(request, movies, done, total, error=None):
    return render(request, "project4/ranking.html",
                  {"movies": movies, "rank_choices": list(range(1, len(movies) + 1)),
                   "progress": done + 1, "total": total, "error": error})


def _ranking(request):
    done = request.session.get("p4_rank_done", 0)
    rankings = request.session.get("p4_rankings", [])
    if done >= len(rankings):
        _advance(request)
        return redirect("project4:task")
    movies = [st.movie_display(i) for i in rankings[done]]
    return _render_ranking(request, movies, done, len(rankings))


def ranking_submit(request):
    if request.method != "POST":
        return redirect("project4:task")
    done = request.session.get("p4_rank_done", 0)
    rankings = request.session.get("p4_rankings", [])
    if done >= len(rankings):
        return redirect("project4:task")
    movie_idxs = rankings[done]
    n = len(movie_idxs)
    raw = [request.POST.get(f"rank_{idx}") for idx in movie_idxs]
    # Check that every rank is used once
    try:
        vals = [int(r) for r in raw]
        valid = sorted(vals) == list(range(1, n + 1))
    except (ValueError, TypeError):
        valid = False
    if not valid:
        movies = [st.movie_display(i) for i in movie_idxs]
        return _render_ranking(
            request, movies, done, len(rankings),
            error="Please assign each rank from 1 to %d exactly once (no repeats, no blanks)." % n,
        )
    assigned = sorted(zip(vals, movie_idxs))
    ordered = [idx for _, idx in assigned]
    r = request.session.get("p4_rank_responses", [])
    r.append({"order": ordered})
    request.session["p4_rank_responses"] = r
    request.session["p4_rank_done"] = done + 1
    return redirect("project4:task")


def results(request):
    pair_resp = request.session.get("p4_pair_responses", [])
    rank_resp = request.session.get("p4_rank_responses", [])
    pairwise = [(r["winner"], r["loser"]) for r in pair_resp]
    rankings = [r["order"] for r in rank_resp]
    w_bt, names = el.fit_bradley_terry(pairwise)
    w_pl, _ = el.fit_plackett_luce(rankings)
    lb, db = el.interpret_w(w_bt, names)
    lp, dp = el.interpret_w(w_pl, names)
    return render(request, "project4/results.html",
                  {"n_pairwise": len(pairwise), "n_rankings": len(rankings),
                   "liked_bt": lb, "disliked_bt": db, "liked_pl": lp, "disliked_pl": dp})


def reset(request):
    for k in [k for k in request.session.keys() if k.startswith("p4_")]:
        del request.session[k]
    return redirect("project4:index")