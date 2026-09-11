# active learning: uncertainty vs random sampling for competence discovery
import os
import time
import uuid
import numpy as np
import joblib

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from django.conf import settings
from . import ml
from . import expert as ex


ACTIVE_METRICS_PATH = os.path.join(ml.DATA_DIR, "active_metrics.joblib")

POOL_SIZE = 30000     # candidate pool the expert could be queried from
BUDGET = 3000         # total expert queries
ROUND_SIZE = 300      # queries added per round
SEEDS = [0, 1, 2, 3, 4]

# Simulated expert competence for each class
TRUE_COMPETENCE = np.array(
    [ex.EXPERT_COMPETENCE[c] for c in range(len(ml.CLASS_NAMES))]
)


def _competence_estimate(true_classes, expert_answers, n_classes):
    # Estimate expert per-class accuracy from the queried labels only.
    est = np.full(n_classes, np.nan)
    for c in range(n_classes):
        m = true_classes == c
        if m.sum() > 0:
            est[c] = (expert_answers[m] == c).mean()
    return est


def _run_one(strategy, pipeline, train_df, seed):
    # One active-learning run. Returns list of (n_queries, competence_MAE).
    rng = np.random.default_rng(seed)
    pool_idx = rng.choice(len(train_df), POOL_SIZE, replace=False)
    pool = train_df.iloc[pool_idx].reset_index(drop=True)
    pool_proba = pipeline.predict_proba(pool["text"])
    pool_maxconf = pool_proba.max(axis=1)
    pool_y = pool["label"].values
    n_classes = len(ml.CLASS_NAMES)

    queried = np.zeros(len(pool), dtype=bool)
    history = []
    while queried.sum() < BUDGET:
        n = min(ROUND_SIZE, BUDGET - queried.sum())
        remaining = np.where(~queried)[0]
        if strategy == "uncertainty":
            pick = remaining[np.argsort(pool_maxconf[remaining])][:n]
        else:  
            pick = rng.choice(remaining, n, replace=False)
        queried[pick] = True

        q = np.where(queried)[0]
        expert_ans = ex.expert_predict(pool_idx[q], pool_y[q])
        est = _competence_estimate(pool_y[q], expert_ans, n_classes)
        mae = float(np.nanmean(np.abs(est - TRUE_COMPETENCE)))
        history.append((int(queried.sum()), mae))
    return history


def _plot_curves(query_counts, unc_mae, rnd_mae):
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    name = f"active_{uuid.uuid4().hex[:8]}.png"
    image_path = os.path.join(settings.MEDIA_ROOT, name)
    image_url = settings.MEDIA_URL + name

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(query_counts, unc_mae, marker="o",
            label="Uncertainty sampling (active)", color="#275CB2")
    ax.plot(query_counts, rnd_mae, marker="s",
            label="Random sampling (baseline)", color="#F28E2B")
    ax.set_xlabel("Number of expert queries")
    ax.set_ylabel("Competence-profile error (MAE)")
    ax.set_title("Active vs random: learning the expert's competence profile")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(image_path, dpi=120)
    plt.close()
    return image_url


def compute_active_metrics():
    # Run both strategies and average the results
    if os.path.exists(ACTIVE_METRICS_PATH):
        metrics = joblib.load(ACTIVE_METRICS_PATH)
        metrics["source"] = "cached"
        # The plot file may have been cleaned; regenerate from stored curves.
        metrics["plot_url"] = _plot_curves(
            metrics["query_counts"], metrics["uncertainty_mae"], metrics["random_mae"]
        )
        return metrics

    t0 = time.time()
    pipeline, _ = ml.get_baseline()
    train_df, _ = ml.load_agnews()

    unc_runs, rnd_runs = {}, {}
    for s in SEEDS:
        for q, mae in _run_one("uncertainty", pipeline, train_df, s):
            unc_runs.setdefault(q, []).append(mae)
        for q, mae in _run_one("random", pipeline, train_df, s):
            rnd_runs.setdefault(q, []).append(mae)

    query_counts = sorted(unc_runs.keys())
    unc_mae = [float(np.mean(unc_runs[q])) for q in query_counts]
    rnd_mae = [float(np.mean(rnd_runs[q])) for q in query_counts]

    plot_url = _plot_curves(query_counts, unc_mae, rnd_mae)

    metrics = {
        "query_counts": query_counts,
        "uncertainty_mae": unc_mae,
        "random_mae": rnd_mae,
        "plot_url": plot_url,
        "budget": BUDGET,
        "pool_size": POOL_SIZE,
        "n_seeds": len(SEEDS),
        "fit_time_seconds": time.time() - t0,
        "source": "freshly trained",
        "early_gain": round(rnd_mae[0] - unc_mae[0], 4),  
    }
    joblib.dump(metrics, ACTIVE_METRICS_PATH)
    return metrics