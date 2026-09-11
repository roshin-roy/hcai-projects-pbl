from django.shortcuts import render
from . import counterfactual as cf
from . import feature_effects as fe
import numpy as np
from . import ml

_CACHE = {}


def _get_sweeps():
    if not _CACHE:
        X, y = ml.load_data()
        data = ml.split_and_scale(X, y)
        _CACHE["X"] = X
        _CACHE["y"] = y
        _CACHE["data"] = data
        _CACHE["feature_names"] = list(X.columns)
        _CACHE["class_names"] = sorted(y.unique().tolist())
        _CACHE["tree_rows"] = ml.sweep_trees(data)
        _CACHE["logreg_rows"] = ml.sweep_logistic(data)
        _CACHE["mad"] = cf.compute_mad(data["X_train"], _CACHE["feature_names"])
        _CACHE["stds"] = cf.compute_stds(data["X_train"], _CACHE["feature_names"])
        _CACHE["X_ref"] = X.values.astype(float)  
    return _CACHE


def index(request):
    cache = _get_sweeps()

    model_class = request.GET.get("model_class", "tree")
    try:
        lam = float(request.GET.get("lambda", 0.0))
    except ValueError:
        lam = 0.0

    if model_class == "logistic":
        rows = cache["logreg_rows"]
        complexity_label = "Number of features with non-zero coefficients"
        lam_max = 0.3
    else:
        model_class = "tree"
        rows = cache["tree_rows"]
        complexity_label = "Number of leaves"
        lam_max = 0.1

    best = ml.pick_best(rows, lam)

    if model_class == "logistic":
        model_image_url = ml.plot_logreg_coefs(
            best["model"], cache["feature_names"], cache["class_names"]
        )
    else:
        model_image_url = ml.plot_tree_image(
            best["model"], cache["feature_names"], cache["class_names"]
        )

    curve_url = ml.plot_complexity_curve(rows, complexity_label)

    context = {
        "model_class": model_class,
        "lam": lam,
        "lam_max": lam_max,
        "complexity_label": complexity_label,
        "best": best,
        "model_image_url": model_image_url,
        "curve_url": curve_url,
        "feature_names": cache["feature_names"],
        "class_names": cache["class_names"],
        "n_examples": len(cache["X"]),
    }

    cf_index_raw = request.GET.get("cf_index")
    cf_target = request.GET.get("cf_target")
    context["cf_index"] = cf_index_raw if cf_index_raw is not None else 0
    context["cf_target"] = cf_target or ""

    if cf_index_raw is not None and cf_target:
        try:
            cf_index = int(cf_index_raw)
            x_row = cache["X"].iloc[cf_index]
            x = x_row.values.astype(float)
            true_label = cache["y"].iloc[cf_index]

            scaler = cache["data"]["scaler"] if model_class == "logistic" else None

            top, info = cf.generate_counterfactuals(
                best["model"], model_class, scaler, x, cf_target,
                cache["feature_names"], cache["mad"], cache["stds"], k=5,
            )

            cf_original_cells = [round(float(v), 2) for v in x]
            cf_results = []
            for d, pt in top:
                cells = []
                for j in range(len(cache["feature_names"])):
                    changed = abs(pt[j] - x[j]) > 1e-6
                    cells.append({"value": round(float(pt[j]), 2), "changed": changed})
                cf_results.append({"distance": round(float(d), 3), "cells": cells})

            context.update({
                "cf_index": cf_index,
                "cf_true_label": true_label,
                "cf_original_cells": cf_original_cells,
                "cf_results": cf_results,
                "cf_info": info,
            })
        except (ValueError, IndexError) as e:
            context["cf_error"] = f"Could not generate counterfactuals for that input: {e}"

    # --- feature effects ---
    context["plot_features"] = fe.PLOT_FEATURES
    fx_feature = request.GET.get("fx_feature")
    context["fx_feature"] = fx_feature or ""

    if fx_feature and fx_feature in fe.PLOT_FEATURES:
        try:
            scaler = cache["data"]["scaler"] if model_class == "logistic" else None
            predict_proba_fn = fe.make_predict_proba_fn(
                model_class, best["model"], scaler, cache["feature_names"]
            )
            j = cache["feature_names"].index(fx_feature)
            X_ref = cache["X_ref"]
            col = X_ref[:, j]
            grid = np.linspace(col.min(), col.max(), 25)
            pdp_vals = fe.compute_pdp(predict_proba_fn, X_ref, j, grid)
            edges, ale_vals = fe.compute_ale(predict_proba_fn, X_ref, j, n_bins=20)
            effects_url = fe.plot_pdp_ale(
                fx_feature, grid, pdp_vals, edges, ale_vals, cache["class_names"]
            )
            context["effects_url"] = effects_url
        except Exception as e:
            context["fx_error"] = f"Could not compute feature effects: {e}"

    return render(request, "project2/index.html", context)