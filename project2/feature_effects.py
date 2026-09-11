import numpy as np

PLOT_FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]


def plot_pdp_ale(feature_name, grid, pdp_vals, ale_edges, ale_vals, class_names):
    # side-by-side PDP and ALE subplots
    import os, uuid
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from django.conf import settings

    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    name = f"effects_{uuid.uuid4().hex[:8]}.png"
    image_path = os.path.join(settings.MEDIA_ROOT, name)
    image_url = settings.MEDIA_URL + name

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for c in range(len(class_names)):
        axes[0].plot(grid, pdp_vals[:, c], marker="o", markersize=3, label=class_names[c])
    axes[0].set_title(f"PDP: {feature_name}")
    axes[0].set_xlabel(feature_name)
    axes[0].set_ylabel("Predicted probability")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if ale_vals is not None:
        for c in range(len(class_names)):
            axes[1].plot(ale_edges, ale_vals[:, c], marker="o", markersize=3, label=class_names[c])
    axes[1].set_title(f"ALE: {feature_name}")
    axes[1].set_xlabel(feature_name)
    axes[1].set_ylabel("Accumulated local effect")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(image_path, dpi=120)
    plt.close()
    return image_url


def make_predict_proba_fn(model_class, model, scaler, feature_names):
    # handles scaling for logistic internally
    import pandas as pd
    if model_class == "logistic":
        def fn(X):
            df = pd.DataFrame(X, columns=feature_names)
            return model.predict_proba(scaler.transform(df))
    else:
        def fn(X):
            df = pd.DataFrame(X, columns=feature_names)
            return model.predict_proba(df)
    return fn


def compute_pdp(predict_proba_fn, X_ref, feature_idx, grid_values):
    X_mod = X_ref.copy()
    n_classes = predict_proba_fn(X_ref[:1]).shape[1]
    out = np.zeros((len(grid_values), n_classes))
    for gi, v in enumerate(grid_values):
        X_mod[:, feature_idx] = v
        proba = predict_proba_fn(X_mod)
        out[gi] = proba.mean(axis=0)
    return out

def compute_ale(predict_proba_fn, X_ref, feature_idx, n_bins=20):
    # ALE: bin feature j by quantiles, compute local prediction differences,
    # accumulate and center. Returns (edges, centered_ale_values).
    # Derivatives: logistic regression is differentiable so these could be
    # computed analytically; a tree is piecewise constant and has none at its
    # splits. We difference across bins for both, so the curves stay comparable.
    x_j = X_ref[:, feature_idx]
    quantile_probs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(x_j, quantile_probs))
    K = len(edges) - 1
    if K < 1:
        return edges, None

    n_classes = predict_proba_fn(X_ref[:1]).shape[1]
    bin_idx = np.clip(np.digitize(x_j, edges[1:-1], right=True), 0, K - 1)

    local_effects = np.zeros((K, n_classes))
    for k in range(K):
        mask = bin_idx == k
        if mask.sum() == 0:
            continue
        X_lo = X_ref[mask].copy()
        X_lo[:, feature_idx] = edges[k]
        X_hi = X_ref[mask].copy()
        X_hi[:, feature_idx] = edges[k + 1]
        p_lo = predict_proba_fn(X_lo)
        p_hi = predict_proba_fn(X_hi)
        local_effects[k] = (p_hi - p_lo).mean(axis=0)

    accumulated = np.cumsum(local_effects, axis=0)
    ale_at_edges = np.vstack([np.zeros((1, n_classes)), accumulated])

    point_vals = ale_at_edges[bin_idx + 1]
    mean_val = point_vals.mean(axis=0)
    centered = ale_at_edges - mean_val
    return edges, centered