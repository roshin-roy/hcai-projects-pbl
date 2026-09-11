
import numpy as np
import pandas as pd

NUMERIC_FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "year"]
CATEGORICAL_GROUPS = {
    "island": ["island_Biscoe", "island_Dream", "island_Torgersen"],
    "sex": ["sex_female", "sex_male"],
}


def compute_mad(X_train, feature_names):
    # MAD per feature (for weighted L1 distance)
    arr = np.asarray(X_train, dtype=float)
    mad = {}
    for j, col in enumerate(feature_names):
        vals = arr[:, j]
        med = np.median(vals)
        m = np.median(np.abs(vals - med))
        mad[col] = m if m > 1e-8 else 1.0  # guard divide-by-zero for constant columns
    return mad


def compute_stds(X_train, feature_names):
    arr = np.asarray(X_train, dtype=float)
    return {col: max(arr[:, j].std(), 1e-6) for j, col in enumerate(feature_names)}


def _perturb_batch(x, N, feature_names, feature_stds, noise_scale, flip_prob, rng):
    # gaussian noise for numeric, random flip for categorical one-hots
    batch = np.tile(x, (N, 1)).astype(float)
    idx_map = {name: i for i, name in enumerate(feature_names)}

    for feat in NUMERIC_FEATURES:
        j = idx_map[feat]
        batch[:, j] += rng.normal(0, noise_scale * feature_stds[feat], size=N)
    y_idx = idx_map["year"]
    batch[:, y_idx] = np.round(batch[:, y_idx])  # year stays a whole number

    for group_name, cols in CATEGORICAL_GROUPS.items():
        col_idxs = np.array([idx_map[c] for c in cols])
        flip_mask = rng.random(N) < flip_prob
        n_flip = int(flip_mask.sum())
        if n_flip > 0:
            new_cat = rng.integers(0, len(cols), size=n_flip)
            rows = np.where(flip_mask)[0]
            batch[np.ix_(rows, col_idxs)] = 0.0
            batch[rows, col_idxs[new_cat]] = 1.0
    return batch


def make_predict_fn(model_class, model, scaler):
    if model_class == "logistic":
        def predict_fn(X):
            Xs = scaler.transform(X)
            return model.predict(Xs), model.predict_proba(Xs)
    else:
        def predict_fn(X):
            return model.predict(X), model.predict_proba(X)
    return predict_fn


def generate_counterfactuals(model, model_class, scaler, x, target_label, feature_names,
                              mad, feature_stds, k=5, N=300, max_rounds=5,
                              base_noise_scale=0.3, flip_prob=0.3, seed=0):
    rng = np.random.default_rng(seed)
    predict_fn = make_predict_fn(model_class, model, scaler)
    mad_vec = np.array([mad[f] for f in feature_names])

    noise_scale = base_noise_scale
    n_samples = N
    p_flip = flip_prob
    all_valid = []
    rounds_used = 0

    for round_i in range(max_rounds):
        rounds_used = round_i + 1
        batch = _perturb_batch(x, n_samples, feature_names, feature_stds,
                               noise_scale, p_flip, rng)
        batch_df = pd.DataFrame(batch, columns=feature_names)
        preds, _ = predict_fn(batch_df)
        mask = (preds == target_label)
        if mask.sum() > 0:
            valid_points = batch[mask]
            dists = np.sum(np.abs(valid_points - x) / mad_vec, axis=1)
            for pt, d in zip(valid_points, dists):
                all_valid.append((d, pt))
        if len(all_valid) >= k:
            break
        n_samples *= 2
        noise_scale *= 1.5
        p_flip = min(0.9, p_flip * 1.3)

    all_valid.sort(key=lambda t: t[0])
    top = all_valid[:k]
    info = {"rounds_used": rounds_used, "final_N": n_samples, "n_found": len(all_valid)}
    return top, info