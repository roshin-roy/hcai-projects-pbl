# two deferral strategies: confidence threshold + learned policy
import os
import time
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from . import ml
from . import expert as ex


DEFER_MODEL_PATH = os.path.join(ml.DATA_DIR, "defer_policy.joblib")
DEFER_METRICS_PATH = os.path.join(ml.DATA_DIR, "defer_metrics.joblib")

# Number of training examples used for the deferral model
DEFER_TRAIN_N = 30000
DEFER_TRAIN_N = 30000


def _make_defer_features(proba):
    # feature vector: softmax probs + max conf + entropy + predicted-class onehot
    max_conf = proba.max(axis=1, keepdims=True)
    pred_class = proba.argmax(axis=1)
    n_classes = proba.shape[1]
    onehot = np.zeros((proba.shape[0], n_classes))
    onehot[np.arange(proba.shape[0]), pred_class] = 1
    eps = 1e-12
    entropy = -np.sum(proba * np.log(proba + eps), axis=1, keepdims=True)
    return np.hstack([proba, max_conf, entropy, onehot])


def _four_outcomes(deferred, clf_correct, exp_correct):
    # deferral outcome breakdown
    return {
        "helpful": int(
            (deferred & ~clf_correct & exp_correct).sum()
        ),
        "harmful": int(
            (deferred & clf_correct & ~exp_correct).sum()
        ),
        "both_correct": int(
            (deferred & clf_correct & exp_correct).sum()
        ),
        "both_wrong": int(
            (deferred & ~clf_correct & ~exp_correct).sum()
        ),
        "missed": int(
            (~deferred & ~clf_correct & exp_correct).sum()
        ),
        "ok_kept": int(
            (~deferred & clf_correct).sum()
        ),
    }


def _system_accuracy(defer_decision, clf_pred, exp_pred, y_true):
    sys_pred = np.where(defer_decision == 1, exp_pred, clf_pred)
    return float((sys_pred == y_true).mean())


def _build_metrics(strategy_name, defer_decision, clf_pred, exp_pred, y_true,
                   threshold, tuning_acc=None):
    sys_acc = _system_accuracy(defer_decision, clf_pred, exp_pred, y_true)
    clf_only = float((clf_pred == y_true).mean())
    exp_only = float((exp_pred == y_true).mean())
    defer_rate = float(defer_decision.mean())
    clf_c = clf_pred == y_true
    exp_c = exp_pred == y_true
    outcomes = _four_outcomes(defer_decision == 1, clf_c, exp_c)

    total_deferred = int((defer_decision == 1).sum())   
    deferral_precision = (outcomes["helpful"] / total_deferred) if total_deferred > 0 else 0.0

    # Per-class defer rates
    per_class = []
    for c, name in enumerate(ex.EXPERT_COMPETENCE.keys()):
        cname = ml.CLASS_NAMES[c]
        m = clf_pred == c
        n = int(m.sum())
        n_def = int(defer_decision[m].sum()) if n > 0 else 0
        per_class.append({
            "name": cname,
            "n_predicted": n,
            "n_deferred": n_def,
            "defer_rate": (n_def / n) if n > 0 else 0.0,
        })

    return {
        "strategy": strategy_name,
        "threshold": float(threshold),
        "tuning_acc": float(tuning_acc) if tuning_acc is not None else None,
        "classifier_only_acc": clf_only,
        "expert_only_acc": exp_only,
        "system_acc": sys_acc,
        "improvement_over_classifier": sys_acc - clf_only,
        "defer_rate": defer_rate,
        "outcomes": outcomes,
        "deferral_precision": deferral_precision,
        "per_class_defer": per_class,
    }


def _train_learned_policy(pipeline, train_df):
    np.random.seed(0)
    sub_idx = np.random.choice(len(train_df), DEFER_TRAIN_N, replace=False)
    train_sub = train_df.iloc[sub_idx].reset_index(drop=True)

    proba = pipeline.predict_proba(train_sub["text"])
    clf_pred = proba.argmax(axis=1)
    expert_pred = ex.expert_predict(sub_idx, train_sub["label"].values)
    clf_correct = clf_pred == train_sub["label"].values
    exp_correct = expert_pred == train_sub["label"].values
    should_defer = ((~clf_correct) & exp_correct).astype(int)

    X_defer = _make_defer_features(proba)
    tr_idx, tune_idx = train_test_split(
        np.arange(len(proba)), test_size=0.3,
        random_state=0, stratify=should_defer,
    )
    defer_clf = LogisticRegression(max_iter=1000, C=1.0)
    defer_clf.fit(X_defer[tr_idx], should_defer[tr_idx])

    # Select the threshold using the tuning split
    tune_scores = defer_clf.predict_proba(X_defer[tune_idx])[:, 1]
    tune_clf = clf_pred[tune_idx]
    tune_exp = expert_pred[tune_idx]
    tune_y = train_sub["label"].values[tune_idx]

    best_thr, best_acc = 0.5, -1
    for thr in np.linspace(0.05, 0.95, 91):
        dec = (tune_scores >= thr).astype(int)
        acc = _system_accuracy(dec, tune_clf, tune_exp, tune_y)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    return defer_clf, best_thr, best_acc


def _naive_threshold(pipeline, train_df):
    # tune confidence threshold on 30k subset
    np.random.seed(0)
    sub_idx = np.random.choice(len(train_df), DEFER_TRAIN_N, replace=False)
    train_sub = train_df.iloc[sub_idx].reset_index(drop=True)
    proba = pipeline.predict_proba(train_sub["text"])
    clf_pred = proba.argmax(axis=1)
    expert_pred = ex.expert_predict(sub_idx, train_sub["label"].values)
    max_conf = proba.max(axis=1)
    y = train_sub["label"].values

    best_thr, best_acc = 0.5, -1
    for thr in np.linspace(0.30, 0.99, 70):
        dec = (max_conf < thr).astype(int)
        acc = _system_accuracy(dec, clf_pred, expert_pred, y)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    return best_thr, best_acc


def _cached_metrics_are_current(metrics):
    required_outcomes = {
        "helpful",
        "harmful",
        "both_correct",
        "both_wrong",
        "missed",
        "ok_kept",
    }

    try:
        for strategy in ("naive", "learned"):
            outcomes = metrics[strategy]["outcomes"]
            if not required_outcomes.issubset(outcomes):
                return False
    except (KeyError, TypeError):
        return False

    return True


def compute_defer_metrics():
    if os.path.exists(DEFER_MODEL_PATH) and os.path.exists(DEFER_METRICS_PATH):
        bundle = joblib.load(DEFER_MODEL_PATH)
        metrics = joblib.load(DEFER_METRICS_PATH)

        if _cached_metrics_are_current(metrics):
            metrics["source"] = "cached"
            return bundle, metrics

    t0 = time.time()
    pipeline, _ = ml.get_baseline()
    train_df, test_df = ml.load_agnews()

    learned_clf, learned_thr, learned_tune_acc = _train_learned_policy(
        pipeline, train_df
    )
    naive_thr, naive_tune_acc = _naive_threshold(pipeline, train_df)

    test_proba = pipeline.predict_proba(test_df["text"])
    test_clf = test_proba.argmax(axis=1)
    test_exp = ex.expert_predict(
        np.arange(len(test_df)),
        test_df["label"].values,
    )
    y_test = test_df["label"].values

    X_test_defer = _make_defer_features(test_proba)
    learned_scores = learned_clf.predict_proba(X_test_defer)[:, 1]
    learned_dec = (learned_scores >= learned_thr).astype(int)

    learned_metrics = _build_metrics(
        "Learned policy",
        learned_dec,
        test_clf,
        test_exp,
        y_test,
        learned_thr,
        learned_tune_acc,
    )

    naive_dec = (test_proba.max(axis=1) < naive_thr).astype(int)

    naive_metrics = _build_metrics(
        "Confidence threshold",
        naive_dec,
        test_clf,
        test_exp,
        y_test,
        naive_thr,
        naive_tune_acc,
    )

    metrics = {
        "learned": learned_metrics,
        "naive": naive_metrics,
        "fit_time_seconds": time.time() - t0,
        "source": "freshly trained",
    }

    bundle = {
        "learned_clf": learned_clf,
        "learned_threshold": learned_thr,
        "naive_threshold": naive_thr,
    }

    joblib.dump(bundle, DEFER_MODEL_PATH)
    joblib.dump(metrics, DEFER_METRICS_PATH)

    return bundle, metrics