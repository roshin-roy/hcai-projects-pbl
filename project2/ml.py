import os
import uuid

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from django.conf import settings

from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_data():
    df = load_penguins().dropna().reset_index(drop=True)
    df = pd.get_dummies(df, columns=["island", "sex"], drop_first=False)
    y = df["species"]
    X = df.drop(columns=["species"])
    return X, y


def split_and_scale(X, y, test_size=0.25, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled,
        "scaler": scaler,
    }


def sweep_trees(data, max_leaf_range=range(2, 31)):
    rows = []
    for k in max_leaf_range:
        clf = DecisionTreeClassifier(max_leaf_nodes=k, random_state=0)
        clf.fit(data["X_train"], data["y_train"])
        acc = accuracy_score(data["y_test"], clf.predict(data["X_test"]))
        rows.append({
            "model": clf,
            "complexity": clf.get_n_leaves(),
            "acc_test": acc,
            "param_label": f"max_leaf_nodes={k}",
        })
    return rows


LOGREG_C_VALUES = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]


def sweep_logistic(data):
    rows = []
    for C in LOGREG_C_VALUES:
        clf = LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=5000)
        clf.fit(data["X_train_scaled"], data["y_train"])
        acc = accuracy_score(data["y_test"], clf.predict(data["X_test_scaled"]))
        n_nonzero = int(np.sum(np.any(clf.coef_ != 0, axis=0)))
        rows.append({
            "model": clf,
            "complexity": n_nonzero,
            "acc_test": acc,
            "param_label": f"C={C}",
        })
    return rows


def pick_best(rows, lam):
    scored = [(r["acc_test"] - lam * r["complexity"], r) for r in rows]
    return max(scored, key=lambda t: t[0])[1]


def _new_media_path(prefix):
    name = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
    return os.path.join(settings.MEDIA_ROOT, name), settings.MEDIA_URL + name


def plot_tree_image(clf, feature_names, class_names):
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    image_path, image_url = _new_media_path("tree")
    plt.figure(figsize=(14, 8))
    plot_tree(clf, feature_names=feature_names, class_names=class_names,
              filled=True, rounded=True, fontsize=8)
    plt.tight_layout()
    plt.savefig(image_path, dpi=120)
    plt.close()
    return image_url


def plot_logreg_coefs(clf, feature_names, class_names):
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    image_path, image_url = _new_media_path("logreg")
    n_classes = clf.coef_.shape[0]
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 5), sharey=True)
    if n_classes == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.barh(feature_names, clf.coef_[i])
        ax.set_title(str(class_names[i]) if i < len(class_names) else f"class {i}")
        ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(image_path, dpi=120)
    plt.close()
    return image_url


def plot_complexity_curve(rows, complexity_label):
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    image_path, image_url = _new_media_path("curve")
    xs = [r["complexity"] for r in rows]
    ys = [r["acc_test"] for r in rows]
    plt.figure(figsize=(7, 4.5))
    plt.plot(xs, ys, marker="o")
    plt.xlabel(complexity_label)
    plt.ylabel("Test accuracy")
    plt.title(f"Accuracy vs {complexity_label}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(image_path, dpi=120)
    plt.close()
    return image_url