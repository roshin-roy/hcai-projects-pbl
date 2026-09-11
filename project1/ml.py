import os
import uuid

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend: no GUI, just save PNGs
from matplotlib import pyplot as plt

from django.conf import settings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge

from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error


# Data loading

def read_csv(file_or_path):
    # drop id-like columns if present
    df = pd.read_csv(file_or_path)
    for col in list(df.columns):
        if str(col).strip().lower() in ("id", "index", "unnamed: 0"):
            df = df.drop(columns=[col])
    return df


def feature_columns(df):
    # everything except last column (the label)
    return df.columns[:-1].tolist()


def target_column(df):
    return df.columns[-1]

def target_is_numeric(df):
    return pd.api.types.is_numeric_dtype(df[target_column(df)])

def detect_problem_type(df):
    y = df.iloc[:, -1]
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    n_unique = y.nunique()
    looks_integer = np.allclose(y.dropna() % 1, 0)
    if looks_integer and n_unique <= max(20, int(0.05 * len(y))):
        return "classification"
    return "regression"


def dataframe_summary(df):
    # summary stats for the template
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "features": feature_columns(df),
        "target": target_column(df),
        "preview_html": df.head(10).to_html(
            classes="data-table", index=False, border=0
        ),
    }


# Visualization

def _new_media_path(prefix):
    name = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
    return os.path.join(settings.MEDIA_ROOT, name), settings.MEDIA_URL + name


def scatter_plot(df, feature_x, feature_y, problem_type):
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    image_path, image_url = _new_media_path("scatter")
    target = target_column(df)

    use_regression_style = (
        problem_type == "regression" and pd.api.types.is_numeric_dtype(df[target])
    )

    plt.figure(figsize=(8, 5))
    if use_regression_style:
        sc = plt.scatter(df[feature_x], df[feature_y],
                         c=df[target], cmap="viridis", alpha=0.8)
        plt.colorbar(sc, label=target)
    else:
        classes = df[target].unique()
        cmap = plt.get_cmap("tab10")
        for i, cls in enumerate(classes):
            subset = df[df[target] == cls]
            plt.scatter(subset[feature_x], subset[feature_y],
                        label=str(cls), color=cmap(i % 10), alpha=0.7)
        plt.legend(title=target)

    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(f"{feature_x} vs {feature_y}")
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()
    return image_url


# Model registry

CLASSIFIERS = {
    "logistic": {
        "label": "Logistic Regression",
        "param": "C",
        "values": [0.01, 0.1, 1, 10, 100],
        "scale": True,
        "build": lambda v: LogisticRegression(C=v, max_iter=1000),
    },
    "knn": {
        "label": "K-Nearest Neighbors",
        "param": "n_neighbors",
        "values": [1, 3, 5, 7, 9, 11, 15],
        "scale": True,
        "build": lambda v: KNeighborsClassifier(n_neighbors=v),
    },
    "svm": {
        "label": "SVM (RBF)",
        "param": "C",
        "values": [0.01, 0.1, 1, 10, 100],
        "scale": True,
        "build": lambda v: SVC(C=v),
    },
    "decision_tree": {
        "label": "Decision Tree",
        "param": "max_depth",
        "values": [1, 2, 3, 5, 7, 10, 15],
        "scale": False,
        "build": lambda v: DecisionTreeClassifier(max_depth=v, random_state=0),
    },
    "random_forest": {
        "label": "Random Forest",
        "param": "n_estimators",
        "values": [10, 25, 50, 100, 200],
        "scale": False,
        "build": lambda v: RandomForestClassifier(n_estimators=v, random_state=0),
    },
}

REGRESSORS = {
    "linear": {
        "label": "Ridge Regression",
        "param": "alpha",
        "values": [0.0, 0.1, 1, 10, 100],
        "scale": True,
        "build": lambda v: Ridge(alpha=v),
    },
    "knn": {
        "label": "K-Nearest Neighbors",
        "param": "n_neighbors",
        "values": [1, 3, 5, 7, 9, 11, 15],
        "scale": True,
        "build": lambda v: KNeighborsRegressor(n_neighbors=v),
    },
    "svm": {
        "label": "SVR (RBF)",
        "param": "C",
        "values": [0.01, 0.1, 1, 10, 100],
        "scale": True,
        "build": lambda v: SVR(C=v),
    },
    "decision_tree": {
        "label": "Decision Tree",
        "param": "max_depth",
        "values": [1, 2, 3, 5, 7, 10, 15],
        "scale": False,
        "build": lambda v: DecisionTreeRegressor(max_depth=v, random_state=0),
    },
    "random_forest": {
        "label": "Random Forest",
        "param": "n_estimators",
        "values": [10, 25, 50, 100, 200],
        "scale": False,
        "build": lambda v: RandomForestRegressor(n_estimators=v, random_state=0),
    },
}


def available_models(problem_type):
    registry = CLASSIFIERS if problem_type == "classification" else REGRESSORS
    return [(key, cfg["label"]) for key, cfg in registry.items()]


def available_scores(problem_type):
    if problem_type == "classification":
        return [("accuracy", "Accuracy"), ("f1_macro", "F1 (macro)")]
    return [("r2", "R\u00b2 score"), ("rmse", "RMSE (lower = better)")]


def _score(problem_type, score_name, y_true, y_pred):
    if score_name == "accuracy":
        return accuracy_score(y_true, y_pred)
    if score_name == "f1_macro":
        return f1_score(y_true, y_pred, average="macro")
    if score_name == "r2":
        return r2_score(y_true, y_pred)
    if score_name == "rmse":
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    raise ValueError(f"Unknown score: {score_name}")


# The supervised-learning pipeline with a hyperparameter sweep
def run_sweep(df, model_name, problem_type, test_size, score_name):
    registry = CLASSIFIERS if problem_type == "classification" else REGRESSORS
    cfg = registry[model_name]

    features = df.iloc[:, :-1]
    X = features.select_dtypes(include="number").copy()
    y = df.iloc[:, -1]

    excluded_columns = [
        col for col in features.columns if col not in X.columns
    ]

    if X.shape[1] == 0:
        raise ValueError("The dataset must contain at least one numeric feature.")

    if len(X) < 5:
        raise ValueError(
            f"Dataset too small ({len(X)} rows). Need at least 5."
        )

    if y.isna().any():
        raise ValueError(
            "The target column contains missing labels. "
            "Please complete or remove these rows before training."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0
    )

    # Training data only to calculate replacement values.
    train_medians = X_train.median()

    empty_columns = train_medians[train_medians.isna()].index.tolist()
    if empty_columns:
        names = ", ".join(str(col) for col in empty_columns)
        raise ValueError(
            f"No observed training values in these columns: {names}. "
            "Remove these columns or provide more data."
        )

    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    if cfg["scale"]:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    higher_score = score_name != "rmse"

    valid_values = cfg["values"]
    if cfg["param"] == "n_neighbors":
        valid_values = [
            value for value in valid_values
            if value <= len(X_train)
        ]

    if not valid_values:
        raise ValueError(
            f"Training set too small ({len(X_train)} samples) "
            f"for the available {cfg['param']} values."
        )

    rows = []
    for value in valid_values:
        model = cfg["build"](value)
        model.fit(X_train, y_train)

        train_score = _score(
            problem_type,
            score_name,
            y_train,
            model.predict(X_train),
        )
        test_score = _score(
            problem_type,
            score_name,
            y_test,
            model.predict(X_test),
        )

        rows.append({
            "value": value,
            "train_score": round(train_score, 4),
            "test_score": round(test_score, 4),
        })

    if higher_score:
        best = max(rows, key=lambda row: row["test_score"])
    else:
        best = min(rows, key=lambda row: row["test_score"])

    plot_url = _sweep_plot(
        rows, cfg["param"], score_name, cfg["label"]
    )

    return {
        "model_label": cfg["label"],
        "param_name": cfg["param"],
        "score_name": score_name,
        "rows": rows,
        "best": best,
        "plot_url": plot_url,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "excluded_columns": excluded_columns,
    }

def _sweep_plot(rows, param_name, score_name, model_label):
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    image_path, image_url = _new_media_path("sweep")

    x = [str(r["value"]) for r in rows]
    train = [r["train_score"] for r in rows]
    test = [r["test_score"] for r in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(x, train, marker="o", label="Train")
    plt.plot(x, test, marker="o", label="Test")
    plt.xlabel(param_name)
    plt.ylabel(score_name)
    plt.title(f"{model_label}: {score_name} vs {param_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()
    return image_url
