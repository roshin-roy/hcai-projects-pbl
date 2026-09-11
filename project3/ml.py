import os
import time
import joblib
import pandas as pd

from django.conf import settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline


COLS = ["label", "title", "description"]
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]  # AG News: labels 1..4

# paths for cached model + data
DATA_DIR = os.path.join(settings.BASE_DIR, "project3", "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
MODEL_PATH = os.path.join(DATA_DIR, "baseline_model.joblib")
METRICS_PATH = os.path.join(DATA_DIR, "baseline_metrics.joblib")


def load_agnews():
    # load CSVs, merge title+description, shift labels to 0-based
    train = pd.read_csv(TRAIN_PATH, header=None, names=COLS)
    test = pd.read_csv(TEST_PATH, header=None, names=COLS)
    for df in (train, test):
        df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")
        df["label"] = df["label"] - 1
    return train, test


def _train_and_evaluate():
    # train + evaluate, return (pipeline, metrics)
    train, test = load_agnews()
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000, ngram_range=(1, 2), stop_words="english"
        )),
        ("clf", LogisticRegression(max_iter=1000, C=1.0)),
    ])
    t0 = time.time()
    pipeline.fit(train["text"], train["label"])
    fit_time = time.time() - t0

    preds = pipeline.predict(test["text"])
    acc = accuracy_score(test["label"], preds)
    report = classification_report(
        test["label"], preds, target_names=CLASS_NAMES, output_dict=True
    )
    metrics = {
        "test_accuracy": acc,
        "fit_time_seconds": fit_time,
        "n_train": len(train),
        "n_test": len(test),
        "class_names": CLASS_NAMES,
        "report": report,
    }
    return pipeline, metrics


def get_baseline():
    # load from cache or train fresh
    if os.path.exists(MODEL_PATH) and os.path.exists(METRICS_PATH):
        pipeline = joblib.load(MODEL_PATH)
        metrics = joblib.load(METRICS_PATH)
        metrics["source"] = "cached"
        return pipeline, metrics

    pipeline, metrics = _train_and_evaluate()
    os.makedirs(DATA_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(metrics, METRICS_PATH)
    metrics["source"] = "freshly trained"
    return pipeline, metrics

_TEST_CACHE = {}


def get_test_set():
    if "test" not in _TEST_CACHE:
        _, test = load_agnews()
        _TEST_CACHE["test"] = test
    return _TEST_CACHE["test"]