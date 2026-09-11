import numpy as np
import os
import uuid
from sklearn.metrics import accuracy_score, classification_report
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from django.conf import settings

from . import ml

# Per-class competence: probability the expert gets the true label right.
# Wrong answers are uniform over the other 3 classes.
EXPERT_COMPETENCE = {
    0: 0.95,  # World      - strong
    1: 0.95,  # Sports     - strong
    2: 0.55,  # Business   - weak
    3: 0.55,  # Sci/Tech   - weak
}


def expert_predict(row_indices, true_labels, seed=42):
    # returns simulated predictions for given rows
    row_indices = np.asarray(row_indices)
    true_labels = np.asarray(true_labels)
    n_classes = len(ml.CLASS_NAMES)
    preds = np.empty(len(row_indices), dtype=int)

    for i, (idx, y_true) in enumerate(zip(row_indices, true_labels)):
        # Deterministic per-row: same row_index -> same expert answer, always.
        rng = np.random.default_rng(seed + int(idx))
        p_correct = EXPERT_COMPETENCE[int(y_true)]
        if rng.random() < p_correct:
            preds[i] = y_true
        else:
            # pick a wrong class uniformly at random
            wrong_choices = [c for c in range(n_classes) if c != y_true]
            preds[i] = rng.choice(wrong_choices)
    return preds


def evaluate_expert(test_df):
    row_indices = np.arange(len(test_df))
    y_true = test_df["label"].values
    y_pred = expert_predict(row_indices, y_true)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=ml.CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    return {
        "overall_accuracy": acc,
        "report": report,
        "competence": EXPERT_COMPETENCE,
        "class_names": ml.CLASS_NAMES,
    }

def plot_comparison_chart(classifier_report, expert_report, class_names):
    # per-class accuracy comparison bar chart
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    name = f"expert_compare_{uuid.uuid4().hex[:8]}.png"
    image_path = os.path.join(settings.MEDIA_ROOT, name)
    image_url = settings.MEDIA_URL + name

    classifier_acc = [classifier_report[c]["recall"] for c in class_names]
    expert_acc = [expert_report[c]["recall"] for c in class_names]

    n = len(class_names)
    x = np.arange(n)
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width/2, classifier_acc, width, label="Classifier", color="#275CB2")
    b2 = ax.bar(x + width/2, expert_acc, width, label="Expert", color="#F28E2B")
    ax.set_ylabel("Per-class accuracy (recall)")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_title("Classifier vs Simulated Expert — per-class accuracy")
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=0.8, label="Random baseline (0.25)")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            ax.text(r.get_x() + r.get_width()/2, h + 0.01, f"{h:.2f}",
                    ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(image_path, dpi=120)
    plt.close()
    return image_url