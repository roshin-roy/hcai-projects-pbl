# interactive human-expert active learning loop
import numpy as np

from . import ml

# Module-level cache of the shared candidate pool (read-only, same for all users)
_POOL = {}

POOL_SIZE = 3000        


def _get_pool():
    # Return the existing pool if it has already been created.
    if _POOL:
        return _POOL

    pipeline, _ = ml.get_baseline()
    train_df, _ = ml.load_agnews()

    rng = np.random.default_rng(0)
    pool_idx = rng.choice(
        len(train_df),
        POOL_SIZE,
        replace=False,
    )

    pool = train_df.iloc[pool_idx].reset_index(drop=True)

    pool_true = pool["label"].to_numpy()

    pool_proba = pipeline.predict_proba(pool["text"])
    pool_conf = pool_proba.max(axis=1)
    pool_pred = pool_proba.argmax(axis=1)

    # Show the lowest-confidence articles first.
    query_order = np.argsort(pool_conf)

    _POOL.update({
        "pool_texts": pool["text"].tolist(),
        "pool_pred": pool_pred,
        "pool_conf": pool_conf,
        "pool_true": pool_true,
        "query_order": query_order.tolist(),
    })

    return _POOL


def next_query_position(labeled_positions):
    # next most-uncertain unlabeled position, or None.
    pool = _get_pool()
    labeled = set(labeled_positions)
    for pos in pool["query_order"]:
        if pos not in labeled:
            return pos
    return None


def get_article_text(position):
    return _get_pool()["pool_texts"][position]


def estimate_competence(labeled_positions, answers):
    # Compare the submitted answers with the true labels
    pool = _get_pool()
    n_classes = len(ml.CLASS_NAMES)

    correct = {c: 0 for c in range(n_classes)}
    total = {c: 0 for c in range(n_classes)}

    for pos in labeled_positions:
        true_class = int(pool["pool_true"][pos])
        human_answer = int(answers[str(pos)])

        total[true_class] += 1

        if human_answer == true_class:
            correct[true_class] += 1

    summary = {}

    for c in range(n_classes):
        summary[c] = {
            "accuracy": (
                correct[c] / total[c]
                if total[c] > 0
                else None
            ),
            "correct": correct[c],
            "total": total[c],
        }

    return summary

def overall_human_accuracy(labeled_positions, answers):
    if not labeled_positions:
        return None

    pool = _get_pool()
    correct = 0

    for pos in labeled_positions:
        if int(answers[str(pos)]) == int(pool["pool_true"][pos]):
            correct += 1

    return correct / len(labeled_positions)