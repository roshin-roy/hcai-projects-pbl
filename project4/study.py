# Functions used to create the study tasks
import numpy as np
from . import features as ft

N_PAIRWISE = 12      # pairwise comparisons in Design 1
N_RANKING = 3        # ranking tasks in Design 2
RANK_SIZE = 10       # movies per ranking task


def _pool_size():
    return len(ft.get_feature_data()["df"])


def build_pools(seed):
    """Randomly select movies for the study tasks."""
    rng = np.random.default_rng(seed)
    n = _pool_size()
    pairs = []
    for _ in range(N_PAIRWISE):
        i, j = rng.choice(n, 2, replace=False)
        pairs.append([int(i), int(j)])
    rankings = []
    for _ in range(N_RANKING):
        grp = rng.choice(n, RANK_SIZE, replace=False)
        rankings.append([int(x) for x in grp])
    return pairs, rankings


def movie_display(idx):
    """Return a dict describing a movie for the template."""
    df = ft.get_feature_data()["df"]
    row = df.iloc[idx]
    year = row["title_year"]
    dur = row["duration"]
    return {
        "idx": int(idx),
        "title": row["movie_title"],
        "year": "" if year != year else int(year),        # NaN check
        "genres": str(row["genres"]).replace("|", ", "),
        "score": row["imdb_score"],
        "duration": "" if dur != dur else int(dur),
    }