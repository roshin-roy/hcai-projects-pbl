# Feature extraction for the movie preference model
import os
import numpy as np
import pandas as pd

from django.conf import settings
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join(settings.BASE_DIR, "project4", "data", "movie_metadata.csv")

# Genres used as features
GENRES = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
          'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'History',
          'Horror', 'Music', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Sport',
          'Thriller', 'War', 'Western']

CONTINUOUS = ['duration', 'imdb_score', 'log_votes']


def load_movies():
    df = pd.read_csv(DATA_PATH)
    # Clean movie titles
    df['movie_title'] = df['movie_title'].astype(str).str.replace('\xa0', '').str.strip()
    return df


def extract_features(df):
    """Return an interpretable feature matrix (one row per movie)."""
    feats = pd.DataFrame(index=df.index)

    # Genre features
    genre_str = df['genres'].fillna('')
    for g in GENRES:
        feats[f'genre_{g}'] = genre_str.apply(
            lambda s: 1.0 if g in s.split('|') else 0.0
        )

    # Release decade
    year = df['title_year'].fillna(df['title_year'].median())
    decade = (year // 10 * 10).astype(int)
    for d in [1980, 1990, 2000, 2010]:
        feats[f'decade_{d}s'] = (decade == d).astype(float)
    feats['decade_pre1980'] = (decade < 1980).astype(float)

    # Runtime
    feats['duration'] = df['duration'].fillna(df['duration'].median())

    # IMDB score
    feats['imdb_score'] = df['imdb_score'].fillna(df['imdb_score'].median())

    # Number of votes
    feats['log_votes'] = np.log1p(df['num_voted_users'].fillna(0))

    # Content rating
    cr = df['content_rating'].fillna('Unknown')
    feats['rating_family'] = cr.isin(['G', 'PG', 'TV-Y', 'TV-G', 'TV-PG']).astype(float)
    feats['rating_teen'] = cr.isin(['PG-13', 'TV-14']).astype(float)
    feats['rating_mature'] = cr.isin(['R', 'NC-17', 'TV-MA', 'X']).astype(float)

    return feats


def scale_features(feats):
    """Scale the continuous features."""
    feats = feats.copy()
    scaler = StandardScaler()
    feats[CONTINUOUS] = scaler.fit_transform(feats[CONTINUOUS])
    return feats, scaler


_CACHE = {}


def get_feature_data():
    """Load and cache the movie features."""
    if "feats" not in _CACHE:
        df = load_movies()
        feats = extract_features(df)
        feats_scaled, scaler = scale_features(feats)
        _CACHE["df"] = df
        _CACHE["feats"] = feats_scaled
        _CACHE["feature_names"] = list(feats_scaled.columns)
        _CACHE["scaler"] = scaler
    return _CACHE