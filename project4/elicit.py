import numpy as np
from scipy.optimize import minimize
from . import features as ft
REG = 0.5


def _X_and_names():
    data = ft.get_feature_data()
    return data["feats"].values, data["feature_names"]


def fit_bradley_terry(pairwise):
    X, names = _X_and_names()
    dim = X.shape[1]
    if not pairwise:
        return np.zeros(dim), names
    def negll(w):
        ll = 0.0
        for win, lose in pairwise:
            d = X[win] @ w - X[lose] @ w
            ll += np.log(1.0 / (1.0 + np.exp(-d)) + 1e-12)
        return -ll + REG * np.sum(w ** 2)
    return minimize(negll, np.zeros(dim), method="L-BFGS-B").x, names


def fit_plackett_luce(rankings):
    X, names = _X_and_names()
    dim = X.shape[1]
    if not rankings:
        return np.zeros(dim), names
    def negll(w):
        ll = 0.0
        for order in rankings:
            u = X[order] @ w
            for k in range(len(order) - 1):
                rem = u[k:]
                m = np.max(rem)
                ll += u[k] - (m + np.log(np.sum(np.exp(rem - m))))
        return -ll + REG * np.sum(w ** 2)
    return minimize(negll, np.zeros(dim), method="L-BFGS-B").x, names


def interpret_w(w, names, top_k=6):
    pretty = []
    for name, weight in zip(names, w):
        label = (name.replace("genre_", "").replace("decade_", "")
                     .replace("rating_", "rating: ").replace("_", " "))
        pretty.append((label, float(weight)))
    pretty.sort(key=lambda t: t[1], reverse=True)
    liked = [p for p in pretty if p[1] > 0][:top_k]
    disliked = [p for p in pretty[::-1] if p[1] < 0][:top_k]
    return liked, disliked