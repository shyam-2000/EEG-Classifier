import os

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from scores import print_metrics


def pca(X: pd.DataFrame, threshold: float = 0.9, savefile: str = None) -> pd.DataFrame:
    if os.path.exists(os.path.join("pca", savefile)):
        return pd.DataFrame(np.load(os.path.join("pca", savefile)))
    l, v = np.linalg.eig(X.corr())
    eigen = pd.DataFrame(v.real.T)
    eigen["l"] = l.real
    eigen.sort_values("l", ascending=False, inplace=True)
    cumulative = np.cumsum(eigen["l"] / sum(eigen["l"]))
    n_comp = sum(cumulative <= threshold) + 1
    components = eigen.head(n_comp).drop("l", axis=1).T
    if savefile is not None:
        np.save(os.path.join("pca", savefile), components)
    return components


def reduce(X: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    components.index = X.columns
    return X @ components


def rf_classification(
    df: pd.DataFrame, labels: pd.Series, neg_class: int, **kwargs
) -> None:
    seed = kwargs.get("seed", 42)

    X_train, X_test, y_train, y_test = train_test_split(
        df, labels, test_size=0.25, random_state=seed
    )

    if kwargs.get("pca", False):
        components = pca(X_train, threshold=0.99, savefile=f"pca{neg_class}.npy")
        X_train = reduce(X_train, components)
        X_test = reduce(X_test, components)

    if kwargs.get("scale", False):
        mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
        X_train = (X_train - mu) / sigma
        X_test = (X_test - mu) / sigma

    # Training RF
    rf = RandomForestClassifier(
        n_estimators=150,
        min_samples_split=4,
        min_samples_leaf=50,
        random_state=seed,
    )
    rf.fit(X_train, y_train)
    dump(rf, f".\\models\\rf\\1v{neg_class}.joblib")
    y_pred = rf.predict(X_test)
    print_metrics(y_test, y_pred, f"RF 1v{neg_class}")
