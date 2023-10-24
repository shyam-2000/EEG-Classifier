import os

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from scores import print_metrics
from sklearn.metrics import accuracy_score


def cross_validation(
    estimator, X: pd.DataFrame, y: pd.DataFrame, neg: int, k=5, seed=42
) -> list:
    np.random.seed(seed)
    idxs = X.index.values
    np.random.shuffle(idxs)
    models = [clone(estimator) for _ in range(k)]
    for i in range(k):
        start = i * len(X) // k
        end = (i + 1) * len(X) // k
        test_fold = idxs[start:end]
        train_fold = np.setdiff1d(idxs, test_fold)
        # X_test = X.loc[test_fold]
        # y_test = y.loc[test_fold]
        X_train = X.loc[train_fold]
        y_train = y.loc[train_fold]
        models[i].fit(X_train, y_train)
        dump(models[i], f".\\models\\svc\\1v{neg}_fold{i}.joblib")

    return models


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


def svm_classification(
    df: pd.DataFrame, labels: pd.Series, neg_class: int, **kwargs
) -> None:
    seed = kwargs.get("seed", 42)
    k = kwargs.get("k", 1)

    X_train, X_test, y_train, y_test = train_test_split(
        df, labels, test_size=0.25, random_state=seed
    )

    if kwargs.get("pca", False):
        components = pca(X_train, threshold=0.95, savefile=f"pca{neg_class}.npy")
        X_train = reduce(X_train, components)
        X_test = reduce(X_test, components)

    if kwargs.get("scale", False):
        mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
        X_train = (X_train - mu) / sigma
        X_test = (X_test - mu) / sigma

    # Training k-folds CV
    svc = SVC(kernel="linear", random_state=seed, C=5)
    svc.fit(X_train, y_train)
    # svc_models = cross_validation(svc, X_train, y_train, neg_class, k=k, seed=seed)
    # y_pred = np.array([model.predict(X_test) for model in svc_models])
    y_pred = svc.predict(X_test)
    # y_pred = np.count_nonzero(predictions, axis=0) > (k // 2)
    if neg_class == 0:
        print(accuracy_score(y_test, y_pred))
    else:
        print_metrics(y_test, y_pred, f"SVC 1v{neg_class}")
