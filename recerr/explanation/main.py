from typing import Callable, List

import numpy as np
import pandas as pd


def explain_with(
    train: pd.DataFrame,
    test: pd.DataFrame,
    recommendations: pd.DataFrame,
    metrics: List[(str, Callable[[pd.DataFrame, pd.DataFrame], np.ndarray])]
) -> pd.DataFrame:
    evaluation = test.data\
        .assign(is_interaction=True)\
        .merge(recommendations.data.assign(is_recommendation=True), how="outer", on=["user_id", "item_id"])\
        .fillna({"is_interaction": False, "is_recommendation": False})

    for name, metric in metrics:
        evaluation[name] = metric(train.data, evaluation)

    return evaluation


def explain_precision(
    train: pd.DataFrame,
    test: pd.DataFrame,
    recommendations: pd.DataFrame,
    metrics: List[(str, Callable[[pd.DataFrame, pd.DataFrame], np.ndarray])]
) -> pd.Series:
    ...


def explain_recall(
    train: pd.DataFrame,
    test: pd.DataFrame,
    recommendations: pd.DataFrame,
    metrics: List[(str, Callable[[pd.DataFrame, pd.DataFrame], np.ndarray])]
) -> pd.Series:
    ...
