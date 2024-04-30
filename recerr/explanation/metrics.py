import numpy as np
import pandas as pd


def precision(historical_data: pd.DataFrame, evaluation: pd.DataFrame) -> np.ndarray:
    """
    The precision of the model on a pair (user, item) is equal to
        1 if it is both a recommendation and an interaction;
        0 if it is only a recommendation;
        None in other cases.
    :param historical_data: pd.DataFrame with interactions before evaluation
    :param evaluation: pd.DataFrame with labeled (user, item) pairs
    :return: precision value for each pair
    """
    assert np.isin(["is_interaction", "is_recommendation"], evaluation.columns)
    return np.where(evaluation["is_recommendation"], np.int32(evaluation["is_interaction"]), np.nan)


def recall(historical_data: pd.DataFrame, evaluation: pd.DataFrame) -> np.ndarray:
    """
    The recall of the model on a pair (user, item) is equal to
        1 if it is both a recommendation and an interaction;
        0 if it is only an interaction;
        None in other cases.
    :param historical_data: pd.DataFrame with interactions before evaluation
    :param evaluation: pd.DataFrame with labeled (user, item) pairs
    :return: recall value for each pair
    """
    assert np.isin(["is_interaction", "is_recommendation"], evaluation.columns)
    return np.where(evaluation["is_interaction"], np.int32(evaluation["is_recommendation"]), np.nan)


def newbie(historical_data: pd.DataFrame, evaluation: pd.DataFrame) -> np.ndarray:
    """
    The lack of newbie on (user, item) is equal to
        1 if user has no interactions in historical_data;
        0 in other cases.

    :param historical_data: pd.DataFrame with interactions before evaluation
    :param evaluation: pd.DataFrame with labeled (user, item) pairs
    :return: newbie value for each pair
    """
    assert "user_id" in historical_data.columns
    assert "user_id" in evaluation.columns
    return 1 - np.int32(np.isin(evaluation["user_id"], historical_data["user_id"].unique()))


def churn(historical_data: pd.DataFrame, evaluation: pd.DataFrame) -> np.ndarray:
    """
    The churn on (user, item) is equal to
        1 if user has interactions in historical_data but has no interactions in evaluation;
        0 in other cases.

    :param historical_data: pd.DataFrame with interactions before evaluation
    :param evaluation: pd.DataFrame with labeled (user, item) pairs
    :return: churn value for each pair
    """
    assert np.isin(["user_id", "is_interaction"], evaluation.columns)
    has_interactions = evaluation[["user_id", "is_interaction"]]\
        .groupby("user_id")\
        .any()\
        .rename({"is_interaction": "has_interactions"})\
        .merge(evaluation, left_index=True, right_on="user_id")\
        .has_interactions\
        .values

    if "newbie" in evaluation.columns:
        return (1 - np.int32(has_interactions)) * (1 - evaluation["newbie"])

    return (1 - np.int32(has_interactions)) * (1 - newbie(historical_data, evaluation))


def lack_of_coverage(historical_data: pd.DataFrame, evaluation: pd.DataFrame) -> np.ndarray:
    """
    The lack of coverage on (user, item) is equal to
        1 if user has interactions in historical_data but has no recommendations;
        0 in other cases.

    :param historical_data: pd.DataFrame with interactions before evaluation
    :param evaluation: pd.DataFrame with labeled (user, item) pairs
    :return: lack_of_coverage value for each pair
    """
    assert np.isin(["user_id", "is_recommendation"], evaluation.columns)
    has_recommendations = evaluation[["user_id", "is_recommendation"]]\
        .groupby("user_id")\
        .any()\
        .rename({"is_recommendation": "has_recommendations"})\
        .merge(evaluation, left_index=True, right_on="user_id")\
        .has_recommendations\
        .values

    if "newbie" in evaluation.columns:
        return (1 - np.int32(has_recommendations)) * (1 - evaluation["newbie"])

    return (1 - np.int32(has_recommendations)) * (1 - newbie(historical_data, evaluation))


def weak_coverage(historical_data: pd.DataFrame, evaluation: pd.DataFrame) -> np.ndarray:
    """
    Let user has n interactions and k recommendations.
    The weak coverage on (user, item) is equal to
        (n - k) / n if n > k and n > 0;
        None in other cases.

    :param historical_data: pd.DataFrame with interactions before evaluation
    :param evaluation: pd.DataFrame with labeled (user, item) pairs
    :return: lack_of_coverage value for each pair
    """
    assert np.isin(["user_id", "is_interaction", "is_recommendation"], evaluation.columns)
    return evaluation[["user_id", "is_interaction", "is_recommendation"]]\
        .groupby("user_id")\
        .sum()\
        .assign(coverage=lambda x: np.where(x.is_interaction > 0, x.is_interaction - x.is_recommendation, np.nan))\
        .coverage


def excessive_coverage(historical_data: pd.DataFrame, evaluation: pd.DataFrame) -> np.ndarray:
    ...


def novelty(historical_data: pd.DataFrame, evaluation: pd.DataFrame) -> np.ndarray:
    """new items"""
    ...


def oblivion(historical_data: pd.DataFrame, evaluation: pd.DataFrame) -> np.ndarray:
    """items exit rotation"""
    ...


def positive_popularity_bias(historical_data: pd.DataFrame, evaluation: pd.DataFrame) -> np.ndarray:
    ...


def negative_popularity_bias(historical_data: pd.DataFrame, evaluation: pd.DataFrame) -> np.ndarray:
    ...

