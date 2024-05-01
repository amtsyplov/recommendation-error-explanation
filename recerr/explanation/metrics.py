import numpy as np
import pandas as pd


def true_positive(evaluation: pd.DataFrame, *args) -> np.ndarray:
    """
    The true positive of the model on a pair (user, item) is equal to
        1 if it is both a recommendation and an interaction;
        0 in other cases.

    :param evaluation: pd.DataFrame with labeled (user, item) pairs
    :return: true_positive value for each pair
    """
    assert np.isin(["is_interaction", "is_recommendation"], evaluation.columns).all()
    return np.int32(evaluation["is_interaction"] & evaluation["is_recommendation"])


def false_positive(evaluation: pd.DataFrame, *args) -> np.ndarray:
    """
    The false positive of the model on a pair (user, item) is equal to
        1 if it is a recommendation but not an interaction;
        0 in other cases.

    :param evaluation: pd.DataFrame with labeled (user, item) pairs
    :return: false_positive value for each pair
    """
    assert np.isin(["is_interaction", "is_recommendation"], evaluation.columns).all()
    return np.int32(~evaluation["is_interaction"] & evaluation["is_recommendation"])


def false_negative(evaluation: pd.DataFrame, *args) -> np.ndarray:
    """
    The false negative of the model on a pair (user, item) is equal to
        1 if it is not a recommendation but an interaction;
        0 in other cases.

    :param evaluation: pd.DataFrame with labeled (user, item) pairs
    :return: false_negative value for each pair
    """
    assert np.isin(["is_interaction", "is_recommendation"], evaluation.columns).all()
    return np.int32(evaluation["is_interaction"] & ~evaluation["is_recommendation"])


def precision(evaluation: pd.DataFrame, *args) -> np.ndarray:
    """
    The precision of the model on a pair (user, item) is equal to
        1 if it is both a recommendation and an interaction;
        0 if it is only a recommendation;
        None in other cases.

    :param evaluation: pd.DataFrame with labeled (user, item) pairs
    :return: precision value for each pair
    """
    assert np.isin(["is_interaction", "is_recommendation"], evaluation.columns).all()
    return np.where(evaluation["is_recommendation"], np.int32(evaluation["is_interaction"]), np.nan)


def recall(evaluation: pd.DataFrame, *args) -> np.ndarray:
    """
    The recall of the model on a pair (user, item) is equal to
        1 if it is both a recommendation and an interaction;
        0 if it is only an interaction;
        None in other cases.

    :param evaluation: pd.DataFrame with labeled (user, item) pairs
    :return: recall value for each pair
    """
    assert np.isin(["is_interaction", "is_recommendation"], evaluation.columns).all()
    return np.where(evaluation["is_interaction"], np.int32(evaluation["is_recommendation"]), np.nan)


def newbie(evaluation: pd.DataFrame, historical_data: pd.DataFrame) -> np.ndarray:
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


def churn(evaluation: pd.DataFrame, historical_data: pd.DataFrame) -> np.ndarray:
    """
    The churn on (user, item) is equal to
        1 if user has interactions in historical_data but has no interactions in evaluation;
        0 in other cases.

    :param historical_data: pd.DataFrame with interactions before evaluation
    :param evaluation: pd.DataFrame with labeled (user, item) pairs
    :return: churn value for each pair
    """
    assert np.isin(["user_id", "is_interaction"], evaluation.columns).all()
    has_interactions = evaluation[["user_id", "is_interaction"]]\
        .groupby("user_id")\
        .any()\
        .rename(columns={"is_interaction": "has_interactions"})\

    has_interactions = evaluation[["user_id"]]\
        .merge(has_interactions, left_on="user_id", right_index=True)\
        .has_interactions\
        .values

    return (1 - np.int32(has_interactions)) * (1 - newbie(evaluation, historical_data))


def weak_coverage(evaluation: pd.DataFrame, historical_data: pd.DataFrame) -> np.ndarray:
    """
    Let user has n interactions and k recommendations.
    The weak coverage on (user, item) is equal to
        (n - k) / n if n > k and n > 0 and user is not a newbie;
        0 in other cases.

    :param historical_data: pd.DataFrame with interactions before evaluation
    :param evaluation: pd.DataFrame with labeled (user, item) pairs
    :return: weak_coverage value for each pair
    """
    assert np.isin(["user_id", "is_interaction", "is_recommendation"], evaluation.columns).all()
    coverage = evaluation[["user_id", "is_interaction", "is_recommendation"]]\
        .groupby("user_id")\
        .sum()\
        .assign(coverage=lambda x: np.where(
            x.is_interaction > 0,
            np.clip(x.is_interaction - x.is_recommendation, 0, None) / x.is_interaction,
            0,
        ))

    coverage = evaluation[["user_id"]].merge(coverage, left_on="user_id", right_index=True).coverage.values
    return coverage * (1 - newbie(evaluation, historical_data))


def excessive_coverage(evaluation: pd.DataFrame, historical_data: pd.DataFrame) -> np.ndarray:
    """
    Let user has n interactions and k recommendations.
    The excessive coverage on (user, item) is equal to
        (k - n) / k if k > n and k > 0 and user is not a churn;
        0 in other cases.

    :param historical_data: pd.DataFrame with interactions before evaluation
    :param evaluation: pd.DataFrame with labeled (user, item) pairs
    :return: excessive_coverage value for each pair
    """
    assert np.isin(["user_id", "is_interaction", "is_recommendation"], evaluation.columns).all()
    coverage = evaluation[["user_id", "is_interaction", "is_recommendation"]]\
        .groupby("user_id")\
        .sum()\
        .assign(coverage=lambda x: np.where(
            x.is_recommendation > 0,
            np.clip(x.is_recommendation - x.is_interaction, 0, None) / x.is_recommendation,
            0,
        ))

    coverage = evaluation[["user_id"]].merge(coverage, left_on="user_id", right_index=True).coverage.values
    return coverage * (1 - churn(evaluation, historical_data))


def novelty(evaluation: pd.DataFrame, historical_data: pd.DataFrame) -> np.ndarray:
    """
    The novelty on (user, item) is equal to
        1 if item has no interactions in historical_data;
        0 in other cases.

    :param historical_data:
    :param evaluation:
    :return: novelty value for each pair
    """
    assert "item_id" in historical_data.columns
    assert "item_id" in evaluation.columns
    return 1 - np.int32(np.isin(evaluation["item_id"], historical_data["item_id"].unique()))


def oblivion(evaluation: pd.DataFrame, historical_data: pd.DataFrame) -> np.ndarray:
    """
    The oblivion on (user, item) is equal to
        1 if item has interactions in historical_data but has no interactions in evaluation;
        0 in other cases.

    :param historical_data: pd.DataFrame with interactions before evaluation
    :param evaluation: pd.DataFrame with labeled (user, item) pairs
    :return: oblivion value for each pair
    """
    assert np.isin(["item_id", "is_interaction"], evaluation.columns).all()
    has_interactions = evaluation[["item_id", "is_interaction"]]\
        .groupby("item_id")\
        .any()\
        .rename(columns={"is_interaction": "has_interactions"})

    has_interactions = evaluation[["item_id"]]\
        .merge(has_interactions, right_index=True, left_on="item_id")\
        .has_interactions\
        .values

    return (1 - np.int32(has_interactions)) * (1 - novelty(evaluation, historical_data))


def positive_popularity_bias(evaluation: pd.DataFrame, historical_data: pd.DataFrame) -> np.ndarray:
    """
    Let item has n interactions and k recommendations.
    The positive popularity bias on (user, item) is equal to
        (k - n) / k if k > n and k > 0 and item is not forgotten;
        0 in other cases.

    :param historical_data: pd.DataFrame with interactions before evaluation
    :param evaluation: pd.DataFrame with labeled (user, item) pairs
    :return: positive_popularity_bias value for each pair
    """
    assert np.isin(["item_id", "is_interaction", "is_recommendation"], evaluation.columns).all()
    bias = evaluation[["item_id", "is_interaction", "is_recommendation"]]\
        .groupby("item_id")\
        .sum()\
        .assign(bias=lambda x: np.where(
            x.is_recommendation > 0,
            np.clip(x.is_recommendation - x.is_interaction, 0, None) / x.is_recommendation,
            0,
        ))

    bias = evaluation[["item_id"]].merge(bias, right_index=True, left_on="item_id").bias.values
    return bias * (1 - oblivion(evaluation, historical_data))


def negative_popularity_bias(evaluation: pd.DataFrame, historical_data: pd.DataFrame) -> np.ndarray:
    """
    Let item has n interactions and k recommendations.
    The negative popularity bias on (user, item) is equal to
        (n - k) / n if n > k and n > 0 and item is not new;
        0 in other cases.

    :param historical_data: pd.DataFrame with interactions before evaluation
    :param evaluation: pd.DataFrame with labeled (user, item) pairs
    :return: negative_popularity_bias value for each pair
    """
    assert np.isin(["item_id", "is_interaction", "is_recommendation"], evaluation.columns).all()
    bias = evaluation[["item_id", "is_interaction", "is_recommendation"]]\
        .groupby("item_id")\
        .sum()\
        .assign(bias=lambda x: np.where(
            x.is_interaction > 0,
            np.clip(x.is_interaction - x.is_recommendation, 0, None) / x.is_interaction,
            0,
        ))

    bias = evaluation[["item_id"]].merge(bias, right_index=True, left_on="item_id").bias.values
    return bias * (1 - novelty(evaluation, historical_data))
