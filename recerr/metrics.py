import numpy as np
from pandas import DataFrame


def evaluate_default(train: DataFrame, test: DataFrame, recommendations: DataFrame) -> DataFrame:
    assert np.isin(["user_id", "item_id"], train.columns).all()
    assert np.isin(["user_id", "item_id"], test.columns).all()
    assert np.isin(["user_id", "item_id"], recommendations.columns).all()

    evaluation = test[["user_id", "item_id"]]\
        .assign(is_interaction=True)\
        .merge(recommendations.assign(is_recommendation=True), how="outer", on=["user_id", "item_id"])\
        .fillna({"is_interaction": False, "is_recommendation": False})\

    evaluation = true_positive(evaluation)
    evaluation = evaluate_default_grouped(train, evaluation, column="user_id", lost_criterion="is_interaction")
    evaluation = evaluate_default_grouped(train, evaluation, column="item_id", lost_criterion="is_recommendation")
    return evaluation


def evaluate_default_grouped(
        historical_data: DataFrame,
        evaluation: DataFrame,
        column: str,
        lost_criterion: str
) -> DataFrame:
    known = historical_data[column].unique()
    column_evaluation = evaluation[[column, "is_interaction", "is_recommendation", "tp"]]\
        .groupby(column)\
        .sum()

    column_evaluation = is_new(column_evaluation, known)
    column_evaluation = is_lost(column_evaluation, known, criterion=lost_criterion)
    column_evaluation = recommendation_superiority(column_evaluation)
    column_evaluation = interation_superiority(column_evaluation)
    column_evaluation.drop(columns=["is_interaction", "is_recommendation", "tp"], inplace=True)
    return evaluation.merge(column_evaluation, left_on=column, right_index=True)


def true_positive(data: DataFrame) -> DataFrame:
    assert np.isin(["is_interaction", "is_recommendation"], data.columns).all()
    return data.assign(tp=lambda x: np.where(x["is_interaction", "is_recommendation"].all(axis=1), 1.0, 0.0))


def is_new(data: DataFrame, known: np.ndarray) -> DataFrame:
    return data.assign(**{f"is_new_{data.index.name}": lambda x: np.where(np.isin(x.index, known), 0.0, 1.0)})


def is_lost(data: DataFrame, known: np.ndarray, criterion: str) -> DataFrame:
    return data.assign(**{
        f"is_lost_{data.index.name}": lambda x: np.where(np.isin(x.index, known) & (x[criterion] == 0), 1.0, 0.0)
    })


def recommendation_superiority(data: DataFrame) -> DataFrame:
    assert np.isin(["is_interaction", "is_recommendation", "tp"], data.columns).all()

    def recommendation_superiority_column(x: DataFrame) -> np.ndarray:
        numerator = np.clip(x["is_recommendation"] - x["is_interaction"], 0, None)
        denominator = np.clip(x["is_recommendation"] - x["tp"], 1, None)
        return numerator / denominator

    return data.assign(**{
        f"{data.index.name}_recommendation_superiority": recommendation_superiority_column
    })


def interation_superiority(data: DataFrame) -> DataFrame:
    assert np.isin(["is_interaction", "is_recommendation", "tp"], data.columns).all()

    def interation_superiority_column(x: DataFrame) -> np.ndarray:
        numerator = np.clip(x["is_interaction"] - x["is_recommendation"], 0, None)
        denominator = np.clip(x["is_interaction"] - x["tp"], 1, None)
        return numerator / denominator

    return data.assign(**{
        f"{data.index.name}_interation_superiority": interation_superiority_column
    })
