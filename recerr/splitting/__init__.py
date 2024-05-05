import numpy as np
from pandas import DataFrame


def time_based_split(data: DataFrame, test_size: float = 0.2) -> (DataFrame, DataFrame):
    assert "timestamp" in data.columns
    data_sorted = data.sort_values("timestamp", ignore_index=True)
    train_size = len(data) * (1 - test_size)
    return data_sorted[:train_size], data_sorted[train_size:]


def all_but_n_split(data: DataFrame, n: int, time_based: bool = True) -> (DataFrame, DataFrame):
    assert "user_id" in data.columns
    data_sorted = data.copy()
    if "timestamp" in data and time_based:
        data_sorted.sort_values(
            by=["user_id", "timestamp"],
            ascending=False,
            inplace=True,
            ignore_index=True,
        )
    else:
        data_sorted.sort_values(
            by="user_id",
            ascending=False,
            inplace=True,
            ignore_index=True,
        )

    data_sorted = data_sorted.assign(mask=lambda x: x.groupby("user_id").cumcount() < n)
    train = data_sorted[~data_sorted["mask"]].drop(columns=["mask"])
    test = data_sorted[data_sorted["mask"]].drop(columns=["mask"])
    return train[::-1], test[::-1]


def user_stratified_split(data: DataFrame, test_size: float = 0.2, time_based: bool = True) -> (DataFrame, DataFrame):
    assert "user_id" in data.columns
    user_counts = data[["user_id"]].assing(interaction=1).groupby("user_id").sum()
    data_sorted = data.merge(user_counts, left_on="user_id", right_index=True)

    if "timestamp" in data and time_based:
        data_sorted.sort_values(
            by=["user_id", "timestamp"],
            ascending=False,
            inplace=True,
            ignore_index=True,
        )
    else:
        data_sorted.sort_values(
            by="user_id",
            ascending=False,
            inplace=True,
            ignore_index=True,
        )

    data_sorted = data_sorted.assign(
        mask=lambda x: x.groupby("user_id").cumcount() < x["interaction"] * test_size
    )

    train = data_sorted[~data_sorted["mask"]].drop(columns=["mask"])
    test = data_sorted[data_sorted["mask"]].drop(columns=["mask"])
    return train[::-1], test[::-1]
