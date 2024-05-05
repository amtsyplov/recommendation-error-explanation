from typing import Callable

import numpy as np
from pandas import DataFrame

from recerr.utils import CriteriaGroup, CriteriaLevel, DEFAULT_COLUMNS


def combine_criteria(data: DataFrame, hierarchy: list[CriteriaLevel]) -> DataFrame:
    combined_criteria = data[DEFAULT_COLUMNS].assign(residual_error=1.0)
    for level in hierarchy:
        combined_criteria = combined_criteria.merge(
            normalize_criteria_level(data, level),
            on=DEFAULT_COLUMNS,
        )

        combined_criteria = update_residual_error(combined_criteria, level)

    return combined_criteria


def normalize_criteria_level(data: DataFrame, level: CriteriaLevel) -> DataFrame:
    normalized_data = data.copy()
    group_names = [group.name for group in level]
    for group in level:
        normalized_data = normalized_data.assign(
            **{group.name: get_criteria_group_participation(group)}
        )

    normalized_data = normalized_data.assign(
        level_participation=lambda x: x[group_names].sum(axis=1)
    )

    normalized_data = normalized_data.assign(
        level_participation=lambda x: np.where(x["level_participation"] > 0, x["level_participation"], 1)
    )

    for group in level:
        for criterion in group.criteria:
            normalized_data = normalized_data.assign(
                **{criterion: lambda x: x[criterion] * x[group.name] / x["level_participation"]}
            )

    columns = sum((group.criteria for group in level), start=DEFAULT_COLUMNS)
    return normalized_data[columns]


def get_criteria_group_participation(criteria_group: CriteriaGroup) -> Callable[[DataFrame], np.ndarray]:
    def wrapper(data: DataFrame) -> np.ndarray:
        return np.where(np.any(data[criteria_group.criteria].values > 0, axis=1), criteria_group.weight, 0.0)

    return wrapper


def update_residual_error(data: DataFrame, level: CriteriaLevel) -> DataFrame:
    criteria = sum((group.criteria for group in level), start=list())
    updated_data = data.copy()
    for criterion in criteria:
        updated_data = updated_data.assign(
            **{criterion: lambda x: x[criterion] * x["residual_error"]}
        )

    return updated_data.assign(residual_error=lambda x: x["residual_error"] - x[criteria].sum())
