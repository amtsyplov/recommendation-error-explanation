import numpy as np
import pandas as pd

from typing import Callable, List


def dummy_evaluation_method(*args, **kwargs) -> float:
    return 1.0


def residual_criterion(explanation: pd.DataFrame) -> pd.DataFrame:
    return explanation.assign(pure_error=lambda x: 1 - explanation.sum(axis=1))


def dynamic_groups_weighing(groups: List[List[str]]) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def wrapper(explanation: pd.DataFrame) -> pd.DataFrame:
        groups_participation = []
        for group in groups:
            groups_participation.append(np.int32(explanation[group].sum(axis=1) > 0))

        groups_participation_sum = sum(groups_participation)
        groups_participation_sum = np.where(groups_participation_sum == 0, 1, groups_participation_sum)

        explanation = explanation.copy(deep=True)
        for group, participation in zip(groups, groups_participation):
            weight = participation / groups_participation_sum
            for column in group:
                explanation[column] *= weight

        return explanation

    return wrapper


class ReasonsTree:
    def __init__(self,
                 name: str,
                 evaluation_method: Callable[..., float | np.ndarray] = dummy_evaluation_method,
                 weight: float | np.ndarray = 1.0,
                 subtrees: List['ReasonsTree'] | None = None,
                 remove_if_subtrees: bool = True,
                 post_processing: Callable[[pd.DataFrame], pd.DataFrame] | None = None):
        self.name = name
        self.evaluation_method = evaluation_method
        self.weight = weight
        self.subtrees = subtrees
        self.post_processing = post_processing
        self.remove_if_subtrees = remove_if_subtrees

    def __call__(self, *args, **kwargs) -> pd.DataFrame:
        value = self.weight * self.evaluation_method(*args, **kwargs)
        if isinstance(value, float):
            value = np.float64([value])

        if self.subtrees is None:
            return pd.DataFrame({self.name: value})

        explanation = pd.DataFrame()
        for subtree in self.subtrees:
            subtree_explanation = subtree(*args, **kwargs)
            for column in subtree_explanation.columns:
                explanation[column] = value * subtree_explanation[column]

        if not self.remove_if_subtrees:
            explanation[self.name] = value

        if self.post_processing is not None:
            explanation = self.post_processing(explanation)

        return explanation

    def __repr__(self):
        return f"ReasonsTree({self.name}, [{', '.join([subtree.name for subtree in self.subtrees])}])"

    def __str__(self):
        return repr(self)
