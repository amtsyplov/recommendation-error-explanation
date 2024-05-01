import pandas as pd

from .reasons_tree import ReasonsTree, residual_criterion, dynamic_groups_weighing
from .metrics import (
    true_positive,
    false_positive,
    false_negative,
    newbie,
    churn,
    weak_coverage,
    excessive_coverage,
    novelty,
    oblivion,
    positive_popularity_bias,
    negative_popularity_bias,
)


def default_post_processing(explanation: pd.DataFrame) -> pd.DataFrame:
    groups = [
        ["newbie", "churn", "weak_coverage", "excessive_coverage"],  # user explanation
        ["novelty", "oblivion", "positive_popularity_bias", "negative_popularity_bias"],  # item explanation
    ]
    return residual_criterion(dynamic_groups_weighing(groups)(explanation))


false_positive_tree = ReasonsTree(
    name="fp",
    evaluation_method=false_positive,
    subtrees=[
        ReasonsTree(name="churn", evaluation_method=churn),
        ReasonsTree(name="excessive_coverage", evaluation_method=excessive_coverage),
        ReasonsTree(name="oblivion", evaluation_method=oblivion),
        ReasonsTree(name="positive_popularity_bias", evaluation_method=positive_popularity_bias),
    ]
)


false_negative_tree = ReasonsTree(
    name="fn",
    evaluation_method=false_negative,
    subtrees=[
        ReasonsTree(name="newbie", evaluation_method=newbie),
        ReasonsTree(name="weak_coverage", evaluation_method=weak_coverage),
        ReasonsTree(name="novelty", evaluation_method=novelty),
        ReasonsTree(name="negative_popularity_bias", evaluation_method=negative_popularity_bias),
    ]
)


default_explanation = ReasonsTree(
    name="default_explanation",
    subtrees=[
        ReasonsTree(name="tp", evaluation_method=true_positive),
        false_positive_tree,
        false_negative_tree,
    ],
    post_processing=default_post_processing,
)
