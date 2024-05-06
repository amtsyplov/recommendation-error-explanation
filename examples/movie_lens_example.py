import os
import click

import pandas as pd
from pandas import DataFrame

from recerr.splitting import time_based_split, all_but_n_split, user_stratified_split
from recerr.metrics import evaluate_default
from recerr.explain import combine_criteria
from recerr.utils import load_hierarchy

from simple_knn import SimpleKNN

pd.set_option('future.no_silent_downcasting', True)


DATASET_SCHEMA = ["user_id", "item_id", "score", "timestamp"]


def read_data(path: str) -> DataFrame:
    data = pd.read_csv(os.path.join(os.path.abspath(path), "u.data"), sep="\t", names=DATASET_SCHEMA)
    return data.sort_values("timestamp", ascending=True, ignore_index=True)


def split_data(data: DataFrame, test_size: float, k: int, splitting: str) -> (DataFrame, DataFrame):
    if splitting == "time_based_split":
        return time_based_split(data, test_size=test_size)
    elif splitting == "all_but_n_split":
        return all_but_n_split(data, n=k)
    return user_stratified_split(data, test_size=test_size)


@click.command()
@click.option("--input_path", default="data/movie_lens_100k")
@click.option("--output_path", default="results/movie_lens_100k")
@click.option("--hierarchy_path", default="default_hierarchy.yaml")
@click.option("--splitting", default="time_based_split")
@click.option("--test_size", default=0.2)
@click.option("--k", default=10)
def movie_lens_example(
        input_path: str,
        output_path: str,
        hierarchy_path: str,
        splitting: str,
        test_size: float,
        k: int,
):
    data = read_data(input_path)
    train, test = split_data(data, test_size, k, splitting)
    model = SimpleKNN().fit(train)
    recommendations = model.predict(k)
    evaluation = evaluate_default(train, test, recommendations)
    hierarchy = load_hierarchy(os.path.abspath(hierarchy_path))
    combined_evaluation = combine_criteria(evaluation, hierarchy)
    os.makedirs(os.path.abspath(output_path), exist_ok=True)
    combined_evaluation.to_csv(
        os.path.join(os.path.abspath(output_path), f"{splitting}_{k}.csv"),
        index=False,
    )


if __name__ == '__main__':
    movie_lens_example()
