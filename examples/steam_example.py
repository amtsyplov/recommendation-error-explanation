import os
import click
import logging

import numpy as np
import pandas as pd
from pandas import DataFrame

from recerr.splitting import time_based_split, all_but_n_split, user_stratified_split
from recerr.metrics import evaluate_default
from recerr.explain import combine_criteria
from recerr.utils import load_hierarchy

from simple_knn import SimpleKNN

pd.set_option('future.no_silent_downcasting', True)

DATASET_SCHEMA = ["user_id", "item_id", "score", "timestamp"]

LOG_FORMAT = f"%(asctime)s %(name)s [%(levelname)s] %(message)s"


def get_stream_handler():
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    return stream_handler


logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)
logger.addHandler(get_stream_handler())


def read_data(path: str) -> DataFrame:
    data = pd.read_csv(
        os.path.join(os.path.abspath(path), "recommendations.csv"),
        usecols=[0, 3, 4, 6],
        names=["item_id", "timestamp", "score", "user_id"],
        skiprows=1,
        parse_dates=["timestamp"],
    )

    data = data[data["timestamp"] > "2021-01-01"]

    users, user_counts = np.unique(data["user_id"], return_counts=True)
    mask = np.isin(data["user_id"], users[user_counts >= 20])

    return data.loc[mask, ["user_id", "item_id", "score", "timestamp"]] \
        .sort_values(["user_id", "timestamp"], ascending=[True, False], ignore_index=True) \
        .assign(score=lambda x: np.where(x.score, 1, -1))


def split_data(data: DataFrame, test_size: float, k: int, splitting: str) -> (DataFrame, DataFrame):
    if splitting == "time_based_split":
        return time_based_split(data, test_size=test_size)
    elif splitting == "all_but_n_split":
        return all_but_n_split(data, n=k)
    return user_stratified_split(data, test_size=test_size)


@click.command()
@click.option("--input_path", default="data/steam")
@click.option("--output_path", default="results/steam")
@click.option("--hierarchy_path", default="default_hierarchy.yaml")
@click.option("--splitting", default="time_based_split")
@click.option("--test_size", default=0.2)
@click.option("--k", default=10)
def steam_example(
        input_path: str,
        output_path: str,
        hierarchy_path: str,
        splitting: str,
        test_size: float,
        k: int,
):
    logger.info(
        f"""Start command: python3 {os.path.basename(__file__)} 
    --input_path {input_path}
    --output_path {output_path}
    --hierarchy_path {hierarchy_path}
    --splitting {splitting}
    --test_size {test_size}
    --k {k}
""")

    data = read_data(input_path)
    logger.info(f"Data reading is finished. Dataset shape: {data.shape}")

    train, test = split_data(data, test_size, k, splitting)
    logger.info(f"Data splitting is finished\ntrain size: {len(train)}\ntest size: {len(test)}")

    model = SimpleKNN().fit(train)
    logger.info("Model fitting is finished")

    recommendations = model.predict(k)
    logger.info(f"Recommendations preparation is finished. Total users: {len(recommendations) // k}")

    evaluation = evaluate_default(train, test, recommendations)
    logger.info(f"Default evaluation is finished. Evaluation shape: {evaluation.shape}")

    hierarchy = load_hierarchy(os.path.abspath(hierarchy_path))
    combined_evaluation = combine_criteria(evaluation, hierarchy)
    logger.info("Combining criteria is finished")

    os.makedirs(os.path.abspath(output_path), exist_ok=True)
    combined_evaluation.to_csv(
        os.path.join(os.path.abspath(output_path), f"{splitting}_{k}.csv"),
        index=False,
    )
    logger.info("Successful finish of the pipeline")


if __name__ == '__main__':
    steam_example()
