import os
import click

import pandas as pd


SERVICE_COLUMNS = ["user_id", "item_id", "is_interaction", "is_recommendation"]


@click.command()
@click.argument("path")
def summarize(path: str):
    dir_name = os.path.abspath(path)
    summary = pd.DataFrame()
    for filepath in os.listdir(dir_name):
        if filepath.endswith("csv") and not filepath.startswith("summary"):
            evaluation = pd.read_csv(os.path.join(dir_name, filepath))

            summary[f"{filepath.split('.')[0]}_precision"] = evaluation[evaluation["is_recommendation"]]\
                .drop(columns=SERVICE_COLUMNS).mean()

            summary[f"{filepath.split('.')[0]}_recall"] = evaluation[evaluation["is_interaction"]]\
                .drop(columns=SERVICE_COLUMNS).mean()

    summary.to_csv(os.path.join(dir_name, "summary.csv"))


if __name__ == '__main__':
    summarize()
