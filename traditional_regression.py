import argparse
import os
from datetime import datetime

import pandas as pd
from pycaret.regression import *


def parse_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True)
    parser.add_argument("-o", "--output_path", default="traditional_regression_out")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_parameter()

    df = pd.read_csv(args.input_path, index_col=[0, 1]) #.iloc[:200]

    os.makedirs(args.output_path, exist_ok=True)

    # Drop 5th PC
    df = df[[c for c in df if not c.endswith("PC5")]]

    all_metrics = []
    start = datetime.now()
    for n_pc in range(4, 0, -1):
        df = df[df.columns[~df.columns.str.endswith(f"PC{n_pc + 1}")]]
        print(f"n_pc:{n_pc}, shape:{df.shape}")

        exp_name = setup(data=df.iloc[:, 1:], target=df.iloc[:, 0], fold=10)
        best_model = compare_models(include=['xgboost', 'lightgbm', 'et', 'rf', 'ridge'])

        all_metrics.append(exp_name.pull().reset_index().assign(n_pcs=n_pc))

    end = datetime.now()
    print(f"Execution time: {(end - start).total_seconds() / 60} minutes")
    result = pd.concat(all_metrics, ignore_index=True).set_index(["index", "n_pcs"])
    result.to_csv(os.path.join(args.output_path, "results.csv"))
