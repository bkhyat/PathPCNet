import argparse
import os
from datetime import datetime

import pandas as pd
from pycaret.regression import *

from utils import get_input_matrix, filter_pc_columns


def parse_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_path", required=True)
    parser.add_argument("-o", "--out_path", required=True, type=str)
    parser.add_argument("--n_pcs", default=3, type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_parameter()
    os.makedirs(args.out_path, exist_ok=True)

    all_df = get_input_matrix(args.in_path, mfp_n_bits=256, n_pcs=args.n_pcs)

    all_metrics = []
    start = datetime.now()
    for n_pc in range(args.n_pcs):
        df = filter_pc_columns(all_df, n_pc+1)
        print(f"n_pc:{n_pc}, shape:{df.shape}")

        exp_name = setup(data=df.iloc[:, 1:], target=df.iloc[:, 0], fold=10)
        best_model = compare_models(include=['xgboost', 'lightgbm', 'et', 'rf', 'ridge'])

        all_metrics.append(exp_name.pull().reset_index().assign(n_pcs=n_pc))

    end = datetime.now()
    print(f"Execution time: {(end - start).total_seconds() / 60} minutes")
    result = pd.concat(all_metrics, ignore_index=True).set_index(["index", "n_pcs"])
    result.to_csv(os.path.join(args.out_path, f"traditional_results_{str(datetime.now())}.csv"))
