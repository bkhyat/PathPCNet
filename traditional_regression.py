import argparse
import os
from datetime import datetime

import pandas as pd
from pycaret.regression import *

from utils import get_input_matrix, filter_pc_columns


def parse_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_path", required=True, type=str,
                        help="Path to the the input directory that contains the processed data files.")
    parser.add_argument("-o", "--out_path", required=True,
                        type=str, help="Output path to store the summary results")
    parser.add_argument("--n_pcs", default=3, type=int,
                        help="Numer of principal components to use. [1-4]")
    parser.add_argument(
        "--models",
        nargs="+",  # one or more values
        choices=["xgboost", "lightgbm", "et", "rf", "ridge"],
        default=["xgboost", "lightgbm", "et", "rf", "ridge"],
        help="Select one or more models"
    )
    parser.add_argument("--n_mfp_bits", default=256, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_parameter()
    os.makedirs(args.out_path, exist_ok=True)

    all_df = get_input_matrix(args.in_path, mfp_n_bits=args.n_mfp_bits, n_pcs=args.n_pcs)
    print(f"===============\n"
          "COMMAND LINE ARGS VALUES\n"
          + "\n".join(f"{key}:\t{val}" for key, val in vars(args).items()) + "\n===============")
    all_metrics = []
    start = datetime.now()
    for n_pc in range(1, args.n_pcs + 1):
        df = filter_pc_columns(all_df, n_pc)
        print(f"n_pc:{n_pc}, shape:{df.shape}")

        exp_name = setup(data=df.iloc[:, 1:], target=df.iloc[:, 0], fold=10)
        best_model = compare_models(include=args.models)

        all_metrics.append(exp_name.pull().reset_index().assign(n_pcs=n_pc))

    end = datetime.now()
    print(f"Execution time: {(end - start).total_seconds() / 60} minutes")
    result = pd.concat(all_metrics, ignore_index=True).set_index(["index", "n_pcs"])
    result.to_csv(os.path.join(args.out_path, f"traditional_results_{str(datetime.now())}.csv"))
