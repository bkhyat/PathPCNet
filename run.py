import argparse
import os
import json
import time
from ast import parse
from pathlib import Path

import torch
from datetime import datetime
import pandas as pd
import numpy as np
import sklearn.model_selection as skms
import sklearn.metrics as skmts
import sklearn.utils as skut
import shap
from model import PathPCNet
from utils import set_seed, get_device, evaluate_model, NumpyDataset, get_input_matrix, filter_pc_columns

SEED = 42
CV_K = 10

LEARNING_RATE = 0.0001
EPOCH = 2000
BATCH_SIZE = 20

SYSTEM_TIME = datetime.now().strftime("%Y%m%d-%H%M%S")

def run_evaluation(df, device, args):
    model_metrics = []
    sdf = skut.shuffle(df, random_state=args.seed_int)
    set_seed(args.seed_int)

    kf = skms.KFold(n_splits=args.cv_int, random_state=args.seed_int, shuffle=True)
    X_df, y_df = sdf.iloc[:, 1:], sdf.iloc[:, 0]

    loss_df_list = []
    ytest_df_list = []
    shap_df_list = []
    best_rmse = float("inf")
    for i, (train_index, test_index) in enumerate(kf.split(X_df, y_df)):
        print(f"CUDA [{device}] Fold {i + 1}")
        Xtrain, Xtest = X_df.values[train_index], X_df.values[test_index]
        ytrain, ytest = y_df.values[train_index], y_df.values[test_index]
        Xtrain, Xvalid, ytrain, yvalid = skms.train_test_split(Xtrain, ytrain, test_size=0.1,
                                                               random_state=args.seed_int)

        to_tensor = lambda x: torch.from_numpy(x.astype('float32'))
        train_dl = torch.utils.data.DataLoader(NumpyDataset(to_tensor(Xtrain), to_tensor(ytrain)),
                                               batch_size=BATCH_SIZE, shuffle=True)
        valid_dl = torch.utils.data.DataLoader(NumpyDataset(to_tensor(Xvalid), to_tensor(yvalid)),
                                               batch_size=BATCH_SIZE)
        test_dl = torch.utils.data.DataLoader(NumpyDataset(to_tensor(Xtest), to_tensor(ytest)),
                                              batch_size=BATCH_SIZE)

        model = PathPCNet(Xtrain.shape[1], dropout_rate=0.3)
        model.init_weights()

        train_loss, valid_loss = model.fit(train_dl, valid_dl, EPOCH, LEARNING_RATE, device, torch.optim.AdamW,
                                           path=os.path.join(args.out_path, f"{SYSTEM_TIME}_checkpoint.pt"))
        preds = model.predict(test_dl, device)

        rmse = np.sqrt(skmts.mean_squared_error(ytest, preds))
        if rmse <= best_rmse:
            best_rmse = rmse

        ytest_df = y_df.iloc[test_index].to_frame()
        ytest_df['prediction'] = preds
        ytest_df['fold'] = i + 1
        ytest_df_list.append(ytest_df)

        loss_df = pd.DataFrame({'fold': [i + 1] * len(train_loss),
                                'epoch': range(1, len(train_loss) + 1),
                                'train loss': train_loss,
                                'valid loss': valid_loss})
        loss_df_list.append(loss_df)
        if args.shap:
            train_dataset = NumpyDataset(torch.from_numpy(Xtrain), torch.from_numpy(ytrain))
            train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=200, shuffle=True)
            background, lbl = next(iter(train_dl))
            explainer = shap.DeepExplainer(model, background[:100].float().to(device))
            shap_arr = explainer.shap_values(torch.from_numpy(Xtest).float())
            shap_arr = np.squeeze(shap_arr)
            shap_df = pd.DataFrame(shap_arr, index=y_df.iloc[test_index].index, columns=X_df.columns)
            # append to result
            shap_df_list.append(shap_df)

        pd.concat(ytest_df_list).to_csv(os.path.join(args.out_path, f"{SYSTEM_TIME}_predictions.csv"))
        pd.concat(loss_df_list).to_csv(os.path.join(args.out_path, f"{SYSTEM_TIME}_loss.csv"))
        if args.shap:
            pd.concat(shap_df_list).to_csv(os.path.join(args.out_path, f"{SYSTEM_TIME}_shap.csv"))

        metrics = evaluate_model(pd.concat(ytest_df_list), 'LN_IC50', 'prediction')
        model_metrics.append(metrics)

    return model_metrics

def parse_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_path", required=True)
    parser.add_argument("-s", "--seed_int", default=SEED, type=int)
    parser.add_argument("-cv", "--cv_int", default=CV_K, type=int)
    parser.add_argument("-o", "--out_path", default="out")
    parser.add_argument("-c", "--cuda", default=-1, type=int)
    parser.add_argument("--shap", default=False, type=bool, help="Whether to calculate shap values")
    # Hyperparameters
    parser.add_argument("-e", "--epochs", default=EPOCH, type=int)
    parser.add_argument("-lr", "--learning_rate", default=LEARNING_RATE, type=float)
    parser.add_argument("-b", "--batch_size", default=BATCH_SIZE, type=int)
    parser.add_argument("--n_pcs", default=3, type=int,
                        help="Numer of principal components to use. [1-4]")
    # This argument is not implemented in this script yet.
    parser.add_argument("--omics",
                        nargs="+",
                        choices=["CNV", "MUT", "EXP"],
                        default=["CNV", "MUT", "EXP"],
                        help="Select one or more Omics")
    parser.add_argument("--n_mfp_bits", default=256, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_parameter()
    os.makedirs(os.path.join(args.out_path), exist_ok=True)
    if args.cuda == -1:
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(args.cuda)
        device = f"cuda:{args.cude}"
        torch.cuda.manual_seed(args.seed_int)
        torch.manual_seed(args.seed_int)
    all_df = get_input_matrix(args.in_path, mfp_n_bits=args.n_mfp_bits, n_pcs=args.n_pcs)
    df = filter_pc_columns(all_df, args.n_pcs)

    print(f"n_pc:{args.n_pcs}, shape:{df.shape}")
    metrics = run_evaluation(df, device, args)
    print("="*40)
    with open(os.path.join(args.out_path, f"{SYSTEM_TIME}_model_metrics.json"), "w") as f:
        json.dump(metrics, f)
    print("Execution Finished\n"+f"Output files written at: {Path(args.out_path).resolve()}\n"+"="*40)