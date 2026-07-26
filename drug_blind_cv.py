import argparse
import os
import json
import random
import time
import torch
import itertools
import multiprocessing as mp
import pandas as pd
import numpy as np
import sklearn.model_selection as skms
from model import PathPCNet
from utils import set_seed, evaluate_model, NumpyDataset
import torch.optim as optim
from sklearn.metrics import r2_score
import datetime

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

seed = 42

# Constants
EPOCH = 2000
MIN_FREE_MEM_MB = 2000
CHECK_POINT_PATH = "checkpoints"
GPU_COUNT = torch.cuda.device_count()

# Hyperparameter grid
LEARNING_RATES = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
DROPOUT_RATES = [0.1, 0.2, 0.3, 0.4, 0.5]
BATCH_SIZES = [8, 12, 16, 20, 24, 28]

# Define optimizer configs
OPTIMIZERS = {
    "adam": (optim.Adam, {}),
    "adamw": (optim.AdamW, {}),
    "rmsprop": (optim.RMSprop, {}),
    "sgd": (optim.SGD, {"momentum": 0.9}),
}

# Cartesian product of all combinations
HYPERPARAM_GRID = list(itertools.product(
    LEARNING_RATES,
    DROPOUT_RATES,
    BATCH_SIZES,
    OPTIMIZERS.keys()
))

data_fold_splits = []
to_tensor = lambda x: torch.from_numpy(x.astype('float32'))

def print_(*args):
    print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]')
    print(*args)

def get_free_gpus(min_free_mb=MIN_FREE_MEM_MB):
    free_gpus = []
    for i in range(torch.cuda.device_count()):
        free = torch.cuda.mem_get_info(i)[0] / 1024 / 1024
        if free >= min_free_mb:
            free_gpus.append(i)
    return free_gpus

def run_hp_job(hp_config, device_id, fold_data, seed, output_path, shared_metrics):
    fold, lr, dr, bs, opt_name, epoch = hp_config
    opt_class, opt_kwargs = OPTIMIZERS[opt_name]

    torch.cuda.set_device(device_id)
    device = f"cuda:{device_id}"
    name = f"{opt_name}_lr{lr}_drop{dr}_bs{bs}"
    print_(f"[GPU {device}] Tuning {name}")

    try:

        Xtrain, ytrain, Xtest, ytest, Xvalid, yvalid = fold_data

        train_dl = torch.utils.data.DataLoader(NumpyDataset(to_tensor(Xtrain.values), to_tensor(ytrain.values)), batch_size=bs,
                                               shuffle=True)
        valid_dl = torch.utils.data.DataLoader(NumpyDataset(to_tensor(Xvalid.values), to_tensor(yvalid.values)), batch_size=bs)
        test_dl = torch.utils.data.DataLoader(NumpyDataset(to_tensor(Xtest.values), to_tensor(ytest.values)), batch_size=bs)

        model = PathPCNet(Xtrain.shape[1], dropout_rate=dr)
        model.init_weights()

        ckpt_path = os.path.join(output_path, CHECK_POINT_PATH, name + f"_fold_{fold+1}_final.pt")
        train_loss, valid_loss, epoch = model.fit(train_dl, valid_dl, epoch, lr, device, opt_class, ckpt_path, validate=True)
        preds = model.predict(test_dl, device)

        y_true = np.array(ytest.values)
        y_pred = np.array(preds)

        # Errors
        errors = y_pred - y_true

        # Metrics
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(errors))

        # Pearson Correlation Coefficient (PCC)
        pcc = np.corrcoef(y_true, y_pred)[0, 1]

        # R^2 Score
        r2 = r2_score(y_true, y_pred)

        metrics = {
            "name": name,
            "fold": fold+1,
            "optimizer": opt_name,
            "learning_rate": lr,
            "dropout_rate": dr,
            "batch_size": bs,
            "epoch": epoch,
            "rmse": rmse,
            "mse": mse,
            "mae": mae,
            "pcc": pcc,
            "r2": r2
        }
        print_(name, metrics)

        metrics["df"] = ytest.to_frame("LN_IC50").assign(prediction=y_pred, fold=fold+1)
        shared_metrics.append(metrics)

    except Exception as e:
        print_(f"[GPU {device}] Error: {e}")
    finally:
        try:
            torch.cuda.empty_cache()
            del model
        except:
            print_("Error Releasing GPU")
            pass

MAX_PARALLEL_PROCESSES = 30

def scheduler(hp_tasks, seed, output_path, shared_metrics):
    processes = []

    while hp_tasks:
        proces_counter = 0
        while len(processes) >= MAX_PARALLEL_PROCESSES:
            proces_counter += 1
            if proces_counter%10==0:
                proces_counter = 0
                print_(f"⏳ Max parallel limit reached ({MAX_PARALLEL_PROCESSES}), waiting...")
            time.sleep(10)
            processes = [p for p in processes if p.is_alive()]

        free_gpus = get_free_gpus()
        counter = 0
        if not free_gpus:
            counter += 1
            if counter%10==0:
                print_(f"⏳ Waiting for a GPU with at least {MIN_FREE_MEM_MB}MB free memory...")
                counter = 0
            time.sleep(10)
            continue

        for gpu_id in free_gpus:
            if not hp_tasks or len(processes) >= MAX_PARALLEL_PROCESSES:
                break
            config = hp_tasks.pop(0)
            p = mp.Process(target=run_hp_job, args=(config, gpu_id, data_fold_splits[config[0]], seed, output_path, shared_metrics))
            p.start()
            processes.append(p)

        time.sleep(5)
        processes = [p for p in processes if p.is_alive()]

    for p in processes:
        p.join()

if __name__ == "__main__":
    assert GPU_COUNT > 0, "No CUDA GPU available"
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Train PathPCNet model")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to final data matrix with Pathway PC features, MFP features, and Target Variable"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="drug_blind_Jul25",
        help="Cross-validation fold number"
    )

    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    os.makedirs(os.path.join(output_path, CHECK_POINT_PATH), exist_ok=True)
    configs = []
    manager = mp.Manager()
    shared_metrics = manager.list()

    completed_file = os.path.join(output_path, "fold_results.json")
    df = pd.read_csv(os.path.join(input_path, "data_matrix_pc1.csv"), index_col=[0, 1])
    sdf = df.sample(frac=1, random_state=seed)
    set_seed(seed)
    best_config_exists = os.path.exists(os.path.join(input_path, "best_configs.json"))
    X_df, y_df = sdf.iloc[:, 1:], sdf.iloc[:, 0]

    adj = pd.Series(1, index=df.index).unstack(fill_value=0).astype(int).T
    outer_mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    configs = []
    if best_config_exists:
        best_config = json.load(open(os.path.join(input_path, "best_configs.json"), "r"))
        for conf in best_config:
            configs.append([conf["fold", conf["learning_rate"], conf["dropout_rate"], conf["batch_size"], conf["optimizer"], conf["epoch"]]])
    elif os.path.exists(os.path.join(output_path, "configs.json")):
        configs_list = json.load(open(os.path.join(output_path, "configs.json"), "r"))
    else:
         configs_list = random.choices(HYPERPARAM_GRID, k=50)
         with open(os.path.join(output_path, "configs.json"), "w") as f:
             json.dump(configs_list, f)
    for i, (train_index, test_index) in enumerate(outer_mskf.split(adj.index, adj.values), 0):
        train_drugs_outer = adj.index[train_index]
        test_drugs = adj.index[test_index]

        Xtest = X_df.loc[X_df.index.get_level_values(1).isin(test_drugs)]
        ytest = y_df.loc[y_df.index.get_level_values(1).isin(test_drugs)]

        inner_mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        adj_train = pd.Series(1, index=df[df.index.get_level_values(1).isin(train_drugs_outer)].index).unstack(
            fill_value=0).astype(int).T
        if not best_config_exists:
            for j, (train_index, test_index) in enumerate(inner_mskf.split(adj_train.index, adj_train.values), 0):
                train_drugs = adj_train.index[train_index]
                val_drugs = adj_train.index[test_index]

                Xtrain, Xvalid = X_df.loc[X_df.index.get_level_values(1).isin(train_drugs)], X_df.loc[
                    X_df.index.get_level_values(1).isin(val_drugs)]
                ytrain, yvalid = y_df.loc[y_df.index.get_level_values(1).isin(train_drugs)], y_df.loc[
                    y_df.index.get_level_values(1).isin(val_drugs)]
                break
            configs.extend([[i]+list(conf)+[EPOCH] for conf in configs_list])
        else:
            #Xvalid and yvalid are assigned the Xtest and ytest values to use the same code without change
            #!! They are not used for training the best model.
            Xtrain, Xvalid = X_df.loc[X_df.index.get_level_values(1).isin(train_drugs)], Xtest
            ytrain, yvalid = y_df.loc[y_df.index.get_level_values(1).isin(train_drugs)], ytest

        data_fold_splits.append([Xtrain, ytrain, Xtest, ytest, Xvalid, yvalid])

    message = "Starting the evaluation of best models" if best_config_exists else "Starting the hyperparameter search...!!!"
    print_(message)
    scheduler(configs, seed, output_path, shared_metrics)

    dfs = []
    for metric in shared_metrics:
        dfs.append(metric.pop("df"))
    predictions = pd.concat(dfs)
    predictions.to_csv(os.path.join(output_path, "predictions.csv"))
    with open(completed_file, "w") as f:
        json.dump(list(shared_metrics), f, indent=2)

    print_("COMPLETED!!")