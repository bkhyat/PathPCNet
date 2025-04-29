import argparse
import json
import multiprocessing as mp
import os
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn.metrics as skmts
import sklearn.model_selection as skms
import sklearn.utils as skut
import torch
import torch.utils.data as torch_data_utils

from model import PathPCNet
from utils import set_seed, get_device, evaluate_model, NumpyDataset

# Constants
LEARNING_RATE = 0.0004
EPOCH = 2000
BATCH_SIZE = 12
GPU_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0
CHECK_POINT_PATH = "checkpoints"

ALL_PREFIXES = ("EXP", "CNV", "MUT")
FEATURE_PREFIXES = tuple(), ("EXP",), ("MUT",), ("CNV",), ("EXP", "MUT"), ("EXP", "CNV"), ("MUT", "CNV")
FEATURE_SUFFIXES = ("PC2", "PC3", "PC4"), ("PC3", "PC4"), ("PC4",), tuple()


# Argument parser
def parse_parameter():
    parser = argparse.ArgumentParser(description="Run parallel experiment on multiple GPUs")
    parser.add_argument("-i", "--input_path", required=True, help="input path")
    parser.add_argument("-s", "--seed_int", default=42, type=int, help="seed for reproducibility. "
                                                                       "default=42")
    parser.add_argument("-c", "--cv_int", default=10, type=int, help="K fold cross validation. default=10")
    parser.add_argument("-o", "--output_path", default="out", help="output path")
    return parser.parse_args()


# Worker for each GPU
def gpu_worker(gpu_id, task_queue, args, shared_metrics):
    device = get_device(cuda=gpu_id)

    while not task_queue.empty():
        try:
            prefix, suffix = task_queue.get_nowait()
        except:
            break

        run_training_job(prefix, suffix, device, args, shared_metrics)


def run_training_job(prefix, suffix, device, args, shared_metrics):
    start_time = datetime.now()
    n_pcs = 4 - len(suffix)
    feature_sets = tuple(p for p in ALL_PREFIXES if p not in prefix)
    comb_name = "+".join(feature_sets) + f"_PC{n_pcs}"

    print(f"[GPU {device}] Starting {comb_name}")

    df = pd.read_csv(args.input_path, index_col=[0, 1], header=0)

    if prefix:
        df.drop(columns=df.columns[df.columns.str.startswith(prefix)], inplace=True)
    if suffix:
        df.drop(columns=df.columns[df.columns.str.endswith(suffix)], inplace=True)

    sdf = skut.shuffle(df, random_state=args.seed_int)
    set_seed(args.seed_int)

    kFold = args.cv_int
    opt_fn = torch.optim.Adam

    kf = skms.KFold(n_splits=kFold, random_state=args.seed_int, shuffle=True)

    X_df = sdf.iloc[:, 1:]
    y_df = sdf.iloc[:, 0]

    best_rmse = 1e10
    best_model = None
    best_fold = 0

    loss_df_list = []
    ytest_df_list = []

    for i, (train_index, test_index) in enumerate(kf.split(X_df, y_df)):
        n_fold = i + 1
        print(f'[GPU {device}] Fold={n_fold}/{args.cv_int}')

        Xtrain_arr = X_df.values[train_index]
        Xtest_arr = X_df.values[test_index]
        ytrain_arr = y_df.values[train_index]
        ytest_arr = y_df.values[test_index]

        Xtrain_arr, Xvalid_arr, ytrain_arr, yvalid_arr = skms.train_test_split(
            Xtrain_arr, ytrain_arr, test_size=0.1, random_state=args.seed_int)

        Xtrain_arr = np.array(Xtrain_arr).astype('float32')
        Xvalid_arr = np.array(Xvalid_arr).astype('float32')
        Xtest_arr = np.array(Xtest_arr).astype('float32')
        ytrain_arr = np.array(ytrain_arr).astype('float32')
        yvalid_arr = np.array(yvalid_arr).astype('float32')
        ytest_arr = np.array(ytest_arr).astype('float32')

        train_dataset = NumpyDataset(torch.from_numpy(Xtrain_arr), torch.from_numpy(ytrain_arr))
        valid_dataset = NumpyDataset(torch.from_numpy(Xvalid_arr), torch.from_numpy(yvalid_arr))
        test_dataset = NumpyDataset(torch.from_numpy(Xtest_arr), torch.from_numpy(ytest_arr))

        train_dl = torch_data_utils.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_dl = torch_data_utils.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_dl = torch_data_utils.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        n_features = Xtrain_arr.shape[1]
        net = PathPCNet(n_features)
        net.init_weights()

        train_loss_list, valid_loss_list = net.fit(train_dl, valid_dl, EPOCH, LEARNING_RATE, device,
                                                   opt_fn,
                                                   os.path.join(args.output_path, CHECK_POINT_PATH, comb_name + ".pt"))
        prediction_list = net.predict(test_dl, device)

        mse = skmts.mean_squared_error(ytest_arr, prediction_list)
        rmse = np.sqrt(mse)

        if rmse <= best_rmse:
            best_rmse = rmse

        loss_df = pd.DataFrame({'fold': [n_fold] * len(train_loss_list),
                                'epoch': [i + 1 for i in range(len(train_loss_list))],
                                'train loss': train_loss_list,
                                'valid loss': valid_loss_list})
        ytest_df = y_df.iloc[test_index].to_frame()
        ytest_df['prediction'] = prediction_list
        ytest_df['fold'] = n_fold

        loss_df_list.append(loss_df)
        ytest_df_list.append(ytest_df)

    all_ytest_df = pd.concat(ytest_df_list, axis=0)
    all_loss_df = pd.concat(loss_df_list, axis=0)

    os.makedirs(args.output_path, exist_ok=True)
    all_ytest_df.to_csv(os.path.join(args.output_path, comb_name + '_prediction.csv'), header=True)
    all_loss_df.to_csv(os.path.join(args.output_path, comb_name + '_loss.csv'), header=True)

    metrics = evaluate_model(all_ytest_df, 'LN_IC50', 'prediction',
                             {"data": "+".join(feature_sets), "N_PCs": n_pcs},
                             start_time)
    shared_metrics.append(metrics)

    # with open(os.path.join(args.output_path, f"{comb_name}_metrics.json"), "w") as f:
    #     json.dump(metrics, f, indent=2)

    print(f"[GPU {device}] Finished {comb_name}")


if __name__ == "__main__":
    assert GPU_COUNT != 0, "No Cuda Available!"
    mp.set_start_method("spawn", force=True)

    args = parse_parameter()

    os.makedirs(os.path.join(args.output_path, CHECK_POINT_PATH), exist_ok=True)

    # Manager for shared list
    manager = mp.Manager()
    shared_metrics = manager.list()

    # Distribute tasks
    tasks = [[] for _ in range(GPU_COUNT)]
    count = 0
    for prefix in FEATURE_PREFIXES:
        for suffix in FEATURE_SUFFIXES:
            gpu_id = count % GPU_COUNT
            tasks[gpu_id].append((prefix, suffix))
            count += 1

    # Create queues
    queues = []
    for task_list in tasks:
        q = mp.Queue()
        for item in task_list:
            q.put(item)
        queues.append(q)

    # Start one worker per GPU
    workers = []
    for gpu_id in range(GPU_COUNT):
        p = mp.Process(target=gpu_worker, args=(gpu_id, queues[gpu_id], args, shared_metrics))
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    # After all workers finish, save the metrics
    metrics_output_path = os.path.join(args.output_path, "all_metrics.json")
    print(shared_metrics)
    with open(metrics_output_path, "w") as f:
        json.dump(list(shared_metrics), f, indent=2)

    print(f"All tasks completed. Metrics saved to {metrics_output_path}")
