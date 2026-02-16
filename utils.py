import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import scipy
import torch
import torch.utils.data as torch_data_utils
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from process_drug_data import get_morgan_fingerprint


def get_device(cuda=None):
    """return device string either cpu or cuda:0"""
    if torch.cuda.is_available():
        if cuda is None:
            cuda = get_gpu_with_max_free_memory()
        elif cuda == -1:
            return "cpu"
        elif cuda >= torch.cuda.device_count():
            print(f"Specified Cuda:{cuda} does not exist. Falling back to CPU.")
            return "cpu"
        device = 'cuda:' + str(cuda)
    else:
        if not (cuda is None or cuda == -1):
            print(f"Cuda not available. Falling back to CPU")
        device = 'cpu'
    print('current device={:}'.format(device))
    return device


def get_gpu_with_max_free_memory():
    best_gpu = None
    max_free_mem = 0

    for i in range(torch.cuda.device_count()):
        stats = torch.cuda.mem_get_info(i)  # (free, total)
        free_memory = stats[0] / (1024 ** 2)  # convert bytes -> MiB

        if free_memory > max_free_mem:
            max_free_mem = free_memory
            best_gpu = i

    return best_gpu


def set_seed(seed=42):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def cal_time(end, start):
    """return time spent"""
    # end = datetime.now(), start = datetime.now()
    datetimeFormat = '%Y-%m-%d %H:%M:%S.%f'
    spend = datetime.strptime(str(end), datetimeFormat) - \
            datetime.strptime(str(start), datetimeFormat)
    return spend


def weight_initializer(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class NumpyDataset(torch_data_utils.Dataset):
    """
    Return torch dataset, Given X and y numpy array
    """

    def __init__(self, X_arr, y_arr):
        self.X = X_arr
        self.y = y_arr
        self.y = self.y.reshape((len(self.y), 1))

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


def evaluate_model(df, y_true_col, y_pred_col, extra_meta=None, start_time=None, verbose=True):
    """
    return RMSE, R2, PCC
    """
    metrics = {'MAE': [], 'MSE': [], 'RMSE': [], 'R2': [], 'PCC': []}

    for fold, group in df.groupby('fold'):
        y_true = group[y_true_col]
        y_pred = group[y_pred_col]

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        pcc, _ = scipy.stats.pearsonr(y_true, y_pred)

        metrics['MAE'].append(mae)
        metrics['MSE'].append(mse)
        metrics['RMSE'].append(rmse)
        metrics['R2'].append(r2)
        metrics['PCC'].append(pcc)

    result = {}
    for metric, values in metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        result[metric] = f"{mean:.3f} ± {std:.3f}"

    if extra_meta:
        result = result | extra_meta

    if start_time:
        end_time = datetime.now()
        result['Execution Time'] = (end_time - start_time).total_seconds()

    if verbose:
        print(result)

    return result


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        eps = 1e-6
        criterion = torch.nn.MSELoss()
        loss = torch.sqrt(criterion(x, y) + eps)
        return loss


class EarlyStopping:  # ref:https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                print(f'Early stopping reached maximum patience.')
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def parse_pathway(raw_file_path):
    pathways_dict = {}
    unique_genes = set()
    with open(raw_file_path, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            genes = [gene.strip() for gene in line[2:]]
            pathways_dict[line[0].strip()] = genes
            unique_genes = unique_genes.union(genes)

    return pathways_dict, unique_genes


def get_pathway_genes(file_path):
    with open(file_path, "r") as f:
        pathways = json.load(f)
    genes_list = []
    for gene in pathways.values():
        genes_list.extend(gene)

    return pathways, set(genes_list)


def get_input_matrix(input_dir_path: str, mfp_n_bits: int = 256, raw=False) -> pd.DataFrame:
    """
    The function reads smiles data from provided input dir then generates morgan fingerprint.
    It finally gives in input matrix with all the features and the response column.
    :param input_dir_path: The directory path to the folder containing preprocessed data in matrix format. This is
    generated from the raw input data processing script.
    :param mfp_n_bits: Number of bits for Morgan Fingerprint. Default is 256
    :return: Returns a dataframe with feature and target values
    """
    # These files are expected to be in the provided input folder
    if raw or not os.path.exists(os.path.join(input_dir_path, "cell_line_feat.tsv")):
        cnv_mat = pd.read_csv(os.path.join(input_dir_path, "cnv_mat.csv"), index_col=0).rename(
            columns=lambda x: f"CNV_{x}")
        mut_mat = pd.read_csv(os.path.join(input_dir_path, "mut_mat.csv"), index_col=0).rename(
            columns=lambda x: f"MUT_{x}")
        exp_mat = pd.read_csv(os.path.join(input_dir_path, "exp_mat.csv"), index_col=0).rename(
            columns=lambda x: f"EXP_{x}")
        cell_line_feat = pd.concat([cnv_mat, mut_mat, exp_mat], axis=1)
    else:
        cell_line_feat = pd.read_csv(os.path.join(input_dir_path, "cell_line_feat.csv"), index_col=0)
    drug_response = pd.read_csv(os.path.join(input_dir_path, "drug_response.csv"))
    drug_smiles = pd.read_csv(os.path.join(input_dir_path, "drug_smiles.csv"), index_col=0)

    drug_mfp = drug_smiles.apply(lambda x: get_morgan_fingerprint([x], mfp_n_bits))
    feat_mat = (
        drug_response
        .merge(cell_line_feat, left_on='COSMIC_ID', right_index=True)
        .merge(drug_mfp, left_on="DRUG_ID", right_index=True)
        .set_index(["COSMIC_ID", "DRUG_ID"])
    )

    return feat_mat
