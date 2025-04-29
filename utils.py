import torch
import numpy as np
from datetime import datetime
import torch.utils.data as torch_data_utils
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import scipy
import pandas as pd


def get_device(cuda=7):
    """return device string either cpu or cuda:0"""
    if torch.cuda.is_available():
        device = 'cuda:'+str(cuda) #'cuda:0'
    else:
        device = 'cpu'
    print('current device={:}'.format(device))
    return device


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


def evaluate_model(df, y_true_col, y_pred_col, extra_meta=None, start_time = None, verbose=True):
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
        result['Execution Time']=(end_time-start_time).total_seconds()

    if verbose:
        print(result)

    return result


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


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        eps = 1e-6
        criterion = torch.nn.MSELoss()
        loss = torch.sqrt(criterion(x, y) + eps)
        return loss


class EarlyStopping: #ref:https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
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