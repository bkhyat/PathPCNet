import torch
import numpy as np

from utils import RMSELoss, EarlyStopping, weight_initializer

class PathPCNet(torch.nn.Module):
    def __init__(self, n_inputs, dropout_rate=0.1):
        super(PathPCNet, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, 1000),
            torch.nn.ELU(),
            torch.nn.Dropout(p=dropout_rate),

            torch.nn.Linear(1000, 800),
            torch.nn.ELU(),
            torch.nn.Dropout(p=dropout_rate),

            torch.nn.Linear(800, 500),
            torch.nn.ELU(),
            torch.nn.Dropout(p=dropout_rate),

            torch.nn.Linear(500, 100),
            torch.nn.ELU(),
            torch.nn.Dropout(p=dropout_rate),

            torch.nn.Linear(100, 1)
        )

    def init_weights(self):
        self.apply(weight_initializer)

    def forward(self, x):
        return self.model(x)


    def fit(self, train_dl, valid_dl, epochs, learning_rate, device, opt_fn, path="checkpoint.pt", verbose=False):
        """
        Return train and valid performance including loss

        :param net: model
        :param train_dl: train dataloader
        :param valid_dl: valid dataloader
        :param epochs: integer representing EPOCH
        :param learning_rate: float representing LEARNING_RATE
        :param device: string representing cpu or cuda:0
        :param opt_fn: optimization function in torch (e.g., tch.optim.Adam)
        :param path: string representing path
        :param verbose: bool
        """
        # setup
        criterion = RMSELoss()  # setup LOSS function
        optimizer = opt_fn(self.parameters(), lr=learning_rate, weight_decay=1e-5)  # setup optimizer
        net = self.to(device)  # load the network onto the device
        train_loss_list = []  # metrics: MSE, size equals to EPOCH
        valid_loss_list = []  # metrics: MSE, size equals to EPOCH
        early_stopping = EarlyStopping(patience=30, verbose=verbose, path=path)  # initialize the early_stopping
        # repeat the training for EPOCH times
        for epoch in range(epochs):
            ## training phase
            net.train()
            # initial loss
            train_epoch_loss = 0.0  # save loss for each epoch, batch by batch
            for i, (X_train, y_train) in enumerate(train_dl):
                X_train, y_train = X_train.to(device), y_train.to(device)  # load data onto the device
                y_train_pred = net(X_train)  # train result
                train_loss = criterion(y_train_pred, y_train.float())  # calculate loss

                optimizer.zero_grad()  # clear gradients
                train_loss.backward()  # backpropagation
                #### add this if you have gradient explosion problem ###
                clip_value = 5
                torch.nn.utils.clip_grad_value_(net.parameters(), clip_value)
                ########climp gradient within -5 ~ 5 ###################
                optimizer.step()  # update weights
                train_epoch_loss += train_loss.item()  # adding loss from each batch
            # calculate total loss of all batches
            avg_train_loss = train_epoch_loss / len(train_dl)
            train_loss_list.append(avg_train_loss)
            ## validation phase
            with torch.no_grad():
                net.eval()
                valid_epoch_loss = 0.0  # save loss for each epoch, batch by batch
                for i, (X_valid, y_valid) in enumerate(valid_dl):
                    X_valid, y_valid = X_valid.to(device), y_valid.to(device)  # load data onto the device
                    y_valid_pred = net(X_valid)  # valid result
                    valid_loss = criterion(y_valid_pred, y_valid.float())  # y_valid.unsqueeze(1)) # calculate loss
                    valid_epoch_loss += valid_loss.item()  # adding loss from each batch
            # calculate total loss of all batches, and append to result list
            avg_valid_loss = valid_epoch_loss / len(valid_dl)
            valid_loss_list.append(avg_valid_loss)
            if verbose:
                print(f"Epoch: {epoch} Training Loss:{avg_train_loss}, Validation Loss: {avg_valid_loss}")

            early_stopping(avg_valid_loss, net)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        # load the last checkpoint with the best model
        net.load_state_dict(torch.load(path))

        return train_loss_list, valid_loss_list

    def predict(self, test_dl, device):
        """
        Return prediction list

        :param net: model
        :param train_dl: train dataloader
        :param device: string representing cpu or cuda:0
        """
        # create result lists
        prediction_list = list()

        with torch.no_grad():
            net = self.to(device)  # load the network onto the device
            net.eval()
            for i, (X_test, y_test) in enumerate(test_dl):
                X_test, y_test = X_test.to(device), y_test.to(device)  # load data onto the device
                y_test_pred = net(X_test)  # test result
                # bring data back to cpu in np.array format, and append to result lists
                prediction_list.append(y_test_pred.cpu().numpy())
                # print(prediction_list)

        # merge all batches
        prediction_list = np.vstack(prediction_list)
        prediction_list = np.hstack(prediction_list).tolist()
        # return
        return prediction_list