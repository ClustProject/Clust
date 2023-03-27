import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import copy
import random
from torch.utils.data import DataLoader, TensorDataset

from Clust.clust.transformation.purpose.machineLearning import LSTMData
from Clust.clust.ML.common import model_manager

# from Clust.clust.ML.regression_YK.interface import BaseRegressionModel
from Clust.clust.ML.regression_YK.models.rnn_model import RNNModel

# from Clust.clust.ML.common.common import p4_testing as p4

class RNNForecast():
    """
    RNN Forecasting model class
    """
    def __init__(self, params):
        """
        Init function of RNN forecasting class.

        Args:
            params (dict): parameters for building an RNN model
        """
        self.params = params
        # model 생성
        # TODO: parameters refactoring
        self.model = RNNModel(
            rnn_type=self.params['rnn_type'],
            input_size=self.params['trainParameter']['input_size'],
            hidden_size=self.params['trainParameter']['hidden_size'],
            num_layers=self.params['trainParameter']['num_layers'],
            output_dim = 1,     # TBD
            dropout_prob = 0.2,  # TBD
            bidirectional=False,    # TBD
            device=self.params['device']
        )

    def train(self, params, train_loader, valid_loader, num_epochs, device):
        """
        train function for the regression task.

        Args:
            params (dict): parameters for train
            train_loader (Dataloader): train data loader
            valid_loader (Dataloader): validation data loader
            num_epochs (integer): the number of train epochs
            device (string): device for train
        """
        self.model.to(device)

        # Training parameters?
        batch_size = params['batch_size']
        n_features = params['train_parameter']['input_size']
        weight_decay = 1e-6
        learing_rate = 1e-3
        self.loss_fn = nn.MSELoss(reduction="mean")
        self.optimizer = optim.Adam(self.model.parameters(), lr=learing_rate, weight_decay=weight_decay)

        self.train_losses = []
        self.val_losses = []

        since = time.time()

        for epoch in range(1, num_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                loss = self._train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in valid_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch <= 10) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{num_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )

        # 전체 학습 시간 계산
        time_elapsed = time.time() - since
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def test(self, params, test_loader, device):
        """
        Predict regression result for test dataset based on the trained model

        Args:
            params (dict): parameters for test  # TBD
            test_loader (DataLoader): data loader
            device (string): device for test

        Returns:
            preds (ndarray): prediction data
            trues (ndarray): original data
            mse (float): mean square error  # TBD
            mae (float): mean absolute error    # TBD
        """
        self.model.eval()   # 모델을 validation mode로 설정
        
        # test_loader에 대하여 검증 진행 (gradient update 방지)
        with torch.no_grad():
            preds, trues = [], []

            for x_test, y_test in test_loader:

                x_test = x_test.view([params['batch_size'], -1, len(params['transformParameter']['feature_col'])]).to(device)
                y_test = y_test.to(device)

                self.model.to(device)

                outputs = self.model(x_test)

                preds.extend(outputs.detach().numpy())
                trues.extend(y_test.detach().numpy())

        preds = np.array(preds).reshape(-1)
        trues = np.array(trues).reshape(-1)

        return preds, trues

    def inference(self, params, inference_loader, device):
        """
        Predict regression result for inference dataset based on the trained model

        Args:
            params (dict): parameters for inference     # TBD
            inference_loader (DataLoader): inference data loader
            device (string): device for inference

        Returns:
            preds (ndarray) : Inference result data
        """
        self.model.eval()   # 모델을 validation mode로 설정
        
        # test_loader에 대하여 검증 진행 (gradient update 방지)
        with torch.no_grad():
            preds = []

            for input in inference_loader:

                inputs = input.to(device)

                self.model.to(device)

                outputs = self.model(inputs)

                preds.extend(outputs.detach().numpy())

        preds = np.array(preds).reshape(-1)

        print(f'** Dimension of result for inference dataset = {preds.shape}')

        return preds

    def export_model(self):
        """
        export trained model 

        Returns:
            self.model (Object): current model object
        """
        return self.model

    def save_model(self, save_path):
        """
        save model to save_path

        Args:
            save_path (string): path to save model
        """
        model_manager.save_pickle_model(self.model, save_path)

    def load_model(self, model_file_path):
        """
        load model from model_file_path

        Args:
            model_file_path (string): path to load saved model
        """
        self.model = model_manager.load_pickle_model(model_file_path)

    # move to utils?
    # for train data
    def create_trainloader(self, batch_size, train, val):
        """
        Create train/valid data loader for torch

        Args:
            batch_size (integer): batch size
            train (dataframe): train X data
            val (dataframe): validation X data

        Returns:
            train_loader (DataLoader): train data loader
            val_loader (DataLoader): validation data loader
        """
        LSTMD = LSTMData()
        trainX_arr, trainy_arr = LSTMD.transform_Xy_arr(train, self.params['transform_parameter'], self.params['clean_param'])
        valX_arr, valy_arr = LSTMD.transform_Xy_arr(val, self.params['transform_parameter'], self.params['clean_param'])

        datasets = []
        for dataset in [(trainX_arr, trainy_arr), (valX_arr, valy_arr)]:
            X_arr = dataset[0]
            y_arr = dataset[1]
            datasets.append(TensorDataset(torch.Tensor(X_arr), torch.Tensor(y_arr)))

        train_set, val_set = datasets[0], datasets[1]
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True)

        return train_loader, val_loader

    # for test data
    def create_testloader(self, batch_size, test_X):
        """
        Create test data loader for torch

        Args:
            batch_size (integer): batch size
            test_x (dataframe): test X data
        
        Returns:
            test_loader (DataLoader) : test data loader
        """
        LSTMD = LSTMData()
        testX_arr, testy_arr = LSTMD.transform_Xy_arr(test_X, self.params['transformParameter'], self.params['cleanTrainDataParam'])
        features = torch.Tensor(testX_arr)
        targets = torch.Tensor(testy_arr)

        test_set = TensorDataset(features, targets)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)
        print("features shape:", features.shape, "targets shape: ", targets.shape)

        return test_loader

    # for inference data
    def create_inferenceloader(self, batch_size, x_data):
        """
        Create inference data loader for torch

        Args:
            batch_size (integer): batch size
            x_data (dataframe): inference X data
            window_num (integer): slice window number
        
        Returns:
            inference_loader (DataLoader) : inference data loader
        """
        x_data = x_data.values.astype(np.float32)
        x_data = x_data.reshape((-1, x_data.shape[0], x_data.shape[1]))

        x_data = torch.Tensor(x_data)
        inference_loader = DataLoader(x_data, batch_size=batch_size, shuffle=False, drop_last=True)

        return inference_loader

    # customized funtions
    def _train_step(self, x, y):
        """The method train_step completes one step of training.

        Given the features (x) and the target values (y) tensors, the method completes
        one step of the training. First, it activates the train mode to enable back prop.
        After generating predicted values (yhat) by doing forward propagation, it calculates
        the losses by using the loss function. Then, it computes the gradients by doing
        back propagation and updates the weights by calling step() function.

        Args:
            x (torch.Tensor): Tensor for features to train one step
            y (torch.Tensor): Tensor for target values to calculate losses

        """
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat = self.model(x)

        # Computes loss
        loss = self.loss_fn(y, yhat)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()