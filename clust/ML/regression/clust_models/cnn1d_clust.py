import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import copy
from torch.utils.data import DataLoader, TensorDataset

from Clust.clust.ML.tool import model as ml_model

from Clust.clust.ML.regression.interface import BaseRegressionModel
from Clust.clust.ML.regression.models.cnn_1d import CNN1D


class CNN1DClust(BaseRegressionModel):
    """
    CNN1D Regression & forecast model class
    """
    def __init__(self, model_params):
        """
        Init function of CNN1D class.

        Args:
            model_params (dict): parameters for building an CNN1D model
        """
        self.model_params = model_params
        # model 생성
        self.model = CNN1D(
            input_channels = self.model_params['input_size'],
            input_seq = self.model_params['seq_len'],
            output_channels = self.model_params['output_channels'],
            kernel_size = self.model_params['kernel_size'],
            stride = self.model_params['stride'],
            padding = self.model_params['padding'],
            drop_out = self.model_params['dropout']
        )

    def train(self, train_params, train_loader, valid_loader):
        """
        train function for the regression task.

        Args:
            train_params (dict): parameters for train
            train_loader (Dataloader): train data loader
            valid_loader (Dataloader): validation data loader
        """
        device = train_params['device']
        epochs = train_params['n_epochs']
        batch_size = train_params['batch_size']
        n_features = self.model_params['input_size']

        self.model.to(device)

        self.loss_fn = nn.MSELoss(reduction="mean")
        self.optimizer = optim.Adam(self.model.parameters(), lr=train_params['lr'], weight_decay=train_params['weight_decay'])

        self.train_losses = []
        self.val_losses = []

        since = time.time()

        for epoch in range(1, epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features])
                x_batch = x_batch.transpose(1, 2).to(device)    # Conv-1d condition
                y_batch = y_batch.view([batch_size, 1]).to(device)
                loss = self._train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in valid_loader:
                    x_val = x_val.view([batch_size, -1, n_features])
                    x_val = x_val.transpose(1, 2).to(device)    # Conv-1d condition
                    y_val = y_val.view([batch_size, 1]).to(device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch <= 10) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )

        # 전체 학습 시간 계산
        time_elapsed = time.time() - since
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def test(self, test_params, test_loader):
        """
        Predict regression result for test dataset based on the trained model

        Args:
            test_params (dict): parameters for test
            test_loader (DataLoader): data loader

        Returns:
            preds (ndarray): prediction data
            trues (ndarray): original data
        """
        device = test_params['device']
        batch_size = test_params['batch_size']
        n_features = self.model_params['input_size']

        self.model.eval()   # 모델을 validation mode로 설정
        
        # test_loader에 대하여 검증 진행 (gradient update 방지)
        with torch.no_grad():
            preds, trues = [], []

            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features])
                x_test = x_test.transpose(1, 2).to(device)    # Conv-1d condition
                y_test = y_test.view([batch_size, 1]).to(device, dtype=torch.float)

                self.model.to(device)
                
                # forward
                # input을 model에 넣어 output을 도출
                outputs = self.model(x_test)
                
                # 예측 값 및 실제 값 축적
                preds.extend(outputs.detach().cpu().numpy())
                trues.extend(y_test.detach().cpu().numpy())

        preds = np.array(preds).reshape(-1)
        trues = np.array(trues)

        return preds, trues

    def inference(self, infer_params, inference_loader):
        """
        Predict regression result for inference dataset based on the trained model

        Args:
            infer_params (dict): parameters for inference
            inference_loader (DataLoader): inference data loader

        Returns:
            preds (ndarray) : Inference result data
        """
        device = infer_params['device']
        batch_size = infer_params['batch_size']
        n_features = self.model_params['input_size']

        self.model.eval()   # 모델을 validation mode로 설정
        
        # test_loader에 대하여 검증 진행 (gradient update 방지)
        with torch.no_grad():
            preds = []
            for x_infer in inference_loader:
                x_infer = x_infer.view([batch_size, -1, n_features])
                x_infer = x_infer.transpose(1, 2).to(device)    # Conv-1d condition
                
                self.model.to(device)
                
                # forward
                # input을 model에 넣어 output을 도출
                outputs = self.model(x_infer)
                
                # 예측 값 및 실제 값 축적
                preds.extend(outputs.detach().cpu().numpy())

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
        ml_model.save_pickle_model(self.model, save_path)

    def load_model(self, model_file_path):
        """
        load model from model_file_path

        Args:
            model_file_path (string): path to load saved model
        """
        self.model = ml_model.load_pickle_model(model_file_path)

    # for train data
    def create_trainloader(self, batch_size, train_x, train_y, val_x, val_y):
        """
        Create train/valid data loader for torch

        Args:
            batch_size (integer): batch size
            train_x (dataframe): train X data
            train_y (dataframe): train y data
            val_x (dataframe): validation X data
            val_y (dataframe): validation y data

        Returns:
            train_loader (DataLoader): train data loader
            val_loader (DataLoader): validation data loader
        """
        datasets = []
        for dataset in [(train_x, train_y), (val_x, val_y)]:
            x_data = np.array(dataset[0])
            y_data = dataset[1]
            datasets.append(TensorDataset(torch.Tensor(x_data), torch.Tensor(y_data)))

        train_set, val_set = datasets[0], datasets[1]

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True)

        return train_loader, val_loader


    # for test data
    def create_testloader(self, batch_size, test_x, test_y):
        """
        Create test data loader for torch

        Args:
            batch_size (integer): batch size
            test_x (dataframe): test X data
            test_y (dataframe): test y data
        
        Returns:
            test_loader (DataLoader) : test data loader
        """
        test_data = TensorDataset(torch.Tensor(test_x), torch.Tensor(test_y))
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)

        return test_loader

    # for inference data
    def create_inferenceloader(self, batch_size, infer_x):
        """
        Create inference data loader for torch

        Args:
            batch_size (integer): batch size
            infer_x (dataframe): inference X data
        
        Returns:
            inference_loader (DataLoader) : inference data loader
        """

        infer_x = torch.Tensor(infer_x)
        inference_loader = DataLoader(infer_x, batch_size=batch_size, shuffle=True)

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
