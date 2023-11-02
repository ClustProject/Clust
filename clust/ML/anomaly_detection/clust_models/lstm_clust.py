import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
import pandas as pd 

import time
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

from torch.utils.data import DataLoader, TensorDataset

from Clust.clust.ML.tool import model as ml_model
from Clust.clust.ML.anomaly_detection.interface import BaseAnomalyDetModel
from Clust.clust.ML.anomaly_detection.models.lstm_vae import LSTMVAE

import warnings
warnings.filterwarnings(action='ignore')



class LSTMClust(BaseAnomalyDetModel):
    """
    RNN Regression & forecast model class
    """
    def __init__(self, model_params):
        """
        Init function of RNN class.

        Args:
            model_params (dict): parameters for building an RNN model
        """
        # model 생성
        # TODO: parameters refactoring
        self.model_params = model_params
        self.model = LSTMVAE(
            input_size = self.model_params['input_size'],
            hidden_size = self.model_params['hidden_size'],
            latent_size = self.model_params['latent_size']
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
        epochs = train_params['num_epochs']
        batch_size = train_params['batch_size']
        n_features = self.model_params['input_size']

        self.model.to(device)

        self.loss_fn = nn.MSELoss(reduction="mean")
        self.optimizer = optim.Adam(self.model.parameters(), lr=train_params['lr'])

        self.train_losses = []
        self.val_losses = []

        since = time.time()

        for epoch in range(1, epochs + 1):
            batch_losses = []
            for x_batch in train_loader:
                x_batch = x_batch.to(device)
                loss = self._train_step(x_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val in valid_loader:
                    x_val = x_val.to(device)
                    self.model.eval()
                    yhat, val_loss = self.model(x_val)
                    batch_val_losses.append(val_loss.item())
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch <= 10) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )

        # 전체 학습 시간 계산
        time_elapsed = time.time() - since
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def test(self, test_params, test_loader, train_loader_threshold):
        """
        Predict result for test dataset based on the trained model

        Args:
            test_params (dict): parameters for test
            test_loader (DataLoader): data loader
            train_loader_threhsold (Dataloader) : data loader

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
            trues, pred_y, pred_train_y = [], [], []

            for x_test, y_test in test_loader:
                x_test = x_test.to(device)
                y_test = y_test.view([1, 1]).to(device, dtype=torch.float)
                
                self.model.to(device)
                
                # forward
                # input을 model에 넣어 output을 도출
                outputs = self.model(x_test)
                
                # 예측 값 및 실제 값 축적
                pred_y.append(outputs[1].detach().cpu().numpy().item())
                trues.append(y_test.detach().cpu().numpy().item())

            for x_test in train_loader_threshold:
                x_test = x_test.to(device)
                self.model.to(device)

                # forward
                # input을 model에 넣어 output을 도출
                outputs = self.model(x_test)
                
                # 예측 값 및 실제 값 축적
                pred_train_y.append(outputs[1].detach().cpu().numpy().item())

        thrs = [70, 80, 85, 90, 95, 96, 97, 98, 99, 99.5]
        df = pd.DataFrame(columns=['Threshold','ACC','Precision','Recall','F1 Score'])

        for thres in thrs:
            threshold = np.percentile(pred_train_y, thres)
            pred_label = []
            for i in range(len(trues)):
                if pred_y[i]>threshold:
                    pred_label.append(1)
                else:
                    pred_label.append(0)
            
            accuracy = accuracy_score(trues, pred_label)

            precision, recall, f_score, support = precision_recall_fscore_support(trues, pred_label,average='binary')
            new = {'Threshold': thres, 'ACC': accuracy, 'Precision': precision, "Recall": recall, "F1 Score": f_score }
            df = df.append(new,ignore_index=True)

        print(df)

        max = df.loc[df['F1 Score'].idxmax()]

        return np.array(pred_label), np.array(trues), max[1], max[2], max[3], max[4]

    
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
        
        with torch.no_grad():
            preds = []

            for x_infer in inference_loader:
                x_infer = x_infer.view([batch_size, -1, n_features]).to(device)

                self.model.to(device)
                
                # forward
                # input을 model에 넣어 output을 도출
                outputs = self.model(x_infer)

                # 예측 값
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
    def create_trainloader(self, batch_size, train_x, val_x):
        """
        Create train/valid data loader for torch

        Args:
            batch_size (integer): batch size
            task (string): task (e.g., regression, forecast)
            train_x (dataframe): train X data
            train_y (dataframe): train y data (regression only)
            val_x (dataframe): validation X data
            val_y (dataframe): validation y data (regression only)

        Returns:
            train_loader (DataLoader): train data loader
            val_loader (DataLoader): validation data loader
        """
        train_data = self.make_window(train_x, window_size=100, shift_size=10)
        val_data = self.make_window(val_x, window_size=100, shift_size=10)

        train_data = torch.tensor(train_data, dtype=torch.float32)
        val_data = torch.tensor(val_data, dtype=torch.float32)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)

        return train_loader, val_loader

    # for test data
    def create_testloader(self, batch_size, train_x, test_x, test_y):
        """
        Create test data loader for torch

        Args:
            batch_size (integer): batch size
            train_x (np.array) : train X data 
            test_x (np.array): test X data
            test_y (np.array): test y data
        
        Returns:
            test_loader (DataLoader) : test data loader
        """

        # train loader for threshold 
        train_data = self.make_window(train_x, window_size=100, shift_size=10)
        train_data = torch.tensor(train_data, dtype=torch.float32)

        train_loader_threshold = DataLoader(train_data, batch_size=1, shuffle=False)

        # test loader
        features = self.make_window(test_x, window_size=100, shift_size=10)
        test_y_ = self.make_window_(test_y, window_size=100, shift_size=10)

        targets = np.zeros(test_y_.shape[0])
        for i in range(test_y_.shape[0]):
            if targets[i] == 1:
                targets[i] = 1
            else:
                targets[i] = 0

        features = torch.tensor(features, dtype=torch.float32)
        targets = torch.Tensor(targets)

        test_data = TensorDataset(features, targets)

        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

        return train_loader_threshold, test_loader

    # for inference data
    def create_inferenceloader(self, batch_size, infer_x):
        """
        Create inference data loader for torch

        Args:
            batch_size (integer): batch size
            infer_x (np.array): inference X data
        
        Returns:
            inference_loader (DataLoader) : inference data loader
        """
        # ensure input shape is [batch_size, seq_len, input_size]
        if len(infer_x.shape) != 3:
            infer_x = np.expand_dims(infer_x, axis=0)

        infer_x = torch.Tensor(infer_x)
        inference_loader = DataLoader(infer_x, batch_size=batch_size, shuffle=False)
        print("inference data shape:", infer_x.shape)

        return inference_loader

    # customized funtions
    def _train_step(self, x):
        """The method train_step completes one step of training.

        Given the features (x) and the target values (y) tensors, the method completes
        one step of the training. First, it activates the train mode to enable back prop.
        After generating predicted values (yhat) by doing forward propagation, it calculates
        the losses by using the loss function. Then, it computes the gradients by doing
        back propagation and updates the weights by calling step() function.

        Args:
            x (torch.Tensor): Tensor for features to train one step
        """
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat, loss = self.model(x)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()
    
    def make_window(self, data, window_size, shift_size):

        num_samples = (data.shape[0]-window_size)//shift_size
        print(num_samples)
        windowed_data = np.zeros((num_samples, window_size, data.shape[1]))
        for i in range(num_samples):
            windowed_data[i] = data.iloc[i*shift_size:i*shift_size+window_size,:]

        return windowed_data
    
    def make_window_(self, data, window_size, shift_size):

        num_samples = (data.shape[0]-window_size)//shift_size
        windowed_data = np.zeros((num_samples, window_size))
        for i in range(num_samples):
            windowed_data[i] = data[i*shift_size:i*shift_size+window_size]

        return windowed_data
