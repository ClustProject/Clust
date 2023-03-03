import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import copy
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error 

from Clust.clust.transformation.type.DFToNPArray import transDFtoNP, trans_df_to_np, trans_df_to_np_inf
from Clust.clust.ML.common import model_manager

from Clust.clust.ML.regression_YK.interface import BaseRegressionModel
from Clust.clust.ML.regression_YK.models.cnn_1d import CNN1D


class CNN1DClust(BaseRegressionModel):
    def __init__(self, param):
        """
        CNN1D Regression class
        """
        self.param = param
        # model 생성
        self.model = CNN1D(
            input_channels=self.param['input_size'],
            input_seq=self.param['seq_len'],
            output_channels=self.param['output_channels'],
            kernel_size=self.param['kernel_size'],
            stride=self.param['stride'],
            padding=self.param['padding'],
            drop_out=self.param['drop_out']
        )

    def train(self, param, train_loader, valid_loader, num_epochs, device):
        """
        train method for RNNClust

        Args:
            param (dict): parameters for train
            train_loader (Dataloader): train data loader
            valid_loader (Dataloader): validation data loader
            num_epochs (integer): the number of train epochs
            device (string): device for train
        """
        self.model.to(device)

        data_loaders_dict = {'train': train_loader, 'val': valid_loader}
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=param['lr'])

        since = time.time()
        val_mse_history = []

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_mse = 10000000

        for epoch in range(num_epochs):
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print()
                print('Epoch {}/{}'.format(epoch + 1, num_epochs))

            # 각 epoch마다 순서대로 training과 validation을 진행
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # 모델을 training mode로 설정
                else:
                    self.model.eval()   # 모델을 validation mode로 설정

                running_loss = 0.0
                running_total = 0

                # training과 validation 단계에 맞는 dataloader에 대하여 학습/검증 진행
                for inputs, labels in data_loaders_dict[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device, dtype=torch.float)
                    
                    # parameter gradients를 0으로 설정
                    optimizer.zero_grad()

                    # forward
                    # training 단계에서만 gradient 업데이트 수행
                    with torch.set_grad_enabled(phase == 'train'):
                        # input을 model에 넣어 output을 도출한 후, loss를 계산함
                        outputs = self.model(inputs)
                        outputs = outputs.squeeze(1)
                        loss = criterion(outputs, labels)

                        # backward (optimize): training 단계에서만 수행
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # batch별 loss를 축적함
                    running_loss += loss.item() * inputs.size(0)
                    running_total += labels.size(0)

                # epoch의 loss 및 accuracy 도출
                epoch_loss = running_loss / running_total

                if epoch == 0 or (epoch + 1) % 10 == 0:
                    print('{} Loss: {:.4f}'.format(phase, epoch_loss))

                # validation 단계에서 validation loss가 감소할 때마다 best model 가중치를 업데이트함
                if phase == 'val' and epoch_loss < best_mse:
                    best_mse = epoch_loss
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                if phase == 'val':
                    val_mse_history.append(epoch_loss)

        # 전체 학습 시간 계산
        time_elapsed = time.time() - since
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val MSE: {:4f}'.format(best_mse))

        # validation loss가 가장 낮았을 때의 best model 가중치를 불러와 best model을 구축함
        self.model.load_state_dict(best_model_wts)

    def test(self, param, test_loader, device):
        """
        Predict RegressionResult for test dataset based on the trained model

        Args:
            test_loader (DataLoader) : data loader
            device (string) : device for test

        Returns:
            preds (ndarray) : prediction data
            trues (ndarray) : original data
            mse (float) : mean square error
            mae (float) : mean absolute error
        """
        self.model.eval()   # 모델을 validation mode로 설정
        
        # test_loader에 대하여 검증 진행 (gradient update 방지)
        with torch.no_grad():
            trues, preds = [], []
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device, dtype=torch.float)

                self.model.to(device)
                
                # forward
                # input을 model에 넣어 output을 도출
                outputs = self.model(inputs)
                
                # 예측 값 및 실제 값 축적
                trues.extend(labels.detach().cpu().numpy())
                preds.extend(outputs.detach().cpu().numpy())

        preds = np.array(preds).reshape(-1)
        trues = np.array(trues)

        mse = mean_squared_error(trues, preds)
        mae = mean_absolute_error(trues, preds)

        print(f'** Performance of test dataset ==> MSE = {mse}, MAE = {mae}')
        print(f'** Dimension of result for test dataset = {preds.shape}')

        return preds, trues, mse, mae

    def inference(self, param, inference_loader, device):
        """

        Args:
            model (model): load trained model
            test_loader (DataLoader) : data loader

        Returns:
            preds (ndarray) : Inference result data
        """
        self.model.eval()   # 모델을 validation mode로 설정
        
        # test_loader에 대하여 검증 진행 (gradient update 방지)
        with torch.no_grad():
            preds = []
            for inputs in inference_loader:
                self.model.to(device)
                
                # forward
                # input을 model에 넣어 output을 도출
                outputs = self.model(inputs)
                
                # 예측 값 및 실제 값 축적
                preds.extend(outputs.detach().cpu().numpy())

        preds = np.array(preds).reshape(-1)

        print(f'** Dimension of result for inference dataset = {preds.shape}')

        return preds

    def export_model(self):
        """
        export trained model 
        """
        return self.model

    def save_model(self, save_path):
        """
        save model to save_path
        """
        model_manager.save_pickle_model(self.model, save_path)

    def load_model(self, model_file_path):
        """
        load model from model_file_path
        """
        self.model = model_manager.load_pickle_model(model_file_path)

    # move to utils?
    # for train data
    def create_trainloader(self, batch_size, train_x, train_y, val_x, val_y, window_num):
        """
        Create train/valid data loader for torch

        Args:
            batch_size (integer): 
            train_x (dataframe): train X data
            train_y (dataframe): train y data
            val_x (dataframe): validation X data
            val_y (dataframe): validation y data

        Returns:
            train_loader (DataLoader):
            val_loader (DataLoader):
            input_size (integer):
            seq_len (integer):
        """
        train_x, train_y = transDFtoNP(train_x, train_y, window_num)
        val_x, val_y = transDFtoNP(val_x, val_y, window_num)

        # input_size = train_x.shape[1]
        # seq_len = train_x.shape[2]

        datasets = []
        for dataset in [(train_x, train_y), (val_x, val_y)]:
            x_data = np.array(dataset[0])
            y_data = dataset[1]
            datasets.append(TensorDataset(torch.Tensor(x_data), torch.Tensor(y_data)))

        train_set, val_set = datasets[0], datasets[1]

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

        return train_loader, val_loader

    # for test data
    def create_testloader(self, batch_size, test_x, test_y, window_num):
        """

        Returns:
            test_loader (DataLoader) : data loader
        """
        test_x, test_y = trans_df_to_np(test_x, test_y, window_num)

        test_data = TensorDataset(torch.Tensor(test_x), torch.Tensor(test_y))
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

        return test_loader

    # for inference data
    def create_inferenceloader(self, batch_size, x_data, window_num):
        """

        Returns:
            data (Array): input data for inference
        """
        x_data = trans_df_to_np_inf(x_data, window_num)

        x_data = torch.Tensor(x_data)
        inference_loader = DataLoader(x_data, batch_size=batch_size, shuffle=True)

        return inference_loader
