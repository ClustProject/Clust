import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import copy
import datetime
from torch.utils.data import DataLoader, TensorDataset

from Clust.clust.transformation.type.DFToNPArray import transDFtoNP, trans_df_to_np, trans_df_to_np_inf
from Clust.clust.ML.tool import model as ml_model

from Clust.clust.ML.classification.interface import BaseRegressionModel
from Clust.clust.ML.classification.models.cnn_1d import CNN1D



class CNNModel(BaseRegressionModel):
    """

    """
    def __init__(self, model_params):
        """
        Init function of CNN1D regression class.

        Args:
            params (dict): parameters for building a CNN1D model
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
            drop_out = self.model_params['dropout'],
            num_classes = self.model_params['num_classes']
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
        n_epochs = train_params['n_epochs']
        batch_size = train_params['batch_size']
        input_size = self.model_params['input_size']

        self.model.to(device)

        data_loaders_dict = {'train': train_loader, 'val': valid_loader}
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=train_params['lr'])

        since = time.time()

        val_acc_history = []

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(n_epochs):
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print()
                print('Epoch {}/{}'.format(epoch + 1, n_epochs))

            # 각 epoch마다 순서대로 training과 validation을 진행
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # 모델을 training mode로 설정
                else:
                    self.model.eval()   # 모델을 validation mode로 설정

                running_loss = 0.0
                running_corrects = 0
                running_total = 0

                # training과 validation 단계에 맞는 dataloader에 대하여 학습/검증 진행
                for inputs, labels in data_loaders_dict[phase]:
                    inputs = inputs.view([batch_size, -1, input_size])
                    inputs = inputs.transpose(1, 2).to(device)  # Conv-1d condition
                    labels = labels.squeeze(dim=-1).to(device, dtype=torch.long)
                    # seq_lens = seq_lens.to(self.parameter['device'])
                    
                    # parameter gradients를 0으로 설정
                    optimizer.zero_grad()
                    
                    # forward
                    # training 단계에서만 gradient 업데이트 수행
                    with torch.set_grad_enabled(phase == 'train'):
                        # input을 model에 넣어 output을 도출한 후, loss를 계산함
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)

                        # output 중 최댓값의 위치에 해당하는 class로 예측을 수행
                        _, preds = torch.max(outputs, 1)

                        # backward (optimize): training 단계에서만 수행
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # batch별 loss를 축적함
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    running_total += labels.size(0)

                # epoch의 loss 및 accuracy 도출
                epoch_loss = running_loss / running_total
                epoch_acc = running_corrects.double() / running_total

                if epoch == 0 or (epoch + 1) % 10 == 0:
                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # validation 단계에서 validation loss가 감소할 때마다 best model 가중치를 업데이트함
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)


        # 전체 학습 시간 계산
        time_elapsed = time.time() - since
        timeElapsed = str(datetime.timedelta(seconds=time_elapsed))
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # validation loss가 가장 낮았을 때의 best model 가중치를 불러와 best model을 구축함
        self.model.load_state_dict(best_model_wts)


    def test(self, test_params, test_loader):
        """
        Predict Regression result for test dataset based on the trained model

        Args:
            test_params (dict): parameters for test  # TBD
            test_loader (DataLoader): data loader

        Returns:
            preds (ndarray): prediction data
            trues (ndarray): original data
            mse (float): mean square error  # TBD
            mae (float): mean absolute error    # TBD
        """
        device = test_params['device']
        batch_size = test_params['batch_size']
        input_size = self.model_params['input_size']

        self.model.eval()   # 모델을 validation mode로 설정
        
        # test_loader에 대하여 검증 진행 (gradient update 방지)
        with torch.no_grad():
            corrects = 0
            total = 0
            preds = []
            probs = []
            trues = []
            for inputs, labels in test_loader:
                inputs = inputs.view([batch_size, -1, input_size])
                inputs = inputs.transpose(1, 2).to(device)  # Conv-1d condition
                labels = labels.to(device, dtype=torch.long)

                self.model.to(device)
    
                # forwinputs = inputs.to(device)ard
                # input을 model에 넣어 output을 도출
                outputs = self.model(inputs)
                prob = outputs
                prob = nn.Softmax(dim=1)(prob)

                # output 중 최댓값의 위치에 해당하는 class로 예측을 수행
                _, pred = torch.max(outputs, 1)
                
                # batch별 정답 개수를 축적함
                corrects += torch.sum(pred == labels.data)
                total += labels.size(0)

                preds.extend(pred.detach().cpu().numpy()) 
                probs.extend(prob.detach().cpu().numpy())
                trues.extend(labels.detach().cpu().numpy())

            preds = np.array(preds)
            probs = np.array(probs)
            trues = np.array(trues)
            
            acc = (corrects.double() / total).item()

        print(f'** Performance of test dataset ==> PROB = {probs}, ACC = {acc}')
        print(f'** Dimension of result for test dataset = {preds.shape}')

        return preds, probs, trues, acc


    def inference(self, infer_params, inference_loader):
        """
        Predict regression result for inference dataset based on the trained model

        Args:
            infer_params (dict): parameters for inference     # TBD
            inference_loader (DataLoader): inference data loader

        Returns:
            preds (ndarray) : Inference result data
        """
        device = infer_params['device']
        batch_size = infer_params['batch_size']
        input_size = self.model_params['input_size']

        self.model.eval()   # 모델을 validation mode로 설정
        
        # test_loader에 대하여 검증 진행 (gradient update 방지)
        with torch.no_grad():
            preds = []
            for inputs in inference_loader:
                inputs = inputs.view([batch_size, -1, input_size])
                inputs = inputs.transpose(1, 2).to(device)    # Conv-1d condition

                self.model.to(device)
                
                # forward
                # input을 model에 넣어 output을 도출
                outputs = self.model(inputs)
                _, pred = torch.max(outputs, 1)
                
                # 예측 값 및 실제 값 축적
                preds.extend(pred.detach().cpu().numpy())

        preds = np.array(preds)

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


    # move to utils?
    # for train data
    def create_trainloader(self, batch_size, train_x, train_y, val_x, val_y):
        """
        Create train/valid data loader for torch

        Args:
            batch_size (integer): batch size
            train_x (np.array): train X data
            train_y (np.array): train y data
            val_x (np.array): validation X data
            val_y (np.array): validation y data

        Returns:
            train_loader (DataLoader): train data loader
            val_loader (DataLoader): validation data loader
        """
        # dim = 3
        # # if self.model_name == "FC_cf":
        # #    dim = 2
        # if type(train_x) !=  np.ndarray:
        #     train_x, train_y = transDFtoNP(train_x, train_y, window_num, dim)
        #     val_x, val_y = transDFtoNP(val_x, val_y, window_num, dim)

        # self.params['input_size'] = train_x.shape[1]
        # if dim != 2:
        #     self.params['seq_len']  = train_x.shape[2] # seq_length

        # input_size = train_x.shape[1]
        # seq_len = train_x.shape[2]

        datasets = []
        for dataset in [(train_x, train_y), (val_x, val_y)]:
            x_data = np.array(dataset[0])
            y_data = dataset[1]
            datasets.append(TensorDataset(torch.Tensor(x_data), torch.Tensor(y_data)))

        train_set, val_set = datasets[0], datasets[1]

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=True)

        return train_loader, val_loader


    # for test data
    def create_testloader(self, batch_size, test_x, test_y):
        """
        Create test data loader for torch

        Args:
            batch_size (integer): batch size
            test_x (np.array): test X data
            test_y (np.array): test y data
        
        Returns:
            test_loader (DataLoader) : test data loader
        """
        # test_x, test_y = trans_df_to_np(test_x, test_y, window_num)

        # x_data = np.array(test_x)
        # y_data = test_y

        test_data = TensorDataset(torch.Tensor(test_x), torch.Tensor(test_y))
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)

        return test_loader

    # for inference data
    def create_inferenceloader(self, batch_size, infer_x):
        """
        Create inference data loader for torch

        Args:
            batch_size (integer): batch size
            x_data (np.array): inference X data
        
        Returns:
            inference_loader (DataLoader) : inference data loader
        """
        # match dimension
        if len(infer_x.shape) != 3:
            infer_x = np.expand_dims(infer_x, axis=0)
        
        inference_data = torch.Tensor(infer_x)
        inference_loader = DataLoader(inference_data, batch_size=batch_size, shuffle=True)

        return inference_loader
