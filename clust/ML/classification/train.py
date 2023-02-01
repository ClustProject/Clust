import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import copy
import datetime
from torch.utils.data import DataLoader
sys.path.append("..")
sys.path.append("../..")

from Clust.clust.transformation.type.DFToNPArray import transDFtoNP
from Clust.clust.ML.classification.models.lstm_fcn import LSTM_FCNs
from Clust.clust.ML.classification.models.cnn_1d import CNN_1D
from Clust.clust.ML.classification.models.rnn import RNN_model
from Clust.clust.ML.classification.models.fc import FC
from Clust.clust.ML.common.train import Train

class ClassificationTrain(Train):
    def __init__(self):
        # seed 고정
        random_seed = 42

        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)
        super().__init__()
        

    def set_param(self, param):
        """
        Set Parameter for train

        Args:
        param(dict): parameter for train
            >>> param = { "device":"cpu",
                         "batch_size":16,
                         "num_classes":,
                         "n_epochs":10,}
        """
        self.parameter = param
        self.n_epochs = param['n_epochs']
        self.num_classes = param['num_classes']
        self.device = param['device']
        self.batch_size = param['batch_size']

    def set_data(self, train_x, train_y, val_x, val_y, window_num=0):
        """
        transform data for train

        Args:
            train_x (dataframe): train X data
            train_y (dataframe): train y data
            val_x (dataframe): validation X data
            val_y (dataframe): validation y data
            window_num (integer) : window size
        """
        
        dim = 3
        # if self.model_name == "FC_cf":
        #    dim = 2
        if type(train_x) !=  np.ndarray:
            train_x, train_y = transDFtoNP(train_x, train_y, window_num, dim)
            val_x, val_y = transDFtoNP(val_x, val_y, window_num, dim)

        self.parameter['input_size'] = train_x.shape[1]
        if dim != 2:
            self.parameter['seq_len']  = train_x.shape[2] # seq_length
        
        ## TODO 아래 코드 군더더기 저럴 필요 없음 어짜피 이 함수는 Train을 넣으면 Train, Valid 나누는 함수로 고정시키 때문에
        # train/validation 데이터셋 구축
        self._set_train_val(train_x, train_y, val_x, val_y)
    

    def set_model(self, model_method):
        """
        

        Args:
            model_method (string): model method name
        """
        model_method = model_method
        if model_method == 'LSTM_cf':
            self.parameter["rnn_type"] = 'lstm'
        elif self.model_method == 'GRU_cf':
            self.parameter["rnn_type"] = 'gru'
        
        # build initialized model
        if (model_method == 'LSTM_cf') | (model_method == "GRU_cf"):
            self.init_model = RNN_model(**self.parameter)
        elif model_method == 'CNN_1D_cf':
            self.init_model = CNN_1D(**self.parameter)
        elif model_method == 'LSTM_FCNs_cf':
            self.init_model = LSTM_FCNs(**self.parameter)
        elif model_method == 'FC_cf':
            self.init_model = FC(**self.parameter)
        else:
            print('Choose the model correctly')


    def train(self):
        """
        Train and return model

        Returns:
            model: train model
        """

        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=True)

        print("Start training model")
        
        # train model
        init_model = self.init_model.to(self.device)
        
        dataloaders_dict = {'train': self.train_loader, 'val': self.valid_loader}
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(init_model.parameters(), lr=self.parameter['lr'])
        
        model = self._train_model(init_model, dataloaders_dict, criterion, self.n_epochs, optimizer, self.device)

        return model
        




    def _set_train_val(self, train_x, train_y, val_x, val_y):
        """
        
        """
        datasets = []
        for dataset in [(train_x, train_y), (val_x, val_y)]:
            x_data = np.array(dataset[0])
            y_data = dataset[1]
            datasets.append(torch.utils.data.TensorDataset(torch.Tensor(x_data), torch.Tensor(y_data)))
        
        self.train_set, self.valid_set = datasets[0], datasets[1]


        
    def _train_model(self, model, dataloaders, criterion, num_epochs, optimizer, device):
        """
        Train the model

        :param model: initialized model
        :type model: model

        :param dataloaders: train & validation dataloaders
        :type dataloaders: dictionary

        :param criterion: loss function for training
        :type criterion: criterion

        :param num_epochs: the number of train epochs
        :type num_epochs: int

        :param optimizer: optimizer used in training
        :type optimizer: optimizer

        :return: trained model
        :rtype: model
        """

        since = time.time()

        val_acc_history = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print()
                print('Epoch {}/{}'.format(epoch + 1, num_epochs))

            # 각 epoch마다 순서대로 training과 validation을 진행
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # 모델을 training mode로 설정
                else:
                    model.eval()   # 모델을 validation mode로 설정

                running_loss = 0.0
                running_corrects = 0
                running_total = 0

                # training과 validation 단계에 맞는 dataloader에 대하여 학습/검증 진행
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device, dtype=torch.long)
                    # seq_lens = seq_lens.to(self.parameter['device'])
                    
                    # parameter gradients를 0으로 설정
                    optimizer.zero_grad()
                    
                    # forward
                    # training 단계에서만 gradient 업데이트 수행
                    with torch.set_grad_enabled(phase == 'train'):
                        # input을 model에 넣어 output을 도출한 후, loss를 계산함
                        outputs = model(inputs)
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
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)


        # 전체 학습 시간 계산
        time_elapsed = time.time() - since
        timeElapsed = str(datetime.timedelta(seconds=time_elapsed))
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # validation loss가 가장 낮았을 때의 best model 가중치를 불러와 best model을 구축함
        model.load_state_dict(best_model_wts)
        return model


