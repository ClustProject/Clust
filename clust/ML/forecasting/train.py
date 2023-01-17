import sys
import torch
import torch.nn as nn
import torch.optim as optim
sys.path.append("..")
sys.path.append("../..")

from torch.utils.data import TensorDataset, DataLoader
from Clust.clust.ML.common.trainer import Trainer
from Clust.clust.ML.common import model_manager
from Clust.clust.ML.forecasting.models import rnn_model
from Clust.clust.ML.forecasting.models import lstm_model
from Clust.clust.ML.forecasting.models import gru_model
from Clust.clust.ML.forecasting.optimizer import Optimization
from Clust.clust.transformation.purpose.machineLearning import LSTMData

# Model 2: RNN 계열
class RNNStyleModelTrainer(Trainer):

    def __init__(self):
        """
        """
        super().__init__()

    def set_param(self, param):
        """
        Set Parameter

        Args:
        param(dict): parameter for clustering
            >>> param = {"transformParameter":{},
                         "cleanParam":"clean",
                         "batch_size":16,
                         "model_parameter":{},
                         "n_epochs":10,}
        """
        # Data 처리
        self.clean_param = param.get('clean_param')
        self.transform_parameter = param.get('transform_parameter')
        # Train torch 처리
        self.batch_size = param.get('batch_size')
        self.model_parameter = param.get('model_parameter')
        self.n_epochs = param.get('n_epochs')


    def set_data(self, train, val):
        """
        Set Data
        """
        LSTMD = LSTMData()
        self.trainX_arr, self.trainy_arr = LSTMD.transformXyArr(train, self.transform_parameter, self.clean_param)
        self.valX_arr, self.valy_arr = LSTMD.transformXyArr(val, self.transform_parameter, self.clean_param)


    def get_model(self, model_name):
        # super().get_model(model_method)
        models = {
            "rnn": rnn_model.RNNModel,
            "lstm": lstm_model.LSTMModel,
            "gru": gru_model.GRUModel,
        }
        self.init_model = models.get(model_name.lower())(**self.model_parameter)
        return self.init_model


    def train(self):
        """
        Training
        
        """
        train_DataSet, train_loader = self._get_torch_loader(self.trainX_arr, self.trainy_arr)
        val_DataSet, val_loader = self._get_torch_loader(self.valX_arr, self.valy_arr)

        weight_decay = 1e-6
        learning_rate = 1e-3
        loss_fn = nn.MSELoss(reduction="mean")

        #from torch import optim
        # Optimization
        self.optimizer = optim.Adam(self.init_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        opt = Optimization(model=self.init_model, loss_fn=loss_fn, optimizer=self.optimizer)

        model = opt.train(train_loader, val_loader, batch_size=self.batch_size, n_epochs=self.n_epochs, n_features=self.model_parameter['input_dim'])
        opt.plot_losses()
        # self.opt = opt
        # 모델의 state_dict 출력
        #self.printState_dict()
        model_manager.save_pickle_model(model)


    def _get_torch_loader(self, X_arr, y_arr):
        features = torch.Tensor(X_arr)
        targets = torch.Tensor(y_arr)
        dataSet = TensorDataset(features, targets)
        loader = DataLoader(dataSet, batch_size=self.batch_size, shuffle=False, drop_last=True)
        print("features shape:", features.shape, "targets shape: ", targets.shape)
        return dataSet, loader





    # def _transfrom_data(self, train, val):
    #     LSTMD = LSTMData()
    #     self.trainX_arr, self.trainy_arr = LSTMD.transformXyArr(train, self.transform_parameter, self.clean_param)
    #     self.valX_arr, self.valy_arr = LSTMD.transformXyArr(val, self.transform_parameter, self.clean_param)













    ##-----------------------------------------------------------------------------------------------------------------
    # def printState_dict(self):
    #     print("Model's state_dict:")
    #     for param_tensor in self.model.state_dict():
    #         print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())

    #     # 옵티마이저의 state_dict 출력
    #     print("Optimizer's state_dict:")
    #     for var_name in self.optimizer.state_dict():
    #         print(var_name, "\t", self.optimizer.state_dict()[var_name])


