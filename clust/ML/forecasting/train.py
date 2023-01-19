import sys
import torch
import torch.nn as nn
import torch.optim as optim
sys.path.append("..")
sys.path.append("../..")

from torch.utils.data import TensorDataset, DataLoader
from Clust.clust.ML.common.train import Train
from Clust.clust.ML.forecasting.models.rnn_model import RNNModel
from Clust.clust.ML.forecasting.models.lstm_model import LSTMModel
from Clust.clust.ML.forecasting.models.gru_model import GRUModel
from Clust.clust.ML.forecasting.optimizer import Optimization
from Clust.clust.transformation.purpose.machineLearning import LSTMData

# Model 2: RNN 계열
class ForecastingTrain(Train):

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
        self.parameter = param
        self.n_epochs = param['n_epochs']
        self.batch_size = param['batch_size']
        self.clean_param = param['clean_param']
        self.model_parameter = param['model_parameter']
        self.transform_parameter = param['transform_parameter']


    def set_data(self, train, val):
        """
        Set Data
        """
        LSTMD = LSTMData()
        self.trainX_arr, self.trainy_arr = LSTMD.transformXyArr(train, self.transform_parameter, self.clean_param)
        self.valX_arr, self.valy_arr = LSTMD.transformXyArr(val, self.transform_parameter, self.clean_param)


    def set_model(self, model_method):
        # super().get_model(model_method)
        if (model_method == 'rnn'):
            self.init_model = RNNModel(**self.model_parameter)
        elif model_method == 'lstm':
            self.init_model = LSTMModel(**self.model_parameter)
        elif model_method == 'gru':
            self.init_model = GRUModel(**self.model_parameter)


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
        optimizer = optim.Adam(self.init_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        opt = Optimization(model=self.init_model, loss_fn=loss_fn, optimizer=optimizer)

        model = opt.train(train_loader, val_loader, batch_size=self.batch_size, n_epochs=self.n_epochs, n_features=self.model_parameter['input_dim'])
        opt.plot_losses()

        return model





    def _get_torch_loader(self, X_arr, y_arr):
        features = torch.Tensor(X_arr)
        targets = torch.Tensor(y_arr)
        dataSet = TensorDataset(features, targets)
        loader = DataLoader(dataSet, batch_size=self.batch_size, shuffle=False, drop_last=True)
        print("features shape:", features.shape, "targets shape: ", targets.shape)

        return dataSet, loader