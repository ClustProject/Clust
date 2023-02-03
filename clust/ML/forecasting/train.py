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
        Set Parameter for transform, train

        Args:
        param(dict): parameter for train


        Example:

            >>> param = { "cleanParam":"clean",
            ...           "batch_size":16,
            ...           "n_epochs":10,
            ...           "transform_parameter":{ "future_step": 1, "past_step": 24, "feature_col": ["COppm"], "target_col": "COppm"},
            ...           "train_parameter": {'input_dim': 3, 'hidden_dim' : 256, 'layer_dim' : 3,
            ...                                'output_dim' : 1, 'dropout_prob' : 0.2}   }

        """

        self.parameter = param
        self.n_epochs = param['n_epochs']
        self.batch_size = param['batch_size']
        self.clean_param = param['cleanParam']
        self.train_parameter = param['trainParameter']
        self.transform_parameter = param['transformParameter']


    def set_data(self, train, val):
        """
        set train, val data & transform data for training

        Args:
            train (dataframe): train data
            val (dataframe): validation data

        """
        LSTMD = LSTMData()
        self.trainX_arr, self.trainy_arr = LSTMD.transform_Xy_arr(train, self.transform_parameter, self.clean_param)
        self.valX_arr, self.valy_arr = LSTMD.transform_Xy_arr(val, self.transform_parameter, self.clean_param)


    def set_model(self, model_method):
        """
        Build model and return initialized model for selected model_name

        Args:
            model_method (string): model method name  
        """

        if (model_method == 'rnn'):
            self.init_model = RNNModel(**self.train_parameter)
        elif model_method == 'lstm':
            self.init_model = LSTMModel(**self.train_parameter)
        elif model_method == 'gru':
            self.init_model = GRUModel(**self.train_parameter)


    def train(self):
        """
        Train model and return model

        Returns:
            model: train model
        """
        train_loader = self._get_torch_loader(self.trainX_arr, self.trainy_arr)
        val_loader = self._get_torch_loader(self.valX_arr, self.valy_arr)

        weight_decay = 1e-6
        learning_rate = 1e-3
        loss_fn = nn.MSELoss(reduction="mean")

        #from torch import optim
        # Optimization
        optimizer = optim.Adam(self.init_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        opt = Optimization(model=self.init_model, loss_fn=loss_fn, optimizer=optimizer)

        model = opt.train(train_loader, val_loader, batch_size=self.batch_size, n_epochs=self.n_epochs, n_features=self.train_parameter['input_dim'])
        opt.plot_losses()

        return model




    def _get_torch_loader(self, X_arr, y_arr):
        """
        
        """
        features = torch.Tensor(X_arr)
        targets = torch.Tensor(y_arr)
        dataSet = TensorDataset(features, targets)
        training_loader = DataLoader(dataSet, batch_size=self.batch_size, shuffle=False, drop_last=True)
        print("features shape:", features.shape, "targets shape: ", targets.shape)

        return training_loader