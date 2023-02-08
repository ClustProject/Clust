import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")
sys.path.append("../..")

from torch.utils.data import TensorDataset, DataLoader
from Clust.clust.ML.common.train import Train
from Clust.clust.ML.common.inference import Inference
from Clust.clust.transformation.purpose.machineLearning import LSTMData

device = "cuda" if torch.cuda.is_available() else "cpu"

class GruTrain(Train):
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


    def set_model(self):
        """
        Build model and return initialized model for selected model_name

        Args:
            model_method (string): model method name  
        """

        self.init_model = LSTMModel(**self.train_parameter)


    def train(self):
        """
        Train model and return model

        Returns:
            model: train model
        """
        self._set_model()
        train_loader = self._get_torch_loader(self.trainX_arr, self.trainy_arr)
        val_loader = self._get_torch_loader(self.valX_arr, self.valy_arr)

        weight_decay = 1e-6
        learning_rate = 1e-3
        loss_fn = nn.MSELoss(reduction="mean")

        #from torch import optim
        # Optimization
        optimizer = optim.Adam(self.init_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        model = self._train_model(train_loader, val_loader, optimizer, self.init_moel, loss_fn,  batch_size=self.batch_size, n_epochs=self.n_epochs, n_features=self.train_parameter['input_dim'])
        self._plot_losses()

        return model


    def _set_model(self):
        """
        Build model and return initialized model for selected model_name

        Args:
            model_method (string): model method name  
        """

        self.init_model = LSTMModel(**self.train_parameter)


    def _get_torch_loader(self, X_arr, y_arr):
        """
        
        """
        features = torch.Tensor(X_arr)
        targets = torch.Tensor(y_arr)
        dataSet = TensorDataset(features, targets)
        training_loader = DataLoader(dataSet, batch_size=self.batch_size, shuffle=False, drop_last=True)
        print("features shape:", features.shape, "targets shape: ", targets.shape)

        return training_loader


    def _train_model(self, train_loader, val_loader, optimizer, model, loss_fn, batch_size=64, n_epochs=50, n_features=1):
        """
        
        """
        self.train_losses = []
        self.val_losses = []

        for epoch in range(1, n_epochs + 1):
            batch_losses = []

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)

                model.train()
                yhat = model(x_batch)
                loss = loss_fn(y_batch, yhat)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                loss = loss.item()
                batch_losses.append(loss)

            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []

                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    model.eval()
                    yhat = model(x_val)
                    val_loss = loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)

                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch <= 10) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )
        return model


    def _plot_losses(self):
        """The method plots the calculated loss values for training and validation
        """
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()



class GruTest(Inference):
    def __init__(self):
        """
        
        """
        super().__init__()

    def set_param(self, param):
        """
        Set Parameter for Inference

        use all model meta

        Args:
            param(dict): model meta


        Example:

            >>> param = { 'trainDataInfo': {...}, 
            ...            'cleanTrainDataParam': {...}, 
            ...            'transformParameter': {...},
            ...            ...
            ...            ...                 }

        """
        self.param = param
        self.clean_param = param['cleanTrainDataParam']
        self.transform_parameter = param['transformParameter']
        self.input_dim = len(self.transform_parameter['feature_col'])
        self.batch_size = 1


    def set_test_data(self, data):
        """
        set data for test & transform data

        Args:
            data (dataframe): Inference data
    

        Example:

        >>> set_data(data)
        ...         data : inference data

        """
        LSTMD = LSTMData()
        self.testX_arr, self.testy_arr = LSTMD.transform_Xy_arr(data, self.transform_parameter, self.clean_param)


    def set_inference_data(self, data):
        """
        set data for inference & transform data

        Args:
            data (dataframe): Inference data
    

        Example:

        >>> set_data(data)
        ...         data : inference data

        """
        data = data.values.astype(np.float32)
        self.data = data.reshape((-1, data.shape[0], data.shape[1]))


    def get_test_result(self, model):
        """
        Predict RegresiionResult based on model result

        Args:
            model (model) : load trained model

        Returns:
            preds (ndarray) : prediction data
            trues (ndarray) : original data
        
        """
        print("\nStart testing data\n")

        test_loader = self._get_test_loader()
        preds, trues = self._test(model, test_loader)

        return preds, trues


    def get_inference_result(self, model):
        """
        Predict RegresiionResult based on model result

        Args:
            model (model) : load trained model

        Returns:
            preds (ndarray) : Inference result data
        
        """
        print("\nStart testing data\n")

        inference_loader = self._get_inference_loader()
        preds = self._inference(model, inference_loader)

        return preds


    def _get_test_loader(self):
        """

        Returns:
            test_loader (DataLoader) : data loader
        """
        features = torch.Tensor(self.testX_arr)
        targets = torch.Tensor(self.testy_arr)

        test_dataSet = TensorDataset(features, targets)
        test_loader = DataLoader(test_dataSet, batch_size=self.batch_size, shuffle=False, drop_last=True)
        print("features shape:", features.shape, "targets shape: ", targets.shape)

        return test_loader
        

    def _get_inference_loader(self):
        """

        Returns:
            inference_loader (DataLoader) : data loader
        """
        data = torch.Tensor(self.data)
        inference_loader = DataLoader(data, batch_size=self.batch_size, shuffle=False, drop_last=True)

        return inference_loader


    def _test(self, model, test_loader):
        """

        Predict RegresiionResult for test dataset based on the trained model

        Args:
            model (model): load trained model
            test_loader (DataLoader) : data loader

        Returns:
            preds (ndarray) : prediction data
            trues (ndarray) : original data
        """

        model.eval()

        with torch.no_grad():
            preds, trues = [], []

            for x_test, y_test in test_loader:

                x_test = x_test.view([self.batch_size, -1, self.input_dim]).to(device)
                y_test = y_test.to(device)

                model.to(device)

                outputs = model(x_test)

                preds.extend(outputs.detach().numpy())
                trues.extend(y_test.detach().numpy())

        preds = np.array(preds).reshape(-1)
        trues = np.array(trues).reshape(-1)
        
        return preds, trues


    def _inference(self, model, inference_loader):
        """


        Args:
            model (model): load trained model
            test_loader (DataLoader) : data loader

        Returns:
            preds (ndarray) : Inference result data
        """

        model.eval()

        with torch.no_grad():
            preds = []

            for input in inference_loader:

                inputs = input.to(device)

                model.to(device)

                outputs = model(inputs)

                preds.extend(outputs.detach().numpy())

        preds = np.array(preds).reshape(-1)

        return preds




class LSTMModel(nn.Module):
    """LSTMModel class extends nn.Module class and works as a constructor for LSTMs.

       LSTMModel class initiates a LSTM module based on PyTorch's nn.Module class.
       It has only two methods, namely init() and forward(). While the init()
       method initiates the model with the given input parameters, the forward()
       method defines how the forward propagation needs to be calculated.
       Since PyTorch automatically defines back propagation, there is no need
       to define back propagation method.

       Attributes:
           hidden_dim (int): The number of nodes in each layer
           layer_dim (str): The number of layers in the network
           lstm (nn.LSTM): The LSTM model constructed with the input parameters.
           fc (nn.Linear): The fully connected layer to convert the final state
                           of LSTMs to our desired output shape.

    """

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        """The __init__ method that initiates a LSTM instance.

        Args:
            input_dim (int): The number of nodes in the input layer
            hidden_dim (int): The number of nodes in each layer
            layer_dim (int): The number of layers in the network
            output_dim (int): The number of nodes in the output layer
            dropout_prob (float): The probability of nodes being dropped out

        """
        super(LSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        ).to(device) # device check

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim).to(device)

    def forward(self, x):
        """The forward method takes input tensor x and does forward propagation

        Args:
            x (torch.Tensor): The input tensor of the shape (batch size, sequence length, input_dim)

        Returns:
            torch.Tensor: The output tensor of the shape (batch size, output_dim)

        """
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(
            0), self.hidden_dim).requires_grad_()

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim, x.size(
            0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out