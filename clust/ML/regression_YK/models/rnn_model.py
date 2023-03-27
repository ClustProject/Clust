import torch.nn as nn
import torch

# device = "cuda" if torch.cuda.is_available() else "cpu"

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_dim, dropout_prob, bidirectional, rnn_type, device):
        """The __init__ method that initiates an RNN instance.

        Args:
            input_size (int): The number of nodes in the input layer
            hidden_size (int): The number of nodes in each layer
            num_layers (int): The number of layers in the network
            output_dim (int): The number of nodes in the output layer
            dropout_prob (float): The probability of nodes being dropped out

        """
        super(RNNModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # defining the type of RNN families (e.g., rnn, lstm, gru,...)
        self.rnn_type = rnn_type
        self.num_directions = 2 if bidirectional == True else 1

        # RNN layers
        if self.rnn_type == 'rnn':
            self.rnn = nn.RNN(
                input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob, bidirectional=bidirectional
            ).to(device)
        # LSTM layers
        elif self.rnn_type == 'lstm':
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob, bidirectional=bidirectional
            ).to(device)
        # GRU layers
        elif self.rnn_type == 'gru':
            self.gru = nn.GRU(
                input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob, bidirectional=bidirectional
            ).to(device)

        # Fully connected layer according to wheter bidirectional
        self.fc = nn.Linear(self.num_directions * hidden_size, output_dim).to(device)

    def forward(self, x):
        """The forward method takes input tensor x and does forward propagation

        Args:
            x (torch.Tensor): The input tensor of the shape (batch size, sequence length, input_size)

        Returns:
            torch.Tensor: The output tensor of the shape (batch size, output_dim)

        """
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.num_directions * self.num_layers, x.size(0), self.hidden_size).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        if self.rnn_type == 'lstm':
            # Initializing cell state for first input with zeros
            c0 = torch.zeros(self.num_directions * self.num_layers, x.size(0), self.hidden_size).requires_grad_()
            # We need to detach as we are doing truncated backpropagation through time (BPTT)
            # If we don't, we'll backprop all the way to the start even after going through another batch
            # Forward propagation by passing in the input, hidden state, and cell state into the model
            out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        elif self.rnn_type == 'gru':
            out, h0 = self.gru(x, h0.detach())
        else:
            out, h0 = self.rnn(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        return out