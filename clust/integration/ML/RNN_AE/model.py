import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Args:
        n_features (int): input dimension (number of variables) 
        embedding_dim (int): embedding dimension (number of new variables) 
        
    Returns:
        Tensor: x - embedded vectors
    
    """
    def __init__(self, n_features, embedding_dim=64):
        super(Encoder, self).__init__()
        self.n_features = n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        
        self.rnn1 = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        """

        """
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        return x


class Decoder(nn.Module):
    """
    Args:
        n_features (int): input dimension (number of variables) 
        output_dim (int): output dimension (same as number of variables)
        
    Returns:
        Tensor: x - reconstructed vectors
    
    """
    def __init__(self, n_features, output_dim):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.n_features, self.hidden_dim = n_features, 2 * n_features
        
        self.rnn1 = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.n_features,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        """

        """
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x = self.output_layer(x)
        return x


class RecurrentAutoencoder(nn.Module):
    """
    Args:
        n_features (int): input dimension (number of variables) 
        embedding_dim (int): embedding dimension (number of new variables)
        
    Returns:
        Tensor: x - reconstructed vectors
    
    """
    def __init__(self, n_features, embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(n_features, embedding_dim)
        self.decoder = Decoder(embedding_dim, n_features)
    
    def forward(self, x):
        """

        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x