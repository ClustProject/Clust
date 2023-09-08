import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import shutil
from pathlib import Path

class RNNPredictor(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, enc_inp_size, rnn_inp_size, rnn_hid_size, dec_out_size, nlayers, dropout=0.5,
                 tie_weights=False,res_connection=False):
        super(RNNPredictor, self).__init__()
        self.enc_input_size = enc_inp_size

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Linear(enc_inp_size, rnn_inp_size)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(rnn_inp_size, rnn_hid_size, nlayers, dropout=dropout)
        # elif rnn_type == 'SRU':
        #     from cuda_functional import SRU, SRUCell
        #     self.rnn = SRU(input_size=rnn_inp_size,hidden_size=rnn_hid_size,num_layers=nlayers,dropout=dropout,
        #                    use_tanh=False,use_selu=True,layer_norm=True)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'SRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(rnn_inp_size, rnn_hid_size, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(rnn_hid_size, dec_out_size)

        if tie_weights:
            if rnn_hid_size != rnn_inp_size:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
        self.res_connection=res_connection
        self.init_weights()
        self.rnn_type = rnn_type
        self.rnn_hid_size = rnn_hid_size
        self.nlayers = nlayers
        #self.layerNorm1=nn.LayerNorm(normalized_shape=rnn_inp_size)
        #self.layerNorm2=nn.LayerNorm(normalized_shape=rnn_hid_size)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_hiddens=False, noise=False):
        emb = self.drop(self.encoder(input.contiguous().view(-1,self.enc_input_size))) # [(seq_len x batch_size) * feature_size]
        emb = emb.view(-1, input.size(1), self.rnn_hid_size) # [ seq_len * batch_size * feature_size]
        if noise:
            # emb_noise = Variable(torch.randn(emb.size()))
            # hidden_noise = Variable(torch.randn(hidden[0].size()))
            # if next(self.parameters()).is_cuda:
            #     emb_noise=emb_noise.cuda()
            #     hidden_noise=hidden_noise.cuda()
            # emb = emb+emb_noise
            hidden = (F.dropout(hidden[0],training=True,p=0.9),F.dropout(hidden[1],training=True,p=0.9))

        #emb = self.layerNorm1(emb)
        output, hidden = self.rnn(emb, hidden)
        #output = self.layerNorm2(output)

        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2))) # [(seq_len x batch_size) * feature_size]
        decoded = decoded.view(output.size(0), output.size(1), decoded.size(1)) # [ seq_len * batch_size * feature_size]
        if self.res_connection:
            decoded = decoded + input
        if return_hiddens:
            return decoded,hidden,output

        return decoded, hidden
