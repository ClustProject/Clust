import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import time
import copy
import random
from torch.utils.data import DataLoader, TensorDataset
import shutil
from pathlib import Path

from Clust.clust.ML.tool import model as ml_model

from Clust.clust.ML.anomaly_detection.interface import BaseAnomalyDetModel
from Clust.clust.ML.anomaly_detection.models.rnn_predictor import RNNPredictor

class RNNAnomalyClust(BaseAnomalyDetModel):
    """ RNN-based predictor model class
    """
    def __init__(self, model_params):
        """
        Init function of RNN predctor

        Args:
            model_params (dict): parameters for building RNN predictor model
        """
        self.params = dict()
        self.model = RNNPredictor(
            rnn_type = model_params['rnn_type'], 
            enc_inp_size = model_params['enc_inp_size'],
            rnn_inp_size = model_params['rnn_inp_size'],
            rnn_hid_size = model_params['rnn_hid_size'],
            dec_out_size = model_params['dec_out_size'],
            nlayers = model_params['nlayers'],
            dropout = model_params['dropout'],
            tie_weights = model_params['tie_weights'],
            res_connection = model_params['res_connection']
        )
        self.params['model_params'] = model_params

    def train(self, train_params, train_dataset, gen_dataset):
        device = train_params['device']

        optimizer = optim.Adam(self.model.parameters(), lr=train_params['lr'], weight_decay=train_params['weight_decay'])
        criterion = nn.MSELoss()

        if train_params['resume'] or train_params['pretrained']:
            print("=> loading checkpoint ")
            checkpoint = torch.load(Path(self.params['model_params']['model_file_path']))
            
            start_epoch = checkpoint['epoch'] +1
            best_val_loss = checkpoint['best_loss']
            optimizer.load_state_dict(checkpoint['optimizer'])

            self.params['train_params'] = checkpoint['params']
            self.params['train_params']['resume'] = train_params['resume']
            self.params['train_params']['pretrained'] = train_params['pretrained']
            self.params['train_params']['epochs'] = train_params['epochs']
            self.params['train_params']['save_interval'] = train_params['save_interval']
            self.params['train_params']['prediction_window_size'] = train_params['prediction_window_size']
            del checkpoint

            epoch = start_epoch
            print("=> loaded checkpoint")

        else:
            epoch = 1
            start_epoch = 1
            best_val_loss = float('inf')
            print("=> Start training from scratch")

        with torch.enable_grad():
            # Turn on training mode which enables dropout.
            self.model.train()
            total_loss = 0
            start_time = time.time()
            hidden = self._init_hidden(train_params['batch_size'])
            for batch, i in enumerate(range(0, train_dataset.size(0) - 1, train_params['bptt'])):
                inputSeq, targetSeq = self._get_batch(train_params, train_dataset, i)
                # inputSeq: [ seq_len * batch_size * feature_size ]
                # targetSeq: [ seq_len * batch_size * feature_size ]

                # Starting each batch, we detach the hidden state from how it was previously produced.
                # If we didn't, the model would try backpropagating all the way to start of the dataset.
                hidden = model.repackage_hidden(hidden)
                hidden_ = model.repackage_hidden(hidden)
                optimizer.zero_grad()

                '''Loss1: Free running loss'''
                outVal = inputSeq[0].unsqueeze(0)
                outVals=[]
                hids1 = []
                for i in range(inputSeq.size(0)):
                    outVal, hidden_, hid = model.forward(outVal, hidden_,return_hiddens=True)
                    outVals.append(outVal)
                    hids1.append(hid)
                outSeq1 = torch.cat(outVals,dim=0)
                hids1 = torch.cat(hids1,dim=0)
                
                # loss1 = criterion(outSeq1.view(args.batch_size,-1), targetSeq.view(args.batch_size,-1))
                loss1 = criterion(outSeq1.contiguous().view(params['batch_size'],-1), targetSeq.contiguous().view(params['batch_size'],-1))

                '''Loss2: Teacher forcing loss'''
                outSeq2, hidden, hids2 = model.forward(inputSeq, hidden, return_hiddens=True)
                # loss2 = criterion(outSeq2.view(args.batch_size, -1), targetSeq.view(args.batch_size, -1))
                loss2 = criterion(outSeq2.contiguous().view(params['batch_size'], -1), targetSeq.contiguous().view(params['batch_size'], -1))

                '''Loss3: Simplified Professor forcing loss'''
                loss3 = criterion(hids1.view(params['batch_size'],-1), hids2.view(params['batch_size'],-1).detach())

                '''Total loss = Loss1+Loss2+Loss3'''
                loss = loss1+loss2+loss3
                loss.backward()

                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm_(model.parameters(), params['clip'])
                optimizer.step()

                total_loss += loss.item()

                if batch % params['log_interval'] == 0 and batch > 0:
                    cur_loss = total_loss / params['log_interval']
                    elapsed = time.time() - start_time
                    print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.4f} | '
                        'loss {:5.2f} '.format(
                        epoch, batch, len(train_dataset) // params['bptt'],
                                    elapsed * 1000 / params['log_interval'], cur_loss))
                    total_loss = 0
                    start_time = time.time()

        self.params['train_params'] = train_params

    def test(self, test_params, test_loader):
        pass

    def anomaly_detect(self):
        pass

    def export_model(self):
        pass

    def save_model(self, state, is_best):
        print("=> saving checkpoint ..")
        
        # params = state['params']
        checkpoint_dir = Path('save',self.params['data'],'checkpoint')
        checkpoint_dir.mkdir(parents=True,exist_ok=True)
        checkpoint = checkpoint_dir.joinpath(self.params['filename']).with_suffix('.pth')

        torch.save(state, checkpoint)
        if is_best:
            model_best_dir = Path('save',params['data'],'model_best')
            model_best_dir.mkdir(parents=True,exist_ok=True)

            shutil.copyfile(checkpoint, model_best_dir.joinpath(params['filename']).with_suffix('.pth'))

        print('=> checkpoint saved.')

    def load_model(self, model_file_path):
        """
        load model from model_file_path

        Args:
            model_file_path (string): path to load saved model
        """
        self.params['model_params']['model_file_path'] = model_file_path
        print("=> loading model weights")
        checkpoint = torch.load(Path(model_file_path))
        
        # self._initialize()
        self.model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded model weights")
        del checkpoint

    def create_trainloader(self, batch_size, timeseries_data):
        
        train_dataset = timeseries_data.batchify(timeseries_data.trainData, batch_size)
        # to generate output figure (target/output)
        gen_dataset = timeseries_data.batchify(timeseries_data.testData, 1)

        return train_dataset, gen_dataset

    
    def create_testloader(self, batch_size, test_dataset):
        pass

    def create_detectloader(self, batch_size, detect_dataset):
        pass

    # customized methods

    def _init_hidden(self, bsz):
        weight = next(self.model.parameters()).data
        if self.model_params['rnn_type'] == 'LSTM':
            return (Variable(weight.new(self.model.nlayers, bsz, self.model.rnn_hid_size).zero_()),
                    Variable(weight.new(self.model.nlayers, bsz, self.model.rnn_hid_size).zero_()))
        else:
            return Variable(weight.new(self.model.nlayers, bsz, self.model.rnn_hid_size).zero_())

    def _repackage_hidden(self,h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == tuple:
            return tuple(self._repackage_hidden(v) for v in h)
        else:
            return h.detach()

    def _extract_hidden(self, hidden):
        if self.rnn_type == 'LSTM':
            return hidden[0][-1].data.cpu()  # hidden state last layer (hidden[1] is cell state)
        else:
            return hidden[-1].data.cpu()  # last layer

    def _initialize(self, params, feature_dim):
        self.model.__init__(rnn_type = params['model'],
                           enc_inp_size=feature_dim,
                           rnn_inp_size = params['emsize'],
                           rnn_hid_size = params['nhid'],
                           dec_out_size=feature_dim,
                           nlayers = params['nlayers'],
                           dropout = params['dropout'],
                           tie_weights = params['tied'],
                           res_connection = params['res_connection'])
        self.model.to(params['device'])

    def _calculate_score(self):
        pass

    def _get_batch(params, source, i):
        seq_len = min(params['bptt'], len(source) - 1 - i)
        data = source[i:i+seq_len] # [ seq_len * batch_size * feature_size ]
        target = source[i+1:i+1+seq_len] # [ (seq_len x batch_size x feature_size) ]
        return data, target
