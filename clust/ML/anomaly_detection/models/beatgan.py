import os,pickle
import numpy as np
import torch
import torch.nn as nn

class Generator(nn.Module):
    '''
    Args:
            in_c : Number of input channel (Number of columns of input data)
            hidden_c : Number of channel of hidden layer 
            latent_c : Number of channel of latent feature (between Encoder - Decoder)
    Example:
        G = Generator(
            in_c     = 51,
            hidden_c = 32,
            latent_c = 16
        )
    '''
    def __init__(self,in_c: int, hidden_c: int, latent_c: int):
        super(Generator,self).__init__()
        self.encoder = Encoder(in_c,hidden_c,latent_c)
        self.decoder = Decoder(in_c,hidden_c,latent_c)
    
    def forward(self,x):
        latent_z = self.encoder(x)
        recon_x = self.decoder(latent_z)
        return recon_x, latent_z
    
class Discriminator(nn.Module):
    '''
    Args:
            in_c : Number of input channel (Number of columns of input data)
            hidden_c : Number of channel of hidden layer 
    Example:
        D = Discriminator(
            in_c     = 51,
            hidden_c = 32
        )
    '''
    def __init__(self,in_c: int, hidden_c: int):
        super(Discriminator,self).__init__()
        
        self.layers = list(Encoder(in_c,hidden_c,1).layers.children())
        self.encoder = nn.Sequential(*self.layers[:-1])
        self.classifier = nn.Sequential(self.layers[-1])
        self.classifier.add_module('Sigomid', nn.Sigmoid())
        
        
    def forward(self,x):
        # Feature extraction 
        features = self.encoder(x)
        classifier = self.classifier(features)
        classifier = classifier.view(-1,1).squeeze(1)
        return classifier, features 
        

def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        # mod.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_normal_(mod.weight.data)
        # nn.init.kaiming_uniform_(mod.weight.data)

    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)
    elif classname.find('Linear') !=-1 :
        torch.nn.init.xavier_uniform(mod.weight)
        mod.bias.data.fill_(0.01)

class Encoder(nn.Module):
    def __init__(self,in_c: int, hidden_c :int, latent_c: int):
        super(Encoder, self).__init__()
        '''
        Args:
            in_c : Number of input channel (Number of columns of input data)
            hidden_c : Number of channel of hidden layer 
            latent_c : Number of channel of latent feature (between Encoder - Decoder)
        Example:
            E = Encoder(
                in_c     = 51,
                hidden_c = 32,
                latent_c = 16
            )
        '''
        self.in_c = in_c # num of features of input data 
        self.hc = hidden_c
        self.layers = self.build_layers(latent_c)
    
    def build_layers(self,latent_c):
        layers = [] 
        for i,c in enumerate([0,1,2,4,8]):
            if i == 0:
                layers.append(conv_block(self.in_c,self.hc,bn=False))        
            else:
                layers.append(conv_block(self.hc*c,self.hc*c*2))
        layers.append(nn.Conv1d(self.hc * 16, latent_c, 10, 1, 0, bias=False))
        return nn.Sequential(*layers)
    
    def forward(self,x):
        return self.layers(x)
    
class Decoder(nn.Module):
    def __init__(self, in_c: int, hidden_c: int, latent_c: int):
        super(Decoder,self).__init__()
        '''
        Args:
            in_c : Number of input channel (Number of columns of input data)
            hidden_c : Number of channel of hidden layer 
            latent_c : Number of channel of latent feature (between Encoder - Decoder)
        Example:
            D = Decoder(
                in_c     = 51,
                hidden_c = 32,
                latent_c = 16
            )
        '''
        self.layers = nn.Sequential(
            conv_tp_block(latent_c,hidden_c*16,10,1,0),
            conv_tp_block(hidden_c*16,hidden_c*8),
            conv_tp_block(hidden_c*8,hidden_c*4),
            conv_tp_block(hidden_c*4,hidden_c*2),
            conv_tp_block(hidden_c*2,hidden_c*1),
            
            nn.ConvTranspose1d(hidden_c ,in_c, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.layers(x)

def conv_block(in_c: int, out_c: int, kernel: int=4, stride: int=2, padding=1, b=False, bn=False):
    ''' Convolution Block for Encoder, Decoder 
    Args:
        in_c  : Number of input channel 
        out_c : Number of output channel 
        b     : Bias 
        bn    : Batch Normalization 
    '''
    cblock = [] 
    cblock.append(nn.Conv1d(in_c, out_c, kernel, stride, padding, bias=b))
    if bn:
        cblock.append(nn.BatchNorm1d(out_c))
    cblock.append(nn.LeakyReLU(0.2,inplace=True))
    return nn.Sequential(*cblock)

def conv_tp_block(in_c: int, out_c: int, kernel: int=4, stride: int=2, padding=1, b=False, bn=False):
    '''Convolution Transpose Block for Decoder 
    Args:
        in_c  : Number of input channel 
        out_c : Number of output channel 
        b     : Bias 
        bn    : Batch Normalization 
    '''
    return nn.Sequential(
        nn.ConvTranspose1d(in_c, out_c, kernel, stride, padding, bias=b),
        nn.BatchNorm1d(out_c),
        nn.ReLU(True)
        )


