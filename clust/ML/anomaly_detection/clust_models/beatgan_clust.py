import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import math
import copy
import random
from sklearn.metrics import roc_auc_score, f1_score
import pandas as pd 
# 각자 모델 별로 데이터셋 class 구축하여 사용하셔도 됩니다. 

import torch
from torch.utils.data import DataLoader, TensorDataset 
from torch.optim.lr_scheduler import _LRScheduler

import sys 
# sys.path.append('/Volume/IITP/') #! 상대 경로로 변경 필요
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from Clust.clust.ML.tool import model as ml_model
from Clust.clust.ML.anomaly_detection.interface import BaseAnomalyDetModel
from Clust.clust.ML.anomaly_detection.models.beatgan import Generator, Discriminator, weights_init  #사용하는 ad 모델 적용
from Clust.clust.ML.anomaly_detection.dataset.beatgan_dataset import BeatganDataset

# import sys 
# sys.path.append('../')
# from ..inference import BaseAnomalyDetectionModel
# from ..models.beatgan import Generator, Discriminator
# from ..dataset.beatgan_dataset import BeatganDataset


class BeatganClust(BaseAnomalyDetModel):
    def __init__(self, model_params):
        """
        Init function of BeatGan class.

        Args:
            model_params (dict): parameters for building an BeatGan model
        """
        # create model
        self.model_params = model_params
        self.G = Generator(
            in_c     = model_params['in_c'],
            hidden_c = model_params['hidden_c'],
            latent_c = model_params['latent_c'] 
        ).apply(weights_init)
        
        self.D = Discriminator(
            in_c     = model_params['in_c'],
            hidden_c = model_params['hidden_c'],
            latent_c = model_params['latent_c'] 
        ).apply(weights_init)
        
        # create criterion 
        self.bce_criterion = nn.BCELoss()
        self.mse_criterion = nn.MSELoss()
        
        # create best model weight 
        self.best_model = {}
        

    def train(self, train_params, train_loader, valid_loader):
        """
        train function for the Anomaly Detection task.

        Args:
            train_params (dict): parameters for train
            train_loader (Dataloader): train data loader
            valid_loader (Dataloader): validation data loader
        """
        device = train_params['device']
        epochs = train_params['n_epochs']

        self.G.to(device)
        self.D.to(device)
        
        # create optimizer 
        self.optimizerG = torch.optim.Adam(self.G.parameters(),lr=train_params['lr'],betas=(0.999,0.999))
        self.optimizerD = torch.optim.Adam(self.D.parameters(),lr=train_params['lr'],betas=(0.999,0.999))
        
        # create scheduler 
        self.schedulerG, self.schedulerD = self._create_scheduler(
                                                optimizerG = self.optimizerG,
                                                optimizerD = self.optimizerD,
                                                epochs     = train_params['n_epochs'],
                                                lr         = train_params['lr']         
                                                )
        
        # create label 
        self.real_label = 1 
        self.fake_label = 0 
        
        # create list to stack loss 
        self.train_losses = []
        self.val_losses = []
        
        # create empty variable for checkpoint 
        best_loss = np.inf

        since = time.time()
        for epoch in range(1, epochs + 1):
            print(f'\nCurrent Epoch : {epoch}')
            
            # train             
            loss_d, loss_g = self._train_step(
                G             = self.G, 
                D             = self.D,
                train_loader  = train_loader,
                bce_criterion = self.bce_criterion,
                mse_criterion = self.mse_criterion,
                optimizerD    = self.optimizerD,
                optimizerG    = self.optimizerG,
                real_label    = self.real_label,
                fake_label    = self.fake_label,
                device        = device
            )
            
            # validation 
            val_loss_d, val_loss_g = self._eval_step(
                G             = self.G, 
                D             = self.D,
                valid_loader  = valid_loader,
                bce_criterion = self.bce_criterion,
                mse_criterion = self.mse_criterion,
                real_label    = self.real_label,
                device        = device
            )
            
            # best checkpoint save 
            if best_loss > np.mean(val_loss_g):
                self.best_model =  {'G' : self.G.state_dict,
                                    'D' : self.D.state_dict}
                best_loss = np.mean(val_loss_g)
            
            # Evaluation f1 metric to check during training 
            test_params = {'device' : device}
            valid_preds, valid_trues = self.test(test_params, valid_loader)
            auroc = roc_auc_score(valid_trues, valid_preds)
            
            self.schedulerD.step()
            self.schedulerG.step()

            if (epoch <= 10) | (epoch % 50 == 0):
                print(f"[{epoch}/{epochs}] Train : [loss_d : {np.mean(loss_d):.4f} loss_g : {np.mean(loss_g):.4f}] | Valid : [loss_d : {np.mean(val_loss_d):.4f} loss_g : {np.mean(val_loss_g):.4f}]")
                print(f"Valid AUROC : {auroc}")

        # 전체 학습 시간 계산
        time_elapsed = time.time() - since
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
        # best model load 
        self.G.state_dict = self.best_model['G']
        self.D.state_dict = self.best_model['D']
        print('Best model loaded and finish training')

    @torch.no_grad()
    def test(self, test_params, test_loader):
        """
        Predict result for test dataset based on the trained model

        Args:
            test_params (dict): parameters for test
            test_loader (DataLoader): data loader

        Returns:
            preds (ndarray): prediction data
            trues (ndarray): original data
        """
        device = test_params['device']

        self.G.eval()
        self.G.to(device)
        
        preds, trues = [], []
        
        for x, y in test_loader:
            x = x.to(device)
            
            recon_x, _ = self.G(x)
            score = torch.mean(torch.pow((x-recon_x),2),dim=(1,2)).detach().cpu().numpy()
            
            preds.extend(score)
            trues.extend(y.detach().cpu().numpy())

        # anomaly score normalization
        min_value = np.min(preds)
        max_value = np.max(preds)
        preds = (np.array(preds) - min_value) / (max_value-min_value)

        preds = np.array(preds).reshape(-1)
        trues = np.array(trues).reshape(-1)

        return preds, trues
    
    @torch.no_grad()
    def inference(self, infer_params, inference_loader):
        """
        Predict Anomaly Detection result for inference dataset based on the trained model

        Args:
            infer_params (dict): parameters for inference
            inference_loader (DataLoader): inference data loader

        Returns:
            preds (ndarray) : Inference result data
        """
        device = infer_params['device']

        self.G.eval()   # 모델을 validation mode로 설정
        self.G.to(device)
        preds = []

        for x in inference_loader:
            x = x.to(device)
            
            recon_x, _ = self.G(x)
            score = torch.mean(torch.pow((x-recon_x),2),dim=(1,2)).detach().cpu().numpy()

            # 예측 값
            preds.extend(score)

        preds = np.array(preds).reshape(-1)

        print(f'** Dimension of result for inference dataset = {preds.shape}')

        return preds

    def export_model(self):
        """
        export trained model 

        Returns:
            {'G': self.G, 'D' :self.D} (dict): current model object
        """
        return {'G': self.G, 'D' :self.D}

    def save_model(self, save_path):
        """
        save model to save_path

        Args:
            save_path (string): path to save model
        """
        model = {'G' : self.G, 'D' : self.D}
        ml_model.save_pickle_model(model, save_path)

    def load_model(self, model_file_path):
        """
        load model from model_file_path

        Args:
            model_file_path (string): path to load saved model
        """
        model = ml_model.load_pickle_model(model_file_path)
        self.G, self.D = model['G'], model['D']

    # for train data
    def create_trainloader(self, batch_size, train_x, train_y, val_x, val_y):
        """
        Create train/valid data loader for torch

        Args:
            batch_size (integer): batch size
            task (string): task (e.g., regression, forecast)
            train_x (dataframe): train X data
            train_y (dataframe): train y data (regression only)
            val_x (dataframe): validation X data
            val_y (dataframe): validation y data (regression only)

        Returns:
            train_loader (DataLoader): train data loader
            val_loader (DataLoader): validation data loader
        """
        
        train_set = BeatganDataset(
                    data_x        = train_x,
                    data_y        = train_y,
                    window_size   = 320,
                    stride        = 1 
                )
        val_set = BeatganDataset(
                    data_x        = val_x,
                    data_y        = val_y,
                    window_size   = 320,
                    stride        = 1 
                )

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True)

        return train_loader, val_loader

    # for test data
    def create_testloader(self, batch_size, test_x, test_y):
        """
        Create test data loader for torch

        Args:
            batch_size (integer): batch size
            test_x (np.array): test X data
            test_y (np.array): test y data
        
        Returns:
            test_loader (DataLoader) : test data loader
        """
        test_set = BeatganDataset(
                    data_x        = test_x,
                    data_y        = test_y,
                    window_size   = 320,
                    stride        = 1 
                )

        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)
        print("features shape:", test_set.X.shape, "targets shape: ", test_set.Y.shape)

        return test_loader

    # for inference data
    def create_inferenceloader(self, batch_size, infer_x):
        """
        Create inference data loader for torch

        Args:
            batch_size (integer): batch size
            infer_x (np.array): inference X data
        
        Returns:
            inference_loader (DataLoader) : inference data loader
        """
        # ensure input shape is [batch_size, seq_len, input_size]
        inference_set = BeatganDataset(
                    data_x        = infer_x,
                    data_y        = None,
                    window_size   = 320,
                    stride        = 1 
                )
        inference_loader = DataLoader(inference_set, batch_size=batch_size, shuffle=False, drop_last=True)

        return inference_loader

    # customized funtions    
    def _updateD(self, G, D, bce_criterion, optimizerD, x, real_label, fake_label, device):
        G.train()
        D.train()
        D.zero_grad()
        
        # Train with real 
        out_d_real,_ = D(x)
        loss_d_real = bce_criterion(
                                    out_d_real.squeeze(1),
                                    torch.full((x.shape[0],), real_label, device=device).type(torch.float32)
                                    )
        
        # Train with fake 
        # with torch.no_grad():
        #     recon_x,_ = G(x)
        recon_x, _ = G(x)
        out_d_fake,_ = D(recon_x)
        loss_d_fake = bce_criterion(
                                    out_d_fake.squeeze(1),
                                    torch.full((x.shape[0],),fake_label,device=device).type(torch.float32)
                                    )
        
        # loss backward 
        loss_d = loss_d_real + loss_d_fake
        optimizerD.zero_grad()
        loss_d.backward()
        optimizerD.step()
        
        return loss_d.item()
    
    def _updateG(self, G, D, mse_criterion, optimizerG, x):
        G.train()
        D.train()
        G.zero_grad()
    
        #reconsturction 
        recon_x,_ = G(x)
        
        with torch.no_grad():
            _,feat_real = D(x) # original feature
        _, feat_fake = D(recon_x) # reconstruction feature 
        
        # loss 
        loss_g_adv = mse_criterion(feat_fake, feat_real) # loss for feature matching 
        loss_g_rec = mse_criterion(recon_x, x) # reconstruction 
        
        # backward 
        loss_g = loss_g_adv + loss_g_rec * 1 # w_adv = 1 
        optimizerG.zero_grad()
        loss_g.backward()
        optimizerG.step()
        
        return loss_g.item()
    
    def _train_step(self, G, D, train_loader,
            bce_criterion, mse_criterion,
            optimizerD, optimizerG,
            real_label, fake_label, device)-> list:
        """
        Args: 
        
        """
        loss_d_list = [] 
        loss_g_list = [] 
        for i, (x, _) in enumerate(train_loader):
            x = x.to(device)
            
            loss_d = self._updateD(
                G             = G, 
                D             = D, 
                bce_criterion = bce_criterion,
                optimizerD    = optimizerD,
                x             = x, 
                real_label    = real_label,
                fake_label    = fake_label,
                device        = device
                )
            
            loss_g = self._updateG(
                G             = G, 
                D             = D, 
                mse_criterion = mse_criterion,
                optimizerG    = optimizerG,
                x             = x
                )
            
            loss_d_list.append(loss_d)
            loss_g_list.append(loss_g)
            
        return loss_d_list, loss_g_list
    
    @torch.no_grad()
    def _eval_step(self, G, D, valid_loader,
            bce_criterion, mse_criterion,
            real_label, device) -> list:
        
        loss_g_list = [] 
        loss_d_list = [] 
        
        G.eval()
        D.eval()
        for i, (x, _) in enumerate(valid_loader):
            x = x.to(device)    
            
            # loss_d
            out_d_real, _ = D(x)
            loss_d_real = bce_criterion(
                                    out_d_real.squeeze(1),
                                    torch.full((x.shape[0],), real_label, device=device).type(torch.float32)
                                    )
            
            recon_x,_ = G(x)
            out_d_fake,_ = D(recon_x)
            loss_d_fake = bce_criterion(
                                        out_d_fake.squeeze(1),
                                        torch.full((x.shape[0],),real_label,device=device).type(torch.float32)
                                        )
            loss_d = loss_d_real + loss_d_fake
            
            # loss_g 
            recon_x,_ = G(x) # reconstruction 
            _,feat_real = D(x) # original feature
            _, feat_fake = D(recon_x) # reconstruction feature 
        
            loss_g_adv = mse_criterion(feat_fake, feat_real) # loss for feature matching 
            loss_g_rec = mse_criterion(recon_x, x) 
            
            loss_g = loss_g_adv + loss_g_rec * 1 # w_adv = 1 
            
            loss_d_list.append(loss_d.item())
            loss_g_list.append(loss_g.item())
        return loss_d_list, loss_g_list
    
    def _create_scheduler(self, optimizerG, optimizerD, 
                          epochs, lr, 
                          min_lr: float =  0.0001,  warmup_ratio: float = 0.1):
        
        schedulerG = CosineAnnealingWarmupRestarts(
            optimizer         = optimizerG, 
            first_cycle_steps = epochs,
            max_lr            = lr,
            min_lr            = min_lr,
            warmup_steps      = int(epochs * warmup_ratio)
        )
        schedulerD = CosineAnnealingWarmupRestarts(
            optimizer         = optimizerD, 
            first_cycle_steps = epochs,
            max_lr            = lr,
            min_lr            = min_lr,
            warmup_steps      = int(epochs * warmup_ratio)
            )
        return schedulerG, schedulerD
        
class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr