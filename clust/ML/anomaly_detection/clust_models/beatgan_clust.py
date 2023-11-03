import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import time
import math
import copy
import random
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, accuracy_score, recall_score, precision_score
import pandas as pd 

import torch
from torch.utils.data import DataLoader, TensorDataset 
from torch.optim.lr_scheduler import _LRScheduler

import sys 
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from Clust.clust.ML.tool import model as ml_model
from Clust.clust.ML.anomaly_detection.interface import BaseAnomalyDetModel
from Clust.clust.ML.anomaly_detection.models.beatgan import Generator, Discriminator, weights_init
from Clust.clust.ML.anomaly_detection.dataset.beatgan_dataset import BeatganDataset


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
            in_c     = model_params['input_size'],
            hidden_c = model_params['hidden_c'],
            latent_c = model_params['latent_c'] 
        ).apply(weights_init)
        
        self.D = Discriminator(
            in_c     = model_params['input_size'],
            hidden_c = model_params['hidden_c']
        ).apply(weights_init)
        
        # create criterion 
        self.bce_criterion = nn.BCELoss()
        self.mse_criterion = nn.MSELoss()
        
        # create best model weight 
        self.best_model = {}

        # create empty thresholding value for F1
        self.thres = 0
        

    def train(self, train_params, train_loader, valid_loader):
        """
        train function for the Anomaly Detection task.

        Args:
            train_params (dict): parameters for train
            train_loader (Dataloader): train data loader
            valid_loader (Dataloader): validation data loader
        """
        self.device = train_params['device']
        epochs = train_params['n_epochs']

        self.G.to(self.device)
        self.D.to(self.device)

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
        best_auroc = 0

        since = time.time()
        for epoch in range(1, epochs + 1):
            print(f'\nCurrent Epoch : {epoch}')
            
            # train             
            loss_d, loss_g = self._train_step(train_loader)

            # validation 
            val_loss_d, val_loss_g = self._eval_step(valid_loader)

            # Evaluation f1 metric to check during training 
            test_params = {'device' : self.device}
            preds, trues, _ = self.test(test_params, valid_loader)
            valid_auroc = roc_auc_score(trues.astype(np.uint8), preds)

            # best checkpoint save 
            if best_auroc < valid_auroc:
                self.best_model =  {'G' : copy.deepcopy(self.G),
                                    'D' : copy.deepcopy(self.D),
                                    'epoch':epoch}
                best_auroc = valid_auroc
                print(f'\nbest auroc : {valid_auroc}')

            if (epoch <= 200) | (epoch % 50 == 0):
                print(f"[{epoch}/{epochs}] Train : [loss_d : {loss_d:.4f} loss_g : {loss_g:.4f}] | Valid : [loss_d : {val_loss_d:.4f} loss_g : {val_loss_g:.4f}]")
                print(f"Valid AUROC : {valid_auroc}")

        # 전체 학습 시간 계산
        time_elapsed = time.time() - since
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        # self.save_model(save_path = './saved_models/Beatgan/model/test_1/model.pth')

    def test(self, test_params, test_loader, thr_loader=None, raw_return:bool=False):
        """
        Predict result for test dataset based on the trained model

        Args:
            test_params (dict): parameters for test
            test_loader (DataLoader): data loader
            thr_loader (DataLoader): data loader for thresholding
            best_thres (bool) : True : find thresholding for best f1 score | False : thresholding using percentile
            f1_return (bool) : return f1 score 
            raw_return (bool) : raw X and raw reoncstructed X returned for XAI         
        """
        device = test_params['device']

        # if model_file_path is not None:
        #     self.load_model(model_file_path=model_file_path)
            
        self.G.eval()
        self.G.to(device)                
        
        with torch.no_grad():
            if thr_loader is not None:
                preds, trues = [], []
                
                print('Thresholding start')
                for x, y in tqdm(thr_loader):
                    x = x.to(device)
                    recon_x, _ = self.G(x)
                    score = torch.mean(torch.pow((x-recon_x),2),dim=(1,2)).detach().cpu().numpy()
                
                    preds.extend(score)
                    trues.extend(y.detach().cpu().numpy())
                
                preds = np.array(preds).reshape(-1)
                trues = np.array(trues).reshape(-1)
                self.thres = self._get_best_thres(trues, preds)
                        
            preds, trues = [], []
            recon_x_list, x_list = [], [] 
                
            print('Test inference start')
            for x, y in tqdm(test_loader):
                x = x.to(device)
                recon_x, _ = self.G(x)
                score = torch.mean(torch.pow((x-recon_x),2),dim=(1,2)).detach().cpu().numpy()
            
                preds.extend(score)
                trues.extend(y.detach().cpu().numpy())
            
                if raw_return:
                    recon_x_list.append(recon_x.detach().cpu().numpy())
                    x_list.append(x.detach().cpu().numpy())

        preds = np.array(preds).reshape(-1) # Anomaly Score 
        trues = np.array(trues).reshape(-1)

        if raw_return:
            return recon_x_list, x_list
        else:
            return preds, trues, self.thres
    
    @torch.no_grad()
    def inference(self, infer_params, inference_loader, thr_loader=None):
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
        
        # to get the best threshold
        if thr_loader is not None:
            preds, trues = [], []
            
            print('Thresholding start')
            for x, y in tqdm(thr_loader):
                x = x.to(device)
                recon_x, _ = self.G(x)
                score = torch.mean(torch.pow((x-recon_x),2),dim=(1,2)).detach().cpu().numpy()
            
                preds.extend(score)
                trues.extend(y.detach().cpu().numpy())
            
            preds = np.array(preds).reshape(-1)
            trues = np.array(trues).reshape(-1)
            self.thres = self.get_best_thres(trues, preds)
        
        preds = []

        for x in inference_loader:
            x = x.to(device)
            print(x.shape)
            
            recon_x, _ = self.G(x)
            score = torch.mean(torch.pow((x-recon_x),2),dim=(1,2)).detach().cpu().numpy()

            # 예측 값
            preds.extend(score)

        preds = np.array(preds).reshape(-1)

        print(f'** Dimension of result for inference dataset = {preds.shape}')
        
        result = (preds > self.thres).astype(np.float16)

        return result


    def export_model(self):
        """
        export trained model 

        Returns:
            Dictionary: {'G': self.G, 'D' :self.D} / current model object
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
                    stride        = 1,
                    inference     = True 
                )
        inference_loader = DataLoader(inference_set, batch_size=batch_size, shuffle=False, drop_last=True)

        return inference_loader

    # customized funtions
    def _get_best_thres(self, trues, pred):

        best_threshold = None
        best_f1_score = 0
        fpr,tpr,thresholds = roc_curve(trues, pred)
        for threshold in thresholds:
            y_pred = (pred >= threshold).astype(int)
            f1 = f1_score(trues, y_pred)
            if f1 > best_f1_score:
                best_f1_score = f1
                best_threshold = threshold
        return best_threshold
    
    def _train_step(self, train_loader)-> list:
        """
        Args: 
            train_loader : train_loader
        
        """
        self.schedulerD.step()
        self.schedulerG.step()
        loss_meter = LossMeter()
        for i, (x, _) in enumerate(train_loader):
            x = x.to(self.device)
                        
            # ! update G 
            recon_x , _ = self.G(x)
            
            _, feat_real = self.D(x) # original feature
            _, feat_fake = self.D(recon_x) # reconstruction feature 
            
            # loss 
            loss_g_adv = self.mse_criterion(feat_fake,feat_real) # loss for feature matching 
            loss_g_rec = self.mse_criterion(recon_x,x) # reconstruction 
            
            # backward 
            loss_g = loss_g_adv + loss_g_rec * 1 # w_adv = 1 
            # self.optimizerG.zero_grad()
            loss_g.backward()
            self.optimizerG.step()
            
            # ! update D 
            self.G.train()
            self.D.train()
            self.D.zero_grad()
            
            # Train with real 
            out_d_real, _ = self.D(x)
            loss_d_real = self.bce_criterion(
                                            out_d_real,
                                            torch.full((x.shape[0],), self.real_label, device=self.device).type(torch.float32)
                                            )
            
            # Train with fake 
            with torch.no_grad():
                recon_x , _ = self.G(x)
            out_d_fake , _  = self.D(recon_x)
            loss_d_fake     = self.bce_criterion(
                                                out_d_fake,
                                                torch.full((x.shape[0],),self.fake_label,device=self.device).type(torch.float32)
                                                )
            
            # loss backward 
            loss_d = loss_d_real + loss_d_fake
            # self.optimizerD.zero_grad()
            loss_d.backward()
            self.optimizerD.step()
            
            
            #logging 
            log = {
            'loss_g_adv' : loss_g_adv ,
            'loss_g_rec' : loss_g_rec ,
            'loss_g'     : loss_g     ,
            'loss_d_real' : loss_d_real ,
            'loss_d_fake': loss_d_fake,
            'loss_d'     : loss_d     }
            loss_meter.update(log)
            
        # Epoch logging 
        log = loss_meter.avg()
        
        return log['loss_d'], log['loss_g']
    
    def _eval_step(self, valid_loader) -> list:
        
        loss_g_list = [] 
        loss_d_list = [] 
        
        self.G.eval()
        self.D.eval()
        loss_meter = LossMeter()
        for i, (x, _) in enumerate(valid_loader):
            x = x.to(self.device)    
            
            with torch.no_grad():
                # loss_d
                out_d_real, _ = self.D(x)
                loss_d_real = self.bce_criterion(
                                        out_d_real,
                                        torch.full((x.shape[0],), self.real_label, device=self.device).type(torch.float32)
                                        )
                
                recon_x,_ = self.G(x)
                out_d_fake,_ = self.D(recon_x)
                loss_d_fake = self.bce_criterion(
                                            out_d_fake,
                                            torch.full((x.shape[0],), self.real_label, device=self.device).type(torch.float32)
                                            )
                loss_d = loss_d_real + loss_d_fake
                
                # loss_g 
                recon_x,_ = self.G(x) # reconstruction 
                _,feat_real = self.D(x) # original feature
                _, feat_fake = self.D(recon_x) # reconstruction feature 
            
                loss_g_adv = self.mse_criterion(feat_fake, feat_real) # loss for feature matching 
                loss_g_rec = self.mse_criterion(recon_x, x) 
                
                loss_g = loss_g_adv + loss_g_rec * 1 # w_adv = 1 
                
                # if D loss too low, then re-initialize netD
                if loss_d.item() < 5e-6:
                    self._reinitialize_netd()
                        
                log = {
                    'loss_g_adv' : loss_g_adv ,
                    'loss_g_rec' : loss_g_rec ,
                    'loss_g'     : loss_g     ,
                    'loss_d_real' : loss_d_real ,
                    'loss_d_fake': loss_d_fake,
                    'loss_d'     : loss_d     }
                loss_meter.update(log)
                    
            # Epoch Logging 
            log = loss_meter.avg()
                
        return log['loss_d'], log['loss_g']
    
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
        
    def _reinitialize_netd(self):
        self.D.apply(weights_init)
        print('Reloading d net')


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

# TBD: Move to?
class LossMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.loss_g_adv  = 0.0
        self.loss_g_rec  = 0.0
        self.loss_g      = 0.0 
        
        self.loss_d_real = 0.0 
        self.loss_d_fake = 0.0 
        self.loss_d      = 0.0 
        
        self.count = 0 
        
    def update(self,log,n=1):
        self.count += n 
        
        self.loss_g_adv  += log['loss_g_adv']
        self.loss_g_rec  += log['loss_g_rec']
        self.loss_g      += log['loss_g'] 
        
        self.loss_d_real += log['loss_d_real'] 
        self.loss_d_fake += log['loss_d_fake'] 
        self.loss_d      += log['loss_d'] 
        
    def avg(self):
        log = {
            'loss_g_adv'  : self.loss_g_adv/self.count,
            'loss_g_rec'  : self.loss_g_rec/self.count,
            'loss_g'      : self.loss_g    /self.count,
            'loss_d_real' : self.loss_d_real/self.count,
            'loss_d_fake' : self.loss_d_fake/self.count,
            'loss_d'      : self.loss_d    /self.count
            }
        self.reset()
        return log 