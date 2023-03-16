import os
import sys
sys.path.append("../")
sys.path.append("../../")

import datetime
import numpy as np
import pandas as pd
import math
from tqdm import tqdm

from Clust.clust.ML.regression_JS.models.loss_transfer import TransferLoss
from Clust.clust.ML.regression_JS.models.AdaRNN import AdaRNN

import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class data_loader(Dataset):
    def __init__(self, data_x, data_y, t=None):
        assert len(data_x) == len(data_y)

        self.T=t
        self.data_x=data_x
        self.data_y = data_y

        self.data_x=torch.tensor(self.data_x, dtype=torch.float32)
        self.data_y=torch.tensor(self.data_y, dtype=torch.float32)
    
    def __getitem__(self, index):
        sample, label_reg =self.data_x[index], self.data_y[index]
        if self.T:
            return self.T(sample)
        else:
            return sample, label_reg

    def __len__(self):
        return len(self.data_x)


class ClustAdaRnn():
    def __init__(self, param):
        self.param = param
        self.batch_size = param["hidden_size"]
        self.dis_type = param["loss_type"]

        print(self.param)

        n_hiddens = [self.batch_size for i in range(self.param["num_layers"])]
        self.model = AdaRNN(
            use_bottleneck=True, 
            bottleneck_width=64, 
            n_input=self.param["d_feat"], 
            n_hiddens=n_hiddens,  
            n_output=self.param["class_num"], 
            dropout=self.param["dropout"], 
            len_seq=self.param["len_seq"], 
            trans_loss=self.dis_type).cuda()
        
    def _pprint(self, *text):
        time = '['+str(datetime.datetime.utcnow() +datetime.timedelta(hours=8))[:19]+'] -'
        print(time, *text, flush=True)

    def train_AdaRNN(self, args, model, optimizer, train_loader_list, epoch, dist_old=None, weight_mat=None):
        model.train()
        criterion = nn.MSELoss()
        criterion_1 = nn.L1Loss()
        loss_all = []
        loss_1_all = []
        dist_mat = torch.zeros(args["num_layers"], args["len_seq"]).cuda()
        len_loader = np.inf
        for loader in train_loader_list:
            if len(loader) < len_loader:
                len_loader = len(loader)
        for data_all in tqdm(zip(*train_loader_list), total=len_loader):
            optimizer.zero_grad()
            list_feat = []
            list_label = []
            for data in data_all:
                feature, label_reg = data[0].cuda().float(), data[1].cuda().float()
                list_feat.append(feature)
                list_label.append(label_reg)
            flag = False
            index = []
            num_domain = len(data_all) - 1
            for i in range(num_domain):
                 for j in range(i+1, num_domain+1):
                      index.append((i,j))

            for temp_index in index:
                s1 = temp_index[0]
                s2 = temp_index[1]
                if list_feat[s1].shape[0] != list_feat[s2].shape[0]:
                    flag = True
                    break
            if flag:
                continue

            total_loss = torch.zeros(1).cuda()
            for i in range(len(index)):
                feature_s = list_feat[index[i][0]]
                feature_t = list_feat[index[i][1]]
                label_reg_s = list_label[index[i][0]]
                label_reg_t = list_label[index[i][1]]
                feature_all = torch.cat((feature_s, feature_t), 0)

                if epoch < args["pre_epoch"]:
                    pred_all, loss_transfer, out_weight_list = model.forward_pre_train(feature_all, len_win=args["len_win"])
                else:
                    pred_all, loss_transfer, dist, weight_mat = model.forward_Boosting(feature_all, weight_mat)
                    dist_mat = dist_mat + dist
                pred_s = pred_all[0:feature_s.size(0)]
                pred_t = pred_all[feature_s.size(0):]

                loss_s = criterion(pred_s, label_reg_s)
                loss_t = criterion(pred_t, label_reg_t)
                loss_l1 = criterion_1(pred_s, label_reg_s)

                total_loss = total_loss + loss_s + loss_t + args["dw"] * loss_transfer
            loss_all.append([total_loss.item(), (loss_s + loss_t).item(), loss_transfer.item()])
            loss_1_all.append(loss_l1.item())
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
            optimizer.step()
        loss = np.array(loss_all).mean(axis=0)
        loss_l1 = np.array(loss_1_all).mean()
        if epoch >= args["pre_epoch"]:
            if epoch > args["pre_epoch"]:
                weight_mat = model.update_weight_Boosting(weight_mat, dist_old, dist_mat)
            return loss, loss_l1, weight_mat, dist_mat
        else:
            weight_mat = torch.ones(args["num_layers"], args["len_seq"]).cuda()
            for i in range(args["num_layers"]):
                for j in range(args["len_seq"]):
                    weight_mat[i, j] = out_weight_list[i][j].item()
            return loss, loss_l1, weight_mat, None

    def _get_evaluation_loss(self, test_loader, prefix='Test'):
        total_loss = 0
        total_loss_1 = 0
        total_loss_r = 0
        
        criterion = nn.MSELoss()
        criterion_1 = nn.L1Loss()
        for loader_x, loader_y in tqdm(test_loader, desc=prefix, total=len(test_loader)):
            loader_x, loader_y = loader_x.cuda().float(), loader_y.cuda().float()
            with torch.no_grad():
                pred = self.model.predict(loader_x)
            loss = criterion(pred, loader_y)
            loss_r = torch.sqrt(loss)
            loss_1 = criterion_1(pred, loader_y)
            total_loss += loss.item()
            total_loss_1 += loss_1.item()
            total_loss_r += loss_r.item()
        loss = total_loss / len(test_loader)
        loss_1 = total_loss_1 / len(test_loader)
        loss_r = loss_r / len(test_loader)
        return loss, loss_1, loss_r

    def train(self, train_loader_list, valid_loader, args):
        if not os.path.exists(args["output_folder_name"]):
            os.makedirs(args["output_folder_name"])

        save_model_path = os.path.join(args["output_folder_name"], args["save_model_name"])

        self._pprint('create model...')

        optimizer = optim.Adam(self.model.parameters(), lr=args["lr"])
    
        best_score = np.inf
        best_epoch, stop_round = 0, 0
        weight_mat, dist_mat = None, None

        for epoch in range(args["n_epochs"]):
            self._pprint('Epoch:', epoch)
            self._pprint('training...')
            loss, lossl1, weight_mat, dist_mat = self.train_AdaRNN(args, self.model, optimizer, train_loader_list, epoch, dist_mat, weight_mat)

            self._pprint('evaluating...')
            self.model.eval()
            val_mse_loss, val_loss_l1, val_loss_r = self._get_evaluation_loss(valid_loader, prefix='Valid')
            self._pprint('train %.6f, valid MSE Loss %.6f, valid L1 Loss %.6f' %(lossl1, val_mse_loss, val_loss_l1))

            if val_mse_loss < best_score:
                best_score = val_mse_loss
                stop_round = 0
                best_epoch = epoch
                self.save_model(save_model_path)
            else:
                stop_round += 1
                if stop_round >= args["early_stop"]:
                    self._pprint('early stop')
                    break

        self._pprint('best val mse loss score:', best_score, '@', best_epoch)
        self._pprint('Finished.')

    def test(self, test_loader):
        self.model.eval()
        test_loss, test_loss_l1, test_loss_r = self._get_evaluation_loss(test_loader, prefix='Test')

        self._pprint('Finished.')
        
        return test_loss, test_loss_l1, test_loss_r
    
    def inference(self, inference_loader):
        self._pprint('inference...')

        self.model.eval()
        i = 0
        for inference_data in inference_loader:
            inference_data = inference_data.cuda().float()
            with torch.no_grad():
                pred = self.model.predict(inference_data)
            predict_list = pred.cpu().numpy()

        self._pprint('Finished.')
        return predict_list

    def create_trainloader(self, train_x, train_y, valid_x, valid_y, k, train_x_start_time, train_x_end_time, shuffle = True):
        split_timelist_by_tdc = self._tdc(k, train_x, train_x_start_time, train_x_end_time, self.dis_type)

        train_loader_list = []
        for i in range(len(split_timelist_by_tdc)):
            tdc_start_time = split_timelist_by_tdc[i][0]
            tdc_end_time = split_timelist_by_tdc[i][1]

            train_loader = self._get_dataloader(train_x, train_y, self.batch_size, shuffle, train_x_start_time, tdc_start_time, tdc_end_time)
            train_loader_list.append(train_loader)

        valid_loader = self._get_dataloader(valid_x, valid_y, self.batch_size, shuffle)

        return train_loader_list, valid_loader

    def _get_dataloader(self, data_x, data_y, batch_size, shuffle=True, original_start_time=None, tdc_start_time=None, tdc_end_time=None):
        
        if original_start_time:
            index_start=(pd.to_datetime(tdc_start_time) - pd.to_datetime(original_start_time)).days
            index_end=(pd.to_datetime(tdc_end_time) - pd.to_datetime(original_start_time)).days

            data_x=data_x[index_start: index_end + 1]
            data_y=data_y[index_start: index_end + 1]
        
        dataset = data_loader(data_x, data_y)
        dataload=DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataload

    def _tdc(self, k, data_x, start_time, end_time, dis_type = 'coral'):
        start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S') # 추후 csv 데이터로 time 읽어온 파라미터 타입 확인 후 삭제할지 말지 결정
        end_time = datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
        num_day = (end_time - start_time).days
        split_N = 10
        data=torch.tensor(data_x, dtype=torch.float32)
        data_shape_1 = data.shape[1] 
        data =data.reshape(-1, data.shape[2])
        data = data.cuda()

        selected = [0, 10]
        candidate = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        start = 0

        if k in [2, 3, 5, 7, 10]:
            while len(selected) -2 < k -1:
                distance_list = []
                for can in candidate:
                    selected.append(can)
                    selected.sort()
                    dis_temp = 0
                    for i in range(1, len(selected)-1):
                        for j in range(i, len(selected)-1):
                            index_part1_start = start + math.floor(selected[i-1] / split_N * num_day) * data_shape_1
                            index_part1_end = start + math.floor(selected[i] / split_N * num_day) * data_shape_1
                            data_part1 = data[index_part1_start: index_part1_end]
                            index_part2_start = start + math.floor(selected[j] / split_N * num_day) * data_shape_1
                            index_part2_end = start + math.floor(selected[j+1] / split_N * num_day) * data_shape_1
                            data_part2 = data[index_part2_start:index_part2_end]
                            criterion_transder = TransferLoss(loss_type= dis_type, input_dim=data_part1.shape[1])
                            dis_temp += criterion_transder.compute(data_part1, data_part2)
                    distance_list.append(dis_temp)
                    selected.remove(can)
                can_index = distance_list.index(max(distance_list))
                selected.append(candidate[can_index])
                candidate.remove(candidate[can_index]) 
            selected.sort()
            split_timelist_by_tdc = []  
            for i in range(1,len(selected)):
                if i == 1:
                    sel_start_time = start_time + datetime.timedelta(days = int(num_day / split_N * selected[i - 1]), hours = 0)
                else:
                    sel_start_time = start_time + datetime.timedelta(days = int(num_day / split_N * selected[i - 1])+1, hours = 0)
                sel_end_time = start_time + datetime.timedelta(days = int(num_day / split_N * selected[i]), hours =23)
                sel_start_time = datetime.datetime.strftime(sel_start_time,'%Y-%m-%d %H:%M')
                sel_end_time = datetime.datetime.strftime(sel_end_time,'%Y-%m-%d %H:%M')
                split_timelist_by_tdc.append((sel_start_time, sel_end_time))
            return split_timelist_by_tdc
        else:
            print("error in number of domain")

    def create_testloader(self, test_x, test_y, shuffle):
        test_loader = self._get_dataloader(test_x, test_y, self.batch_size, shuffle)
        return test_loader

    def create_inferenceloader(self, inference_x, shuffle=True):
        inference_x = torch.tensor(inference_x, dtype=torch.float32)
        inference_loader = DataLoader(inference_x, batch_size=self.batch_size, shuffle=shuffle)
        return inference_loader

    def save_model(self, save_model_path):
        torch.save(self.model.state_dict(), save_model_path)

    def load_model(self, model_file_path):
        self.model.load_state_dict(torch.load(model_file_path))


