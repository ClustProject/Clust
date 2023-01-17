
# 여기에는 독립적으로 모델을 생성하고 저장하는 부분만 기술되도록
import sys
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from KETIToolDL.TrainTool.Brits import Brits_model

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
class BritsTraining():
    def __init__(self, data, json_path):
        ## TODO MODIFY 1000
        self.inputData = data
        self.json_path = json_path
 
    def train(self, epoch=100, learning_rate=0.01):
        # setting
        torch.random.manual_seed(0)
        np.random.seed(0)
        print(self.inputData)
     
        Brits_model.makedata(self.inputData, self.json_path)
        data_iter = Brits_model.get_loader(self.json_path, batch_size=64)
        length = len(self.inputData)
        model = Brits_model.Brits_i(108, 1, 0, length, device).to(device)
        
        # Brits_model.Brits_i(hidden_state_dim, impute_weight, label_weight, length, device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        model.train()

        progress = tqdm(range(epoch))
        loss_graphic = []
        for i in progress:
            total_loss = 0.0
            for idx, data in enumerate(data_iter):
                data = Brits_model.to_var(data, device)
                ret = model.run_on_batch(data, optimizer, i)
                total_loss += ret["loss"]
            loss_graphic.append(total_loss.tolist())
            progress.set_description("loss: {:0.4f}".format(total_loss / len(data_iter)))
        return model