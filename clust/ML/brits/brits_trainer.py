import os
import sys
sys.path.append("..")
sys.path.append("../..")

# Model 1: Brits
import torch
import torch.nn as nn
import torch.optim as optim
from Clust.clust.ML.brits.train import BritsTraining
from Clust.clust.ML.common.trainer import Trainer


# Model 1: Brits
class BritsTrainer(Trainer):
    def _train_save_model(self, df): 
        Brits = BritsTraining(df, self.modelFilePath[0])
        model = Brits.train()
        torch.save(model.state_dict(), self.modelFilePath[1])
        print(self.modelFilePath)
