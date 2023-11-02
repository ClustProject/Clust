import torch 
import numpy as np 
import pandas as pd 
from torch.utils.data import Dataset 

class BeatganDataset(Dataset):
    def __init__(self, data_x: pd.DataFrame, data_y: pd.DataFrame = None,
                 window_size: int = 320, stride: int = 1,
                 inference:bool = False):
        super(BeatganDataset, self).__init__()
        '''
        Dataset for BeatGAN 
        
        Args:
            data_x (pd.DataFrame) : time sereis data for input 
            data_y (str) : anomaly annotation data (ground truth) 
            window_size (int) : the length of time series 
            stride (int) : the number of stride 
            
        Example:
            df = pd.read_csv('./data/train_ver2.csv')
            df = df.drop(columns = 'timestamp')
            dataset = BeatganDataset(
                        data_x        = train_x,
                        data_y        = train_y,
                        window_size   = 320,
                        stride        = 1 
                    )
        '''
        self.X = data_x
        self.Y = data_y
        
        self.window_size = window_size
        self.stride = stride 
        
        self.inference = inference
        
    def __len__(self):
        return int((len(self.X)-self.window_size)//self.stride)
    
    def __getitem__(self, idx):
        '''
        Return:
            if dataset is inference_set, return only y 
            else return both x and y 
        '''
        # window slicing 
        x = self.X.iloc[idx*self.stride : idx*self.stride + self.window_size].values 
        
        if self.Y is not None:
            y = self.Y.iloc[idx*self.stride : idx*self.stride + self.window_size].values 
        
        # Transpose x from (w,c) -> (c,w) for CNN 
        x = torch.Tensor(np.transpose(x,axes=(1,0))).type(torch.float32)
        
        # Aggregation of label y : choose last label 
        if self.Y is not None:
            y = y.max()
            return x,y 
        
        else:
            return x
        