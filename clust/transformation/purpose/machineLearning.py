import numpy as np

def split_data_by_ratio(data, split_ratio, mode=None, window_size=None):
        """
        Split Data By Ratio. It usually makes train/validation data and train/test data
        """
        if mode == "Classification": # Xdata Freq : 11분 15초 / ydata Freq : 1 Days
            data_date = np.unique(data.index.date)
            length1 = int(len(data_date)*split_ratio)
            data1, data2 = data[:str(data_date[length1-1])], data[str(data_date[length1]):]
            
        elif mode == "windows_split": # 입력 windows 를 기준으로 split
            import math
            round_num = math.ceil(len(data)/window_size)
    
            data_date = []
            for i in range(round_num):
                try:
                    data_date.append(data.iloc[[(i+1)*window_size-1]].index)
                except IndexError: # 결합 데이터의 길이가 공통 기간으로 기준 삼을 경우 y class data와 끝 시간이 일치하지 않은 경우 (Classification)
                    print("data length : ",len(data), " but  {} th windows index : ".format(i), (i+1)*window_size-1)
                    data_date.append(data.iloc[[len(data)-1]].index)

            length1 = int(len(data_date) * split_ratio)
            data1 = data[:str(data_date[length1-1][0])]
            data2 = data[len(data1):]
            
        else:
            length1=int(len(data)*split_ratio)
            data1, data2 = data[:length1], data[length1:]
        return data1, data2

def check_nan_status(np_X, np_y, nan_limit_num):
    nan_X = np.isnan(np_X).sum()
    # step1
    if (nan_limit_num > nan_X):
        ok = ~np.isnan(np_X)
        xp = ok.ravel().nonzero()[0]
        fp = np_X[~np.isnan(np_X)]
        x  = np.isnan(np_X).ravel().nonzero()[0]
        # Replacing nan values
        np_X[np.isnan(np_X)] = np.interp(x, xp, fp)
    else:
        pass
    return np_X, np_y

def trans_by_step_info(X, y, transformParameter):
    """transform for RNN style training

    Args:
        X (dataframe): _description_
        y (dataframe): _description_
        transformParameter (_type_): transform parameter

    Returns:
        X_array, y_array(numpy.array): transformed data for RNN style training
    """
    X_array, y_array =[], []
    n_steps = transformParameter['past_step']
    m_steps = transformParameter['future_step']
    max_nan_limit_ratio = transformParameter['max_nan_limit_ratio']
    nan_limit_num = int(n_steps*max_nan_limit_ratio)
    print("nan_limit_num: ", nan_limit_num)
    for i in range(n_steps, len(X)-m_steps):
        np_X = X.iloc[i-n_steps:i, :].values
        np_y = y.iloc[i+m_steps, :].values
        # step2
        np_X, np_y = check_nan_status(np_X, np_y, nan_limit_num)
        # step2
        if np.isnan(np_X).any() | np.isnan(np_y).any():
            pass
        else:
            X_array.append(np_X)
            y_array.append(np_y)
            
    X_array, y_array = np.array(X_array), np.array(y_array)
    print("Original num:", len(X), "Final num:", len(X_array), "NaN num:",  len(X)-len(X_array))
    return X_array, y_array




# TODO 아래 지울까?
class LSTMData():
    def __init__(self):
        pass
    
    # def getTorchLoader(self, X_arr, y_arr, batch_size):
    #     features = torch.Tensor(X_arr)
    #     targets = torch.Tensor(y_arr)
    #     dataSet = TensorDataset(features, targets)
    #     loader = DataLoader(dataSet, batch_size=batch_size, shuffle=False, drop_last=True)
    #     print("features shape:", features.shape, "targets shape: ", targets.shape)
    #     return dataSet, loader


    def transform_Xy_arr(self, data, transformParameter, clean_param=True):
        feature_col= transformParameter["feature_col"]
        target_col= transformParameter["target_col"]
        future_step= transformParameter["future_step"]
        past_step= transformParameter["past_step"]
        if clean_param==True:
            pass
        else:
            data = data.interpolate(method='linear').dropna()
        self.dataX, self.datay = self._split_Xy(data, feature_col, target_col)
        dataX_, datay_ = self._adapt_Xy_By_target_info(self.dataX, self.datay, future_step )
        self.dataX_arr, self.datay_arr  = self._get_clean_Xy(dataX_, datay_, past_step, clean_param)
        return self.dataX_arr, self.datay_arr

    def _split_Xy(self, data, X_col, y_col):
        X = data[X_col]
        y = data[[y_col]]
        return X, y

    def _adapt_Xy_By_target_info(self, X, y, future_num, method='step'):
        data_X= X[:-future_num]
        if method=='step':
            if future_num ==0:
                data_y = y
            else:
                data_y = y[future_num:]
        return data_X, data_y

    def _get_clean_Xy(self, X, y, past_step, clean_param):
            """
            If clean param is True -> get only data without NaN
            Clean Param is False -> get all data after linear interpolation
            """
            Clean_X, Clean_y = list(), list()
            Nan_num=0
            print("1. Original Data Lenagh:", len(X))
            # Remove set having any nan data
            for i in range(len(X)- past_step+1):
                seq_x = X[i:i+past_step].values
                seq_y = y.iloc[[i+past_step-1]].values
                if clean_param == True:
                    if np.isnan(seq_x).any() | np.isnan(seq_y).any():
                        Nan_num=Nan_num+1
                    else:
                        Clean_X.append(seq_x)
                        Clean_y.append(seq_y)
                else:
                    Clean_X.append(seq_x)
                    Clean_y.append(seq_y)
            print("2. Removed Data Length:", Nan_num)
            Clean_X = array(Clean_X)
            Clean_y = array(Clean_y).reshape(-1, len(y.columns))
            #Clean_y = array(Clean_y)
            print("3. Clean Data Leangth:", len(Clean_X))
            return Clean_X, Clean_y

    



"""
아래 코드 쓰이는가?
"""
import pandas as pd
import numpy as np
from numpy import array
class LearningDataSet():
    def __init__(self, learning_information):
        self.learning_information = learning_information
        self.future_num = learning_information['future_num']
        self.past_num = learning_information['past_num']
        self.target_feature = learning_information['target_feature']
        print("future num:", self.future_num)
    

    def get_LSTMStyle_X(self, data_X):
        print("self.past_num:", self.past_num)
        # if learning method is LSTM
        n_seq = 2
        learning_method = self.learning_information['learning_method']
        n_features = data_X.shape[-1]
        print(n_features)
        #n_features = len(data_X.columns)   
        if learning_method=='CNNLSTM':      
            n_steps = int(self.past_num/n_seq)
            data_X = data_X.reshape((data_X.shape[0],n_seq, n_steps, n_features ))
        elif learning_method =='ConvLSTM':
            # reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
            n_steps = int(self.past_num/n_seq)
            data_X = data_X.reshape((data_X.shape[0], n_seq, 1, n_steps, n_features))
        else:
            n_steps = self.past_num

        
        self.learning_information['n_seq'] = n_seq   
        self.learning_information['n_steps'] = n_steps 

        return data_X, self.learning_information
    
    # Separate the original dataset into X and y by the target colum, future num, method
    # multivariate multi-step stacked lstm example   
    # Method: mean, step, others

    def make_dataset_by_target(self, data, method='mean'):
        y = data[[self.target_feature]]
        data_y = pd.DataFrame()
        data_X= data[:len(data)-self.future_num]

        # method == step
        # if future_num is N, data_y(n) is y(n+future_num-1))
        if method=='step':
            data_y = y[(self.future_num-1):]

        #  method == mean
        # if future_num is N, data_y(n) is Mean(y(n)~y(n+(N-1))
        else: 
            for i in range(self.future_num):
                j = i
                y[self.target_feature+'+'+str(j)] = y[self.target_feature].shift(-j)
            y = y.drop(self.target_feature, axis=1)[:len(data)-self.future_num]
            if method=='mean':
                y = y.mean(axis=1)   
            elif method=='max':
                y = y.max(axis=1)
            elif method=='min':
                y = y.min(axis=1)
            else: 
                y = y.mean(axis=1) 
            data_y[self.target_feature+'_CurrentAndFuture_'+method+''+str(self.future_num)] = y
            # Modify the code below to adaptively change the shape of y depending on the situation 
            # by making more specific rules in the future
            
        return data_X, data_y
    
    
   
    


