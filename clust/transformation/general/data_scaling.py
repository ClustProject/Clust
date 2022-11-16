import sys
sys.path.append("../")
sys.path.append("../..")
from clust.transformation.general import dataScaler
from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
import os
import joblib

class DataFrameScaling():
    def __init__(self, data, scaling_method):
        self.scaling_method = scaling_method
        self.scale_columns = dataScaler.get_scalable_columns(data)
        self.data = data

    #scaler Manipulation
    def set_scaler(self, scaler_file_name='None'):
        if os.path.isfile(scaler_file_name):
            self.scaler = self.set_scaler_from_file(scaler_file_name)
        else:
            scaler = RobustScaler()
            self.scaler = scaler.fit(self.data[self.scale_columns])
            self.save_scaler(scaler_file_name, self.scaler)
        return self.scaler
    
    def save_scaler(self, scaler_file_name, scaler):
        import os
        dir_name = os.path.dirname(scaler_file_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        joblib.dump(scaler, scaler_file_name)
        
    def set_scaler_from_file(self, scaler_file_name):
        self.scaler = joblib.load(scaler_file_name)
        return self.scaler
    
    # Scaling
    def scaling_datasets(self, dataset_list):
        t_dataset = []
        for dataset in dataset_list:
            dataset_t = dataset.copy()
            dataset_t = self.scaling_dataset(dataset_t)
            t_dataset.append(dataset_t)   
        return t_dataset
    
    def scaling_dataset(self,dataset_t):
        for t_method in self.scaling_method:
            if t_method =='scale':
                dataset_t= self._get_scaled_dataframe(dataset_t)
            if t_method =='log':
                dataset_t= self._get_log_dataframe(dataset_t)  
        return dataset_t
        
    def _get_scaled_dataframe(self, data):
        output_df = data.copy()
        input_d = data[self.scale_columns]
        output =self.scaler.transform(input_d)
        output_df[self.scale_columns] = output
        return output_df

    def _get_log_dataframe(self, data):
        output= data.copy()
        output[self.scale_columns] = np.log(output[self.scale_columns]+1)
        return output
    

class DataInverseScaling(): 
    def __init__(self,scaling_method, target_column, scaler ,scale_columns):
        self.scaling_method = scaling_method
        self.target_column = target_column
        self.scaler = scaler
        self.scale_columns = scale_columns

    def get_inv_Scaling_data(self, data):

        for t_method in self.scaling_method:
            if t_method =='log':
                inv_data= self._get_inverse_log_data(data)
            if t_method =='scale':
                inv_data= self._get_inverse_scaled_data(data)
        return inv_data
    
    def _get_inverse_scaled_data(self, data):
        dummy = pd.DataFrame(np.zeros((len(data), len(self.scale_columns))), columns=self.scale_columns)
        dummy[self.target_column] = data
        dummy = pd.DataFrame(self.scaler.inverse_transform(dummy), columns = self.scale_columns)
        return dummy[self.target_column].values

    def _get_inverse_log_data(self, data):
        output = np.exp(data)-1
        return output
