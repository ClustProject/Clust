
import torch
import os
from Clust.clust.ML.test_sota.brits import brits_model
import copy 
import numpy as np
from sklearn.preprocessing import StandardScaler

from Clust.clust.ML.common import inference

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class BritsInference(inference):
    def __init__(self, data, column_name, model_path):
        self.inputData = data
        self.column_name = column_name
        self.model_path = model_path

    def get_result(self):
        output = self.inputData.copy()
        if os.path.isfile(self.model_path[0]):
            print("Brits Model exists")
            print(self.model_path[0])
            loaded_model = brits_model.Brits_i(108, 1, 0, len(output), device).to(device)
            loaded_model.load_state_dict(copy.deepcopy(torch.load(self.model_path[1], device)))
            
            brits_model.makedata(output, self.model_path[0])
            data_iter = brits_model.get_loader(self.model_path[0], batch_size=64)
            
            result = self.predict_result(loaded_model, data_iter, device, output)
            result_list = result.tolist()
            nan_data = output[output.columns[0]].isnull()
            for i in range(len(nan_data)):
                if nan_data.iloc[i] == True:
                    output[output.columns[0]].iloc[i] = result_list[i]
        else:
            print("No Brits Model File")
            pass
        
        return output
    
    def predict_result(self, model, data_iter, device, data):
        imputation = self.evaluate(model, data_iter, device)
        scaler = StandardScaler()
        scaler = scaler.fit(data[self.column_name].to_numpy().reshape(-1,1))
        result = scaler.inverse_transform(imputation[0])
        return result[:, 0]

    def evaluate(self, model, data_iter, device):
        model.eval()
        imputations = []
        for idx, data in enumerate(data_iter):
            data = brits_model.to_var(data, device)
            ret = model.run_on_batch(data, None)
            eval_masks = ret['eval_masks'].data.cpu().numpy()
            imputation = ret['imputations'].data.cpu().numpy()
            imputations += imputation[np.where(eval_masks == 1)].tolist()
        imputations = np.asarray(imputations)
        return imputation