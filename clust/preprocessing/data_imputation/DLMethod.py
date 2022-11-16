import pandas as pd
import os

class DLImputation():
    def __init__ (self, data, method, parameter):
        self.method = method
        self.parameter = parameter
        self.data = data
        ####
        self.trainDataPathList =parameter['trainDataPathList']

    def getResult(self):
        result = self.data.copy()
        ### Brits
        if self.method == 'brits':
            print("brits_imputation")
            
            for column_name in self.data.columns:
                trainDataPathList = self.trainDataPathList
                trainDataPathList.append(column_name)
                ## Path
                from KETIToolDL import modelInfo
                MI = modelInfo.ModelFileManager()
                modelFilePath = MI.getModelFilePath(trainDataPathList, self.method)
                result = britsColumnImputation(self.data[[column_name]], column_name, modelFilePath)
                result[column_name] = result
        ### Define Another Imputation 
        else:
            result = self.data
        return result

## Define each DL imputation interface
def britsColumnImputation(data, column_name, modelPath):
    print(modelPath[0])
    if os.path.isfile(modelPath[0]):
        from KETIToolDL.PredictionTool.Brits import inference
        n =300
        dataset = [data[[column_name]][i:i+n] for i in range(0, len(data), n)]
        result = pd.DataFrame()
        print(len(data))
        for split_data in dataset:
            print(len(split_data))
            result_split = inference.BritsInference(split_data, column_name, modelPath).get_result()
            result = pd.concat([result, result_split])
    else:
        result = data.copy()
        print("No Brits Folder")

    return result

        
