import sys, os
sys.path.append("../")
sys.path.append("../..")
from Clust.clust.ML.common import model_path_setting

def get_model_file_path(trainDataPathList, method):
    """ get fullModelFilePath
    Ths function makes fullModelFilePath list.
    trainDataPathList and other paths obtained by method can be used for creating fullModelFilePath.

    :param trainDataPathList: It includes train data information to generate model path
    :type trainDataPathList: list of str

    :param method: train method
    :type method: str

    example
        >>>  from KETIToolDL import modelInfo
        >>>  MI = modelInfo.ModelFileManager()
        >>>  trainDataPathList =['DBName', 'MSName', 'columnName' ]
        >>>  trainMethod ='brits'
        >>>  modelFilePath = MI.getModelFilePath(trainDataPathList, self.trainMethod)
    """ 
    modelInfoList = model_path_setting.myModelInfoList
    modelInfo = modelInfoList[method]
    
    modelFullPath =modelInfo['modelRootPath']+modelInfo['modelInfoPath']+trainDataPathList
    modelFolderPath=''
    for addfolder in modelFullPath:
        modelFolderPath = os.path.join(modelFolderPath, addfolder)

    if not os.path.exists(modelFolderPath):
        os.makedirs(modelFolderPath)
    
    modelFilePath=[]
    for i, model_name in enumerate(modelInfo['modelFileNames']):
        modelFilePath.append(os.path.join(modelFolderPath, model_name))
    return modelFilePath