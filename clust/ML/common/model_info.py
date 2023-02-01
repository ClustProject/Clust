import sys, os
sys.path.append("../")
sys.path.append("../..")
from Clust.clust.ML.common import model_path_setting

def get_model_file_path(train_data_path_list, method):
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
    model_info_list = model_path_setting.my_model_info_list
    model_info = model_info_list[method]
    
    model_full_path =model_info['model_root_path']+model_info['model_info_path']+train_data_path_list
    model_folder_path=''
    for add_folder in model_full_path:
        model_folder_path = os.path.join(model_folder_path, add_folder)

    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)
    
    model_file_path=[]
    for i, model_name in enumerate(model_info['model_file_names']):
        model_file_path.append(os.path.join(model_folder_path, model_name))
    return model_file_path