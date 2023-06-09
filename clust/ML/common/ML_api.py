import sys

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from Clust.clust.ML.common import ML_pipeline

def chagne_type_str_to_bool(dict_data):

    for key, value in dict_data.items():

        if isinstance(value, dict):
            dict_data[key] = chagne_type_str_to_bool(value)

        elif isinstance(value, str):

            if value.lower() == 'true':
                dict_data[key] = True
            elif value.lower() == 'false':
                dict_data[key] = False
            elif value.lower() == 'none':
                dict_data[key] = None

    return dict_data

def check_model_name(model_name, model_name_info):
    """It makes model name by default value and additional information

    Args:
        model_name (string): default model name
        model_name_info (array): model name information

    Returns:
        model_name(str): final model name
    """
    # model name & path
    if model_name is None or model_name == 'None':
        model_name=""
        for key in model_name_info:
            model_name+=key+'_'
        
    return model_name

def get_train_data_meta(meta_client, params):
    """get train data meta information

    Args:
        meta_client (mongodb client):mongodb
        params (dict): it must include 'bk_name_X', and 'ms_name_X' keys.

    Returns:
        result(dict): measurement information meta
    """
    bk_name = params['bucket_name']
    ms_name = params['ms_name']
    data_meta = meta_client.get_document_by_json(bk_name, ms_name, {'ms_name': ms_name})  
    try:
        result = data_meta[0]
    except:
        result = {}
        
    return result

def ML_data_preparation(param, influxdb_client):
    # 1. Oirignla data ingestion
    data_X, data_y = ML_pipeline.Xy_data_preparation(param['ingestion_param_X'], 
                                                 param['data_y_flag'], 
                                                 param['ingestion_param_y'],
                                                 'ms_all', 
                                                 influxdb_client)
    # 2. Scaling
    dataX_scaled, datay_scaled, scale_file_path_info = ML_pipeline.Xy_data_scaling_train(param['ingestion_param_X']['ms_name'], 
                                                                                     data_X, 
                                                                                     param['ingestion_param_y']['ms_name'], 
                                                                                     data_y, 
                                                                                     param['scaler_param'])
    
    
    
    # 3.clean column
    dataX_scaled = ML_pipeline.clean_low_quality_column(dataX_scaled, 
                                                        param['transform_param'])

    # 4. split train/Val
    split_ratio = 0.8
    train_X, val_X, train_y, val_y, param['transform_param']= ML_pipeline.split_data_by_mode(dataX_scaled, 
                                                                                             datay_scaled, 
                                                                                             split_ratio, 
                                                                                             param['transform_param'])
    
    # 5. Transform array style
    train_X_array, train_y_array = ML_pipeline.transform_data_by_split_mode(param['transform_param'], 
                                                                            train_X, 
                                                                            train_y)
    val_X_array, val_y_array = ML_pipeline.transform_data_by_split_mode(param['transform_param'], 
                                                                        val_X, 
                                                                        val_y)
        
        
    
    return train_X_array, train_y_array, val_X_array, val_y_array

def ML_training(train_X_array,  train_y_array, val_X_array, val_y_array, param):
    # model info update

    from Clust.clust.ML.common import model_parameter_setting
    param['model_info']['seq_len'] = train_X_array.shape[1] 
    param['model_info']['input_size'] = train_X_array.shape[2] 
    param['model_info']['model_parameter'] = model_parameter_setting.set_model_parameter(param['model_info']) 
    model_info = param['model_info']

    from Clust.clust.ML.tool import model as ml_model
    train_data_path_list = [model_info['model_name'] , param['ingestion_param_X']['ms_name']]
    model_file_path = ml_model.get_model_file_path(train_data_path_list, model_info['model_method'] )


    param['model_info']['model_file_path'] = {
        "modelFile":{
            "fileName":"model.pth",
            "filePath":model_file_path
        }
    }

    # model training
    # input 순서 일관되도록 펑션 수정
    if model_info['model_purpose'] == 'regression':    
        ML_pipeline.CLUST_regresstion_train(train_X_array, 
                                            train_y_array, 
                                            val_X_array, 
                                            val_y_array,
                                            param['model_info']
                                            )
    elif model_info['model_purpose']  == 'classification':
        ML_pipeline.CLUST_classification_train(train_X_array, 
                                               train_y_array, 
                                               val_X_array, 
                                               val_y_array, 
                                               param['model_info'])


    return param


