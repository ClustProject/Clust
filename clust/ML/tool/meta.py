db_name ='model'
collection_name ='meta'


def save_model_meta_data(mongodb_client, model_meta):
    try :
        result = mongodb_client.insert_document(db_name, collection_name, model_meta)
        print("======== OK ========")
        print(result)
        return 200
    except Exception as e : 
        print("======== Fail ========")
        print(e)
        return 500


def set_meta_for_train_data(data_meta, split_mode, feature_X_list, feature_y_list, data_y_flag, data_clean_option, transform_parameter, scaler_param, data_name_X, data_name_y, X_scalerFilePath, y_scalerFilePath):
    model_info_meta ={
        "trainDataInfo":data_meta,
        "dataSplitMode": split_mode,
        "featureList":feature_X_list,
        "target": feature_y_list,
        "data_y_flag": data_y_flag,
        "data_clean_option":data_clean_option,
        "transformParameter":transform_parameter,
        "scalerParam":scaler_param,
        "trainDataName":[data_name_X, data_name_y], 

        "files":{
            "modelFile":{
                "fileName":"",
                "filePath":""
            },
            "XScalerFile":{
                "fileName":"scaler.pkl",
                "filePath":X_scalerFilePath       
            },
            "yScalerFile":{
                "fileName":"scaler.pkl",
                "filePath":y_scalerFilePath      
            }
        }
    }

    return model_info_meta


def set_meta_for_model_data(model_info_meta, model_name, model_purpose, model_method, model_tags, train_parameter, model_parameter, model_file_path):

    model_info_meta['modelName'] = model_name
    model_info_meta['modelPurpose'] = model_purpose
    model_info_meta['modelMethod'] = model_method
    model_info_meta['modelTags'] = model_tags
    model_info_meta['trainParameter'] = train_parameter
    model_info_meta['modelParameter'] = model_parameter

    model_info_meta['files']['modelFile']['fileName'] = "model.pth"
    model_info_meta['files']['modelFile']['filePath'] = model_file_path


    return model_info_meta



def model_meta_update(data_meta, model_name, split_mode, feature_X_list, feature_y_list, data_y_flag, model_purpose, model_method, model_tags, model_clean, train_parameter, model_parameter, transform_parameter, scaler_param, data_name_X, data_name_y, model_file_path, X_scalerFilePath, y_scalerFilePath):
    model_info_meta ={
        "trainDataInfo":data_meta,
        "modelName":model_name,
        "dataSplitMode": split_mode,
        "featureList":feature_X_list,
        "target": feature_y_list,
        "data_y_flag": data_y_flag,
        "trainDataType":'timeseries',
        "modelPurpose":model_purpose,
        "modelMethod":model_method,
        "modelTags":model_tags,
        "modelCleanLevel":model_clean,
        "trainParameter": train_parameter,
        "modelParameter": model_parameter,
        "transformParameter":transform_parameter,
        "scalerParam":scaler_param,
        "trainDataName":[data_name_X, data_name_y], 

        "files":{
            "modelFile":{
                "fileName":"model.pth",
                "filePath":model_file_path
            },
            "XScalerFile":{
                "fileName":"scaler.pkl",
                "filePath":X_scalerFilePath       
            },
            "yScalerFile":{
                "fileName":"scaler.pkl",
                "filePath":y_scalerFilePath      
            }
        }
    }

    return  model_info_meta



# def set_meta_for_train_data(model_info_meta, data_meta, split_mode, feature_X_list, feature_y_list, data_y_flag, data_clean_option, transform_parameter, scaler_param, data_name_X, data_name_y, X_scalerFilePath, y_scalerFilePath):

#     model_info_meta['trainDataInfo'] = data_meta
#     model_info_meta['dataSplitMode'] = split_mode
#     model_info_meta['featureList'] = feature_X_list
#     model_info_meta['target'] = feature_y_list
#     model_info_meta['data_y_flag'] = data_y_flag
#     model_info_meta['data_clean_option'] = data_clean_option
#     model_info_meta['trainDataType'] = 'timeseries'
#     model_info_meta['transformParameter'] = transform_parameter
#     model_info_meta['scalerParam'] = scaler_param
#     model_info_meta['trainDataName'] = [data_name_X, data_name_y]

#     model_info_meta['files']['XScalerFile']['fileName'] = "scaler.pkl"
#     model_info_meta['files']['XScalerFile']['filePath'] = X_scalerFilePath
#     model_info_meta['files']['XScalerFile']['fileName'] = "scaler.pkl"
#     model_info_meta['files']['XScalerFile']['filePath'] = y_scalerFilePath

#     return model_info_meta