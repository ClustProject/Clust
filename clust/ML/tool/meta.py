db_name ='model'
collection_name ='meta'

def get_model_meta_data(mongodb_client, search):
    meta = mongodb_client.get_document_by_json(db_name, collection_name, search)
    model_meta = meta[0]
    return model_meta

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
