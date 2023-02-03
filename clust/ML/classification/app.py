import sys
import pandas as pd

sys.path.append("../")
sys.path.append("../../")

from sklearn.metrics import classification_report
from Clust.clust.ML.common import model_manager
from Clust.clust.ML.common.common import p2_dataSelection as p2
from Clust.clust.ML.common.common import p4_testing as p4
from Clust.clust.ML.classification.test import ClassificationTest as CT
from Clust.clust.ML.classification.inference import ClassificationInference as CI

def get_test_result(data_name_X, data_name_y, data_meta, model_meta, data_folder_name=None, window_num=0, db_client=None):
    
    data_save_mode_X = data_meta[data_name_X]["integrationInfo"]["DataSaveMode"]
    data_save_mode_y = data_meta[data_name_y]["integrationInfo"]["DataSaveMode"]
    data_X = p2.getSavedIntegratedData(data_save_mode_X, data_name_X, data_folder_name)
    data_y = p2.getSavedIntegratedData(data_save_mode_y, data_name_y, data_folder_name)
    
    X_scaler_file_path = model_meta['files']['XScalerFile']["filePath"]
    y_scaler_file_path = model_meta['files']['yScalerFile']["filePath"]
    model_file_path = model_meta['files']['modelFile']["filePath"]

    feature_list = model_meta["featureList"]
    target = model_meta["target"]
    scaler_param = model_meta["scalerParam"]
    model_method = model_meta["model_method"]
    train_parameter = model_meta["trainParameter"]

    dim = None
    if model_method == "FC_cf":
        dim = 2

    test_X, scaler_X = p4.getScaledTestData(data_X[feature_list], X_scaler_file_path, scaler_param)
    test_y, scaler_y = p4.getScaledTestData(data_y[target], y_scaler_file_path, scaler_param)# No Scale

    # test_y = datay[target] # for classification

    # 4. Testing
    batch_size=1
    train_parameter['batch_size'] = 1
    scaler_param="noScale" # for classification


    ct = CT()
    ct.set_param(train_parameter)
    ct.set_data(test_X, test_y, window_num, dim)
    model = model_manager.load_pickle_model(model_file_path)
    preds, probs, trues, acc =  ct.get_result(model)

    result_metrics = classification_report(trues, preds, output_dict = True)
    df_result = p4.getPredictionDFResult(preds, trues, scaler_param, scaler_y, featureList= target, target_col = target[0])
    
    return df_result, result_metrics, acc




def get_inference_result(data_X, model_meta, window_num=0, db_client=None):
    
    X_scaler_file_path = model_meta['files']['XScalerFile']["filePath"]
    y_scaler_file_path = model_meta['files']['yScalerFile']["filePath"]
    model_file_path = model_meta['files']['modelFile']["filePath"]

    feature_list = model_meta["featureList"]
    target = model_meta["target"]
    scaler_param = model_meta["scalerParam"]
    model_method = model_meta["model_method"]
    train_parameter = model_meta["trainParameter"]

    dim = None
    if model_method == "FC_cf":
        dim = 2

    input_X, scaler_X = p4.getScaledTestData(data_X[feature_list], X_scaler_file_path, scaler_param)
    # sacler_y = p4.getScalerFromFile(y_scaler_file_path)

    train_parameter['batch_size'] = 1
    scaler_param="noScale" # for classification

    print(scaler_param)

    ci = CI()
    ci.set_param(train_parameter)
    ci.set_data(input_X, window_num, dim)
    model = model_manager.load_pickle_model(model_file_path)
    preds =  ci.get_result(model)


    if scaler_param =='scale':
        base_df_for_inverse = pd.DataFrame(columns=target, index=range(len(preds)))
        base_df_for_inverse[target] = preds
        prediction_result = pd.DataFrame(scaler_X.inverse_transform(base_df_for_inverse), columns=target, index=base_df_for_inverse.index)
    else:
        prediction_result = pd.DataFrame(data={'value':preds}, index=range(len(preds)))

    return prediction_result
