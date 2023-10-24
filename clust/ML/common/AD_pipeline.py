import sys, os
sys.path.append("../")
sys.path.append("../../")

import pandas as pd

# for KETI_anamaly_detection code
# def trainset_from_local(params):
#     from YKTest.anomaly_detection import preprocess_data

#     TimeseriesData = preprocess_data.PickleDataLoad(data_type=params['data'], filename=params['filename'],
#                                                     augment_test_data=params['augment'])
#     train_dataset = TimeseriesData.batchify(TimeseriesData.trainData, params['batch_size'])
#     test_dataset = TimeseriesData.batchify(TimeseriesData.testData, params['eval_batch_size'])

#     return train_dataset, test_dataset

# Predictor train pipeline
def CLUST_anomalyDet_train(train_X, train_y, val_X, val_y, model_info):
    """ 
    모델 train 수행 후 모델 저장

    Args:
        train_X (DataFrame): 입력 train X
        train_y (DataFrame):입력 train y
        val_X (DataFrame): 입력 val X
        val_y (DataFrame):입력 val y
        model_info (dict): 학습 정보
    """
    from Clust.clust.ML.anomaly_detection.train import AnomalyDetTrain
    train_parameter = model_info['train_parameter']
    model_method = model_info['model_method']
    model_parameter = model_info['model_parameter']
    model_file_path = model_info['model_file_path']['modelFile']['filePath']    # TBD, to save model

    adt = AnomalyDetTrain()
    adt.set_param(train_parameter)
    adt.set_model(model_method, model_parameter)
    adt.set_data(train_X, train_y, val_X, val_y)
    adt.train()
    adt.save_best_model(model_file_path)

def CLUST_anomalyDet_test(test_X, test_y, model_info):
    """ 
    Anomaly Detection Test

    Args:
        test_X (DataFrame): 입력 test X
        test_y (DataFrame): 입력 test y
        model_info (dict): 모델 파라미터
        model_method (str): 모델 메서드
        model_file_path (str): 모델 파일 패스
        model_parameter (dict): 파라미터

    Returns:
        preds, trues (np.arrau): 예측값, 실제값
    """
    from Clust.clust.ML.anomaly_detection.test import AnomalyDetTest
    test_parameter = model_info['train_parameter']
    model_method = model_info['model_method']
    model_parameter = model_info['model_parameter']
    model_file_path = model_info['model_file_path']['modelFile']['filePath']

    adt = AnomalyDetTest()
    adt.set_param(test_parameter)
    adt.set_model(model_method, model_file_path, model_parameter)
    adt.set_data(test_X, test_y)
    preds, trues = adt.test()

    return preds, trues

def CLUST_anomalyDet_inference(infer_X, model_info):
    """
    get inference prediction for regression model
    
    Args:
        infer_X (np.array): inference data X
        model_info (dict): model parameters

    Returns:
        preds (np.array): prediction value array
    """
    from Clust.clust.ML.anomaly_detection.inference import AnomalyDetInference
    inference_parameter = model_info['train_parameter']
    model_method = model_info['model_method']
    model_parameter = model_info['model_parameter']
    model_file_path = model_info['model_file_path']['modelFile']['filePath']

    adi = AnomalyDetInference()
    adi.set_param(inference_parameter)
    adi.set_model(model_method, model_file_path, model_parameter)
    adi.set_data(infer_X)
    preds = adi.inference()
    
    return preds