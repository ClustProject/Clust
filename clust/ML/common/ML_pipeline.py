import sys, os
sys.path.append("../")
sys.path.append("../../")

############# Xy
def Xy_data_preparation(ingestion_param_X, data_y_flag, ingestion_param_y, ingestion_method, db_client):
    """
    학습 하기 위한 X, y 데이터를 준비

    Args:
        ingestion_param_X (dict): influx db에서 x 데이터를 읽기 위한 파라미터
        data_y_flag (_type_): y데이터가 별도로 있는지 없는지를 판단하는 파라미터
        ingestion_param_y (_type_):influx db에서 y 데이터를 읽기 위한 파라미터
        ingestion_method (_type_): influx db에서의 데이터 인출 방법
        db_client (_type_): influx db client

    Returns:
        pd.DataFrame : data_X, data_y

    """
    print(ingestion_param_X)
    from Clust.clust.data import data_interface
    data_X = data_interface.get_data_result(ingestion_method, db_client, ingestion_param_X)
    if data_y_flag:
        data_y = data_interface.get_data_result(ingestion_method, db_client, ingestion_param_y)

    else: # y 가 없다면 데이터를 만들어야 함
        feature_y_list = ingestion_param_y['feature_list']
        data_y = data_X[feature_y_list]
        
    return data_X, data_y


def Xy_data_scaling_train(data_name_X, data_X, data_name_y, data_y, scaler_param):
    """
    X, y 값에 대한 SCALING을 진행한 데이터 생성, 생성한 스케일러의 이름을 자동 생성하고 전달함

    Args:
        data_name_X (string): X 데이터 이름
        data_X (pd.DataFrame): X 데이터
        data_name_y (string): y 데이터 이름
        data_y (pd.DataFrame): y 데이터
        scaler_param (dict): scaler 관련 파라미터

    >>> scaler_param = { "scaler_path" : 스케일러를 저장하기 위한 기본 주소,
    ...                  "scaler_flag": 스케일링 적용 여부,
    ...                  "scale_method":스케일 방법 }

    Returns:
        pd.DataFrame : dataX_scaled, datay_scaled / 스케일링을 거친 X, y 데이터프레임

    """

    # X Data Scaling
    from Clust.clust.ML.tool import scaler
    scalerRootPath_X = os.path.join(scaler_param['scaler_path'], data_name_X)
    dataX_scaled, X_scalerFilePath = scaler.get_data_scaler(scaler_param['scaler_flag'], scalerRootPath_X, data_X, scaler_param['scale_method'])   
    print(X_scalerFilePath)
    # y Data Scaling
    scalerRootPath_y = os.path.join(scaler_param['scaler_path'], data_name_y)
    datay_scaled, y_scalerFilePath = scaler.get_data_scaler(scaler_param['scaler_flag'], scalerRootPath_y, data_y, scaler_param['scale_method'])
    
    scaler_param['scaler_file_path'] = {
        "XScalerFile":{
            "fileName":"scaler.pkl",
            "filePath":X_scalerFilePath       
        },
        "yScalerFile":{
            "fileName":"scaler.pkl",
            "filePath":y_scalerFilePath
        }
    }

    return dataX_scaled, datay_scaled

def Xy_data_scaling_test(data_X, data_y, X_scaler_file_path, y_scaler_file_path, scaler_param):
    """
    X, y 값에 대한 SCALING을 진행한 데이터 생성, 생성한 스케일러의 이름을 자동 생성하고 전달함

    Args:
        data_X (pd.DataFrame): X 데이터
        data_y (pd.DataFrame): y 데이터
        X_scaler_file_path (str): X 스케일러 파일 path
        y_scaler_file_path (str): y 스케일러 파일 path
        scaler_param (dict): scaler 관련 파라미터

    Returns:
        pd.DataFrame : test_X, test_y

    Returns:
        scaler : scaler_X, scaler_y

    """
    from Clust.clust.ML.tool import scaler as ml_scaler
    test_X, scaler_X = ml_scaler.get_scaled_test_data(data_X, X_scaler_file_path, scaler_param)
    test_y, scaler_y = ml_scaler.get_scaled_test_data(data_y, y_scaler_file_path, scaler_param)
    
    return test_X, scaler_X , test_y, scaler_y 

def X_data_scaling_infer(data_X, X_scaler_file_path, y_scaler_file_path, scaler_param):
    """
    X 값에 대한 SCALING을 진행한 데이터 생성, 생성한 스케일러의 이름을 자동 생성하고 전달함

    Args:
        data_X (pd.DataFrame): X 데이터
        X_scaler_file_path (str): X 스케일러 파일 path
        y_scaler_file_path (str): y 스케일러 파일 path
        scaler_param (dict): scaler 관련 파라미터

    Returns:
        pd.DataFrame : infer_X

    Returns:
        scaler : scaler_y

    """
    from Clust.clust.ML.tool import scaler as ml_scaler
    infer_X, scaler_X = ml_scaler.get_scaled_test_data(data_X, X_scaler_file_path, scaler_param)
    scaler_y = ml_scaler.get_scaler_file(y_scaler_file_path)
    
    return infer_X, scaler_y

############# Clean
def clean_low_quality_column(data, transform_info):
    """
    퀄러티가 좋지 않은 컬럼은 삭제함

    Args:
        data (pd.DataFrame):입력 데이터
        transform_info (dict): 어느정도의 퀄러티까지의 컬럼을 삭제할 것인가에 대한 기준

    Returns:
        pd.DataFrame : data

    """
    
    model_clean = transform_info['data_clean_option']
    nan_process_info = transform_info['nan_process_info']
    if model_clean:
        from Clust.clust.quality import quality_interface
        quality_param = {"nan_processing_param":nan_process_info}
        data = quality_interface.get_data_result("data_with_clean_feature", data, quality_param)
    
    
    else:
        pass
    return data

## Split
from Clust.clust.transformation.purpose import machineLearning as ML
from Clust.clust.ML.common import tool
def split_data_by_mode(X, y, split_ratio, transform_param):
    """학습 및 검증 데이터 분할

    Args:
        X (pd.DataFrame):입력 X 데이터
        y (pd.DataFrame):입력 y 데이터
        split_ratio (float): 학습/검증을 분할하기 위한 비율
        transform_param(dict): 데이터를 변환하기 위한 파라미터

    >>> transform_param = { "split_mode" : 분할하는 방법,
    ...                     "day_window_size": 일을 기준으로 한 윈도우 사이즈 }

    Returns:
        pd.DataFrame : train_x, val_x, train_y, val_y

    Returns:
        dictionary : transform_param(분할된 데이터 및 transform 파라미터)

    """
    if transform_param['split_mode'] =='window_split':
        transform_param['future_step'] = None
        transform_param['past_step'] = tool.get_default_day_window_size(X)
        print(transform_param)
        train_x, val_x = ML.split_data_by_ratio(X, split_ratio, transform_param['split_mode'], transform_param['past_step'])

    else:
        train_x, val_x = ML.split_data_by_ratio(X, split_ratio, None, None)
        
    train_y, val_y = ML.split_data_by_ratio(y, split_ratio, None, None)
    
    return train_x, val_x, train_y, val_y, transform_param

def transform_data_by_split_mode(transformParameter, X, y):
    """학습 직전의 배열 형태의 데이터 생성 (split 모드에 의해 준비)

    Args:
        transformParameter (dict): split mode 에 따른 변형을 위한 파라미터
        X (pd.DataFrame): x 데이터
        y (pd.DataFrame): y 데이터

    >>> transformParameter = { split mode, window_split or step_split }

    Returns:
        np.array : X_array, y_array(학습을 위해 최종적으로 준비된 데이터)

    """
    print("transformParameter['split_mode']: ", transformParameter['split_mode'])
    if transformParameter['split_mode'] =='window_split':
        from Clust.clust.transformation.type import DFToNPArray
        X_array, y_array= DFToNPArray.trans_DF_to_NP_by_windowNum(X, y, transformParameter)
        
    elif transformParameter['split_mode'] == 'step_split':
        X_array, y_array = ML.trans_by_step_info(X, y, transformParameter)
        
    return X_array, y_array


######train pipeline
def CLUST_regression_train(train_X_array, train_y_array, val_X_array, val_y_array, model_info):
    """regression 수행하고 적절한 모델을 저장함

    Args:
        train_X_array (np.array): 입력 train X
        train_y_array (np.array):입력 train y
        val_X_array (np.array): 입력 val X
        val_y_array (np.array):입력 val y
        model_info (dict): 모델 학습 정보

    >>> train_parameter = {'lr': 0.0001,
                            'weight_decay': 1e-06,
                            'device': 'cpu',
                            'n_epochs': 10,
                            'batch_size': 16}

    >>> model_method = [LSTM_rg|GRU_rg|CNN_1D_rg|LSTM_FCNs_rg]

    >>> modelParameter = {'rnn_type': 'lstm',
                            'input_size': 3,
                            'hidden_size': 64,
                            'num_layers': 2,
                            'output_dim': 1,
                            'dropout': 0.1,
                            'bidirectional': True}
            
    """
    from Clust.clust.ML.regression.train import RegressionTrain as RML
    train_parameter = model_info['train_parameter']
    model_method = model_info['model_method']
    model_parameter = model_info['model_parameter']
    model_file_path = model_info['model_file_path']['modelFile']['filePath']

    rml = RML()
    rml.set_param(train_parameter)
    rml.set_model(model_method, model_parameter)
    rml.set_data(train_X_array, train_y_array, val_X_array, val_y_array)
    rml.train()
    rml.save_best_model(model_file_path)
    

# Test pieline
from Clust.clust.ML.regression.test import RegressionTest as RT
def CLUST_regresstion_test(test_X_array, test_y_array, model_info):
    """ Regression Test

    Args:
        test_X_array (np.array): 입력 test X
        test_y_array np.array): 입력 test y
        model_info (dict): 모델 파라미터

    >>> model_info = { "model_method" : 모델 메서드,
    ...                "model_file_path": 모델 파일 패스, 
    ...                "model_parameter": 파라미터 }
    
    Returns:
        np.array : preds, trues /  예측값, 실제값

    """
    test_parameter = model_info['train_parameter']
    model_method = model_info['model_method']
    model_parameter = model_info['model_parameter']
    model_file_path = model_info['model_file_path']['modelFile']['filePath']

    rt = RT()
    rt.set_param(test_parameter)
    rt.set_model(model_method, model_file_path, model_parameter)
    rt.set_data(test_X_array, test_y_array)
    preds, trues = rt.test()
    
    return preds, trues


def get_scaler_information_by_y_flag(data_y_flag, scaler_X, scaler_y, feature_X_list, feature_y_list):
    """ get scaler for final prediction result by data_y_flag

    Args:
        data_y_flag (Bool):data_y_flag
        scaler_X (_type_):scaler X
        scaler_y (_type_): scaler Y
        feature_X_list (list): feature_X_list
        feature_y_list (list): feature_y_list

    Returns:
        scaler: scaler

    Returns:
        list :feature_list
    """
    if data_y_flag:
        scaler = scaler_y
        feature_list = feature_y_list
    else:
        scaler = scaler_X
        feature_list = feature_X_list
        
    return scaler, feature_list


def get_final_metrics(preds, trues, scaler_param, scaler, feature_list, target):
    """
    get final result (after inverse scaling), test metrics
    
    Args:
        preds (np.array): prediction value array
        trues (np.array): true value array
        scaler_param (string): with/without scaler
        scaler (scaler): scaler of prediction value
        feature_list (array): full featurelist of scaler
        target (string): target feature

    Returns:
        pd.dataframe : df_result

    Returns:
        dictionary : result_metrics(final result and test metrics)

    """
    from Clust.clust.tool.stats_table import metrics
    from Clust.clust.ML.tool import data as ml_data 
    df_result = ml_data.get_prediction_df_result(preds, trues, scaler_param, scaler, feature_list, target)
    result_metrics =  metrics.calculate_metrics_df(df_result)
        
    return df_result, result_metrics


# regression inference pipeline
from Clust.clust.ML.regression.inference import RegressionInference as RI
def CLUST_regression_inference(infer_X, model_info):
    """
    get inference prediction for regression model

    Args:
        infer_X (np.array): inference data X
        model_info (dict): model parameters

    Returns:
        np.array : preds(prediction value array)

    """

    inference_parameter = model_info['train_parameter']
    model_method = model_info['model_method']
    model_parameter = model_info['model_parameter']
    model_file_path = model_info['model_file_path']['modelFile']['filePath']

    ri = RI()
    ri.set_param(inference_parameter)
    ri.set_model(model_method, model_file_path, model_parameter)
    ri.set_data(infer_X)
    preds = ri.inference()
    
    return preds



# classification train pipeline
from Clust.clust.ML.classification.train import ClassificationTrain as CML
def CLUST_classification_train(train_X_array, train_y_array, val_X_array, val_y_array, model_info):
    """
    classification 수행하고 적절한 모델을 저장함

    Args:
        train_X_array (np.array): 입력 train X
        train_y_array (np.array):입력 train y
        val_X_array (np.array): 입력 val X
        val_y_array (np.array):입력 val y
        model_info (dict): 모델 학습 정보

    >>> train_parameter = {'lr': 0.0001,
                            'weight_decay': 1e-06,
                            'device': 'cpu',
                            'n_epochs': 10,
                            'batch_size': 16}

    >>> model_method = [LSTM_cf|GRU_cf|CNN_1D_cf|LSTM_FCNs_cf]

    >>> modelParameter = {'rnn_type': 'lstm',
                            'hidden_size': 64,
                            'num_layers': 2,
                            'output_dim': 1,
                            'dropout': 0.1,
                            'bidirectional': True,
                            'num_classes': 6}
            
    """

    train_parameter = model_info['train_parameter']
    model_method = model_info['model_method']
    model_parameter = model_info['model_parameter']
    model_file_path = model_info['model_file_path']['modelFile']['filePath']
    
    cml = CML()
    cml.set_param(train_parameter)
    cml.set_model(model_method, model_parameter)
    cml.set_data(train_X_array, train_y_array, val_X_array, val_y_array)
    cml.train()
    cml.save_best_model(model_file_path)


# classification test pipeline
from Clust.clust.ML.classification.test import ClassificationTest as CT
def clust_classification_test(test_X_array, test_y_array, model_info):
    """ Classification Test

    Args:
        test_X_array (np.array): 입력 test X
        test_y_array np.array): 입력 test y
        model_info (dict): 모델 파라미터
            
    >>> model_info = { "train_parameter" : 학습 파라미터,
    ...                "model_method" : 모델 메서드,
    ...                "model_file_path": 모델 파일 패스, 
    ...                "model_parameter": 파라미터 }

    
    Returns:
        np.array : preds, trues / 예측값, 실제값

    """

    test_parameter = model_info['train_parameter']
    model_method = model_info['model_method']
    model_parameter = model_info['model_parameter']
    model_file_path = model_info['model_file_path']['modelFile']['filePath']

    ct = CT()
    ct.set_param(test_parameter)
    ct.set_model(model_method, model_file_path, model_parameter)
    ct.set_data(test_X_array, test_y_array)
    preds, probs, trues, acc = ct.test()
    
    return preds, probs, trues, acc


from Clust.clust.ML.classification.inference import ClassificationInference as CI
def clust_classification_inference(infer_X, model_info):
    """
    get inference prediction for classification model

    Args:
        infer_X (np.array): inference data X
        model_info (dict): model parameters

    Returns:
        np.array : preds(prediction value array)

    """

    inference_parameter = model_info['train_parameter']
    model_method = model_info['model_method']
    model_parameter = model_info['model_parameter']
    model_file_path = model_info['model_file_path']['modelFile']['filePath']

    ci = CI()
    ci.set_param(inference_parameter)
    ci.set_model(model_method, model_file_path, model_parameter)
    ci.set_data(infer_X)
    preds = ci.inference()
    
    return preds