import sys, os
sys.path.append("../")
sys.path.append("../../")

############# Xy
def Xy_data_preparation(ingestion_param_X, data_y_flag, ingestion_param_y, ingestion_method, db_client):
    """학습 하기 위한 X, y 데이터를 준비

    Args:
        ingestion_param_X (dict): influx db에서 x 데이터를 읽기 위한 파라미터
        data_y_flag (_type_): y데이터가 별도로 있는지 없는지를 판단하는 파라미터
        ingestion_param_y (_type_):influx db에서 y 데이터를 읽기 위한 파라미터
        ingestion_method (_type_): influx db에서의 데이터 인출 방법
        db_client (_type_): influx db client

    Returns:
        data_X(pd.DataFrame), data_y(pd.DataFrame):각각의 데이터프레임
    """
    from Clust.clust.data import data_interface
    data_X = data_interface.get_data_result(ingestion_method, db_client, ingestion_param_X)
    if data_y_flag:
        data_y = data_interface.get_data_result(ingestion_method, db_client, ingestion_param_y)

    else: # y 가 없다면 데이터를 만들어야 함
        feature_y_list = ingestion_param_y['feature_list']
        data_y = data_X[feature_y_list]
        
    return data_X, data_y

def Xy_data_scaling_train(data_name_X, data_X, data_name_y, data_y, scaler_path, scaler_flag, scale_method):
    """X, y 값에 대한 SCALING을 진행한 데이터 생성, 생성한 스케일러의 이름을 자동 생성하고 전달함

    Args:
        data_name_X (string): X 데이터 이름
        data_X (pd.DataFrame): X 데이터
        data_name_y (string): y 데이터 이름
        data_y (pd.DataFrame): y 데이터
        scaler_path (str): 스케일러를 저장하기 위한 기본 주소
        scaler_flag (str): 스케일링 적용 여부
        scale_method (string): 스케일 방법

    Returns:
        _data_X(pd.DataFrame), data_y(pd.DataFrame):스케일링을 거친 각각의 데이터프레임
    """
    scalerRootPath_X = os.path.join(scaler_path, data_name_X)

    # X Data Scaling
    from Clust.clust.ML.tool import scaler
    scalerRootPath_X = os.path.join(scaler_path, data_name_X)
    dataX_scaled, X_scalerFilePath = scaler.get_data_scaler(scaler_flag, scalerRootPath_X, data_X, scale_method)   
    
    # X Data Scaling
    scalerRootPath_y = os.path.join(scaler_path, data_name_y)
    datay_scaled, y_scalerFilePath = scaler.get_data_scaler(scaler_flag, scalerRootPath_y, data_y, scale_method)
    
    return dataX_scaled, X_scalerFilePath, datay_scaled, y_scalerFilePath

from Clust.clust.ML.tool import scaler as ml_scaler
def Xy_data_scaling_test(data_X, data_y, X_scaler_file_path, y_scaler_file_path, scaler_param):
    """_summary_

    Args:
        data_X (_type_): _description_
        data_y (_type_): _description_
        X_scaler_file_path (_type_): _description_
        y_scaler_file_path (_type_): _description_
        scaler_param (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    test_X, scaler_X = ml_scaler.get_scaled_test_data(data_X, X_scaler_file_path, scaler_param)
    test_y, scaler_y = ml_scaler.get_scaled_test_data(data_y, y_scaler_file_path, scaler_param)
    return test_X, scaler_X , test_y, scaler_y 

############# Clean
def clean_low_quality_column(model_clean, nan_process_info, data):
    """퀄러티가 좋지 않은 컬럼은 삭제함

    Args:
        model_clean (bool): True/False
        nan_process_info (dict): 어느정도의 퀄러티까지의 컬럼을 삭제할 것인가에 대한 기준
        data (pd.DataFrame):입력 데이터

    Returns:
        data (pd.DataFrame):처리 데이터
    """
    if model_clean:
        from Clust.clust.quality import quality_interface
        quality_param = {"nan_processing_param":nan_process_info}
        data = quality_interface.get_data_result("data_with_clean_feature", data, quality_param)
    
    
    else:
        pass
    return data

## Split
from Clust.clust.transformation.purpose import machineLearning as ML
def split_data_by_mode(split_mode, split_ratio, dataX, datay, day_window_size):
    """학습 및 검증 데이터 분할

    Args:
        split_mode (str): 분할하는 방법
        split_ratio (float): 학습/검증을 분할하기 위한 비율
        dataX (pd.DataFrame):입력 X 데이터
        datay (pd.DataFrame):입력 y 데이터
        day_window_size (int): 일을 기준으로 한 윈도우 사이즈

    Returns:
        train_x, val_x, train_y, val_y (pd.DataFrame): 분할된 데이터
    """
    if split_mode =='windows_split':
        train_x, val_x = ML.split_data_by_ratio(dataX, split_ratio, split_mode, day_window_size)
    else:
        train_x, val_x = ML.split_data_by_ratio(dataX, split_ratio, None, None)

    train_y, val_y = ML.split_data_by_ratio(datay, split_ratio, None, None)
    
    return train_x, val_x, train_y, val_y

def transform_data_by_split_mode(split_mode, transformParameter, X, y):
    """학습 직전의 배열 형태의 데이터 생성 (split 모드에 의해 준비)

    Args:
        split_mode (str): _description_
        transformParameter (dict): split mode 에 따른 변형을 위한 파라미터
        X (pd.DataFrame): x 데이터
        y (pd.DataFrame): y 데이터

    Returns:
        X_array, y_array (np.array): 학습을 위해 최종적으로 준비된 데이터
    """
    if split_mode =='windows_split':
        from Clust.clust.transformation.type import DFToNPArray
        X_array, y_array= DFToNPArray.trans_DF_to_NP_by_windowNum(X, y, transformParameter)
        
    elif split_mode == 'step_split':
        X_array, y_array = ML.trans_by_step_info(X, y, transformParameter)
        
    return X_array, y_array


######train pipeline
def CLUST_regresstion_train(train_parameter, model_method, modelParameter, model_file_path, train_X_array, train_y_array, val_X_array, val_y_array):
    """regression 수행하고 적절한 모델을 저장함

    Args:
        train_parameter (dict): train_parameter 하기 위한 기본 정보
        model_method (string): 학습을 하기 위한 고유의 학습 방법
        modelParameter (_type_): model_method에 의한 학습에 필요한 파라미터
        model_file_path (_type_): 파일 저장 정보
        train_X_array (np.array): 입력 train X
        train_y_array (np.array):입력 train X
        val_X_array (np.array): 입력 train X
        val_y_array (np.array):입력 train X
    >>> train_parameter = {'lr': 0.0001,
            'weight_decay': 1e-06,
            'device': 'cpu',
            'n_epochs': 10,
            'batch_size': 16}
    >>> model_method = [LSTM_rg|GRU_rg|CNN_1D_rg|LSTM_FCNs_rg|FC_rg]
    >>> modelParameter = {'rnn_type': 'lstm',
                'input_size': 3,
                'hidden_size': 64,
                'num_layers': 2,
                'output_dim': 1,
                'dropout': 0.1,
                'bidirectional': True}
            
    """
    from Clust.clust.ML.regression.train import RegressionTrain as RML

    rml = RML()
    rml.set_param(train_parameter)
    rml.set_model(model_method, modelParameter)
    rml.set_data(train_X_array, train_y_array, val_X_array, val_y_array)
    rml.train()
    rml.save_best_model(model_file_path)
    

# Test pieline
from Clust.clust.ML.regression.test import RegressionTest as RT
def CLUST_regresstion_test(test_X_array, test_y_array, testParameter, model_method, model_file_path, modelParameter):
    """ Regression Test

    Args:
        test_X_array (np.array): 입력 test X
        test_y_array np.array): 입력 test y
        testParameter (dict): 테스트 파라미터
        model_method (str): 주요 테스트 메서드
        model_file_path (str): 모델 파일 패스
        modelParameter (dict): 파라미터

    Returns:
        preds, trues (np.arrau): 예측값, 실제값
    """
        
    rt = RT()
    rt.set_param(testParameter)
    rt.set_model(model_method, model_file_path, modelParameter)
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
        scaler, feature_list: scaler, feature list of scaler
    """
    if data_y_flag:
        scaler = scaler_y
        feature_list = feature_y_list
    else:
        scaler = scaler_X
        feature_list = feature_X_list
        
    return scaler, feature_list


def get_final_metrics(preds, trues, scaler_param, scaler, feature_list, target):
    """get final result (after inverse scaling), test metrics
    Args:
        preds (np.array): prediction value array
        trues (np.array): true value array
        scaler_param (string): with/without scaler
        scaler (_type_): scaler of prediction value
        feature_list (array): full featurelist of scaler
        target (string): target feature

    Returns:
        df_result(pd.dataframe), result_metrics(dict): final result and test metrics
    """
    from Clust.clust.tool.stats_table import metrics
    from Clust.clust.ML.tool import data as ml_data 
    df_result = ml_data.get_prediction_df_result(preds, trues, scaler_param, scaler, feature_list, target)
    result_metrics =  metrics.calculate_metrics_df(df_result)
        
    return df_result, result_metrics
