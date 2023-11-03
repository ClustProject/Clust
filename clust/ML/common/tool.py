import sys
sys.path.append("../")
sys.path.append("../../")
import pandas as pd

def random_nan_df(df, nan_ratio):
    """테스트를 위해 랜덤하게 nan 값을 넣음

    Args:
        df (pd.DataFrame): 입력
        nan_ratio (float): nan 적용 비율

    Returns:
        pd.DataFrame : df

    """
    for col in df.columns:
        df.loc[df.sample(frac=nan_ratio). index, col] = pd.np.nan
    return df

def get_default_day_window_size(data):
    """ 1일을 기준으로 그 안의 데이터 갯수를 세어 독립된 길이가 몇인지 찾는 함수

    Args:
        data (pd.DataFrame):입력 데이터

    Returns:
        integer : day_window_size(일을 기준으로 한 윈도우 사이즈)
        
    """
    from datetime import timedelta 
    # define window size by clust structure
    first_date = data.index[0]
    day_window_size = data.loc[first_date:first_date + timedelta(days =0, hours=23, minutes=59, seconds=59)].shape[0]
    
    return day_window_size

def get_default_model_name(model_name, app_name, model_method, model_clean):
    """ 모델 네임이 정의되지 않은 경우 모델 이름을 생성

    Args:
        model_name (_type_): _description_
        app_name (_type_): _description_
        model_method (_type_): _description_
        model_clean (_type_): _description_

    Returns:
        string : model name
    """
    if model_name is None:
        model_name = app_name+ '_'+ model_method + '_' + str(model_clean)
    else:
        pass
    return model_name

def get_default_model_path(model_name, data_name, model_method, train_parameter):
    """ 모델 이름에 따른 모델 패스를 지정

    Args:
        model_name (str): 모델 이름
        data_name (str): 활용한 데이터 이름
        model_method (str): 모델 이름
        train_parameter (dict): 학습을 위한 파라미터

    Returns:
        string : modelFilePath(최종 모델 파일 패스)
    """
    
    from Clust.clust.transformation.general.dataScaler import encode_hash_style
    trainParameter_encode =  encode_hash_style(str(train_parameter))
    trainDataPathList = [model_name, data_name, trainParameter_encode]
    from Clust.clust.ML.tool import model as ml_model
    modelFilePath = ml_model.get_model_file_path(trainDataPathList, model_method)
    
    return modelFilePath