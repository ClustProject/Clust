import sys
sys.path.append("../")
sys.path.append("../..")
import datetime

def set_default_param():
    default_param={}
    ## 1. refine_param
    data_freq_min = 60
    refine_frequency = datetime.timedelta(minutes= data_freq_min)

    default_param['data_refinement'] = {"remove_duplication": {'flag': True}, 
                    "static_frequency": {'flag': True, 'frequency': refine_frequency}}
    
    ## 2. outlier_param

    default_param['data_outlier'] ={
        "certain_error_to_NaN": {'flag': True, }, 
        "uncertain_error_to_NaN":{'flag': False}}
    
    ## 3. split_param
    default_param['data_split']={
        "split_method":"cycle",
        "split_param":{
            'feature_cycle' : "Day",
            'feature_cycle_times' : 1}
    }
    
    ## 4. select_param
    default_param['data_selection']={
        "select_method":"keyword_data_selection", 
        "select_param":{
            "keyword":"*"
        }
    }
    
    ## 5. integration_param
    data_freq_min = 60 
    integration_frequency = datetime.timedelta(minutes= data_freq_min)

    default_param['data_integration']={
        "integration_param":{"feature_name":"in_co2", "duration":None, "integration_frequency":integration_frequency},
        "integration_type": "one_feature_based_integration"
    }
    
    ## 6. quality_param
    default_param['data_quality_check'] = {
        "quality_method" : "data_with_clean_feature", 
        "quality_param" : {
            "nan_processing_param":{
                'type':"num", 
                'ConsecutiveNanLimit':4, 
                'totalNaNLimit':24}}
    }
    
    ## 7. imputation_param
    default_param['data_imputation'] = {
                        "flag":True,
                        "imputation_method":[{"min":0,"max":300,"method":"linear", "parameter":{}}, 
                                            {"min":0,"max":10000,"method":"mean", "parameter":{}}],
                        "total_non_NaN_ratio":1 }
    
    ## 8. smoothing_param
    default_param['data_smoothing']={'flag': True, 'emw_param':0.3}
    
    ## 9. scaling_param
    default_param['data_scaling']={'flag': True, 'method':'minmax'} 
    return default_param

def set_outlier_param(param):
    """ 
    외부 파라미터를 입력받아 다시 한번 outlier parameter로 생성하는 함수

    Args:
        param (dict): outlier 관련 파라미터
    
    Note
    -------
    uncertain_error_to_NaN.flag가 True일 경우 uncertain_error_to_NaN.algorithm 에 따라 uncertain_error_to_NaN 에서 입력 받는 key들이 달라짐.  

    Example:
        uncertain_error_to_NaN의 Parameter 예시
        1. algorithm : SR
        >>> param = {
        ...     'certain_error_to_NaN': {'flag': True},
        ...     'uncertain_error_to_NaN': {
        ...         'flag': True,
        ...         'outlier_detector_config':{
        ...             'algorithm': 'SR',
        ...             'percentile': 95,
        ...             'alg_parameter' : {'period': 24}
        ...         }
        ...     }}}
        
        2. algorithm : IF
        >>> param = {
        ...     'certain_error_to_NaN': {'flag': True},
        ...     'uncertain_error_to_NaN': {
        ...         'flag': True,
        ...         'outlier_detector_config':{ 
        ...             'algorithm': 'IF',
        ...             'percentile': 95,
        ...             'alg_parameter' : {'estimator': 100}
        ...         }
        ...     }}

        3. algorithm : KDE
        >>> param = {
        ...     'certain_error_to_NaN': {'flag': True},
        ...     'uncertain_error_to_NaN': {
        ...         'flag': True,
        ...         'outlier_detector_config':{
        ...             'algorithm': 'KDE',
        ...             'percentile': 95,
        ...             'alg_parameter' : {'leaf_size': 40}
        ...         }
        ...     }}

        4. algorithm : LOF
        >>> param = {
        ...     'certain_error_to_NaN': {'flag': True},
        ...     'uncertain_error_to_NaN': {
        ...         'flag': True,
        ...         'outlier_detector_config':{
        ...             'algorithm': 'LOF',
        ...             'percentile': 95,
        ...             'alg_parameter' : {'neighbors': 20}
        ...         }
        ...     }}

        5. algorithm : MoG
        >>> param = {
        ...     'certain_error_to_NaN': {'flag': True},
        ...     'uncertain_error_to_NaN': {
        ...         'flag': True,
        ...         'outlier_detector_config':{ 
        ...             'algorithm': 'MoG',
        ...             'percentile': 95,
        ...             'alg_parameter' : {'component': 1}
        ...         }
        ...     }}

        6. algorithm : IQR
        >>> param = {
        ...     'certain_error_to_NaN': {'flag': True},
        ...     'uncertain_error_to_NaN': {
        ...         'flag': True,
        ...         'outlier_detector_config':{
        ...             'algorithm': 'IQR',
        ...             'percentile': 95,
        ...             'alg_parameter' : {'weight': 100}
        ...         }
        ...     }}

        7. algorithm : SD
        >>> param = {
        ...     'certain_error_to_NaN': {'flag': True},
        ...     'uncertain_error_to_NaN': {
        ...         'flag': True, 
        ...         'outlier_detector_config':{
        ...             'algorithm': 'SD',
        ...             'percentile': 95,
        ...             'alg_parameter' : {'period': 24, 'limit': 15}
        ...         }
        ...     }}

    Returns:
        data_outlier_param: 최종 파라미터
    """
    
    data_outlier_param = {}
    certain_param=param['certain_error_to_NaN']
    if certain_param['flag']:
        certain_param['abnormal_value_list'] = {'all':[99.9, 199.9, 299.9, 9999, -99.9, -199.9, -299.9, -9999, -9999.0]} 
        certain_param['data_min_max_limit']  = {'max_num': {'in_temp': 80,
            'in_humi': 100,
            'in_co2': 10000,
            'in_voc': 60000,
            'in_noise': 90,
            'in_pm10': 1000,
            'in_pm25': 1000,
            'in_pm01': 1000},
            'min_num': {'in_temp': -40,
            'in_humi': 0,
            'in_co2': 0,
            'in_voc': 0,
            'in_noise': 35,
            'in_pm10': 0,
            'in_pm25': 0,
            'in_pm01': 0}} # TODO
        
    uncertain_param = param['uncertain_error_to_NaN']
    if uncertain_param['flag']:
        outler_alg_config = uncertain_param['outlier_detector_config']
        outler_alg_config['alg_parameter'] = get_outlier_detection_param(outler_alg_config['algorithm'], outler_alg_config['alg_parameter'])
        uncertain_param ['param'] = {'outlierDetectorConfig' : [outler_alg_config]}
        del uncertain_param['outlier_detector_config']
        
    data_outlier_param['certain_error_to_NaN'] = certain_param
    data_outlier_param['uncertain_error_to_NaN'] = uncertain_param

    return data_outlier_param

def get_outlier_detection_param(algorithm, param):
    """ outlier detection을 위한 parameter로 알고리즘과 이에 근거한 parameter를 입력 받으

    Args:
        algorithm (string): outlier detection algorithm ['SR'|'IF'| 'KDE'| 'LOF'|'MoG'|'IQR'| 'SD']
        param (dictionary): parameter for algorithm
        >>> SR - period
        >>> IF - percentile,  estimator
        >>> KDE - leaf_size
        >>> LOF - neighbors, percentile
        >>> MoG - component
        >>> IQR - weight
        >>> SD - period, limit
        

    Returns:
        result(dict): final parameter
    """
    
    if algorithm =='SR':
        period = param['period'] # period는 신호의 몇개 단위가 주기인지에 대한 적절한 값
        result = { 
            # Multivariable 불가능
            'SR_series_window_size': int(period/2), # less than period, int, 데이터 크기에 적합하게 설정
            'SR_spectral_window_size': period, # as same as period, int, 데이터 크기에 적합하게 설정
            'SR_score_window_size': period *  2 
        }
    elif algorithm =='IF':
        percentile = param['percentile']
        estimator = param ['estimator'] # 100
        result = { # Estimators (1~100)
            # percentile만 조절
            'IF_estimators': estimator, # ensemble에 활용하는 모델 개수, i(default: 100, 데이터 크기에 적합하게 설정) 
            'IF_max_samples': 'auto', # 각 모델에 사용하는 샘플 개수(샘플링 적용), int or float(default: 'auto') 
            'IF_contamination': (100-percentile)/100, #'auto', # 모델 학습시 활용되는 데이터의 outlier 비율, ‘auto’ or float(default: ’auto’, float인 경우 0 초과, 0.5 이하로 설정)
            'IF_max_features': 1.0, # 각 모델에 사용하는 변수 개수(샘플링 적용), int or float(default: 1.0)
            'IF_bootstrap': True} # bootstrap적용 여부, bool(default: False)
        
    elif algorithm == 'KDE':
        leaf_size = param['leaf_size'] # 40
        result = { #leafSize (1~100)
            # Multivariable 가능
            'KDE_bandwidth': 1.0, # kernel의 대역폭, float(default: 1.0)
            'KDE_algorithm': 'auto', # 사용할 tree 알고리즘, {‘kd_tree’,‘ball_tree’,‘auto’}(default: ’auto’) 중 택 1
            'KDE_kernel': 'gaussian', # kernel 종류, {'gaussian’, ‘tophat’, ‘epanechnikov’, ‘exponential’, ‘linear’, ‘cosine’}(default: ’gaussian’) 중 택 1
            'KDE_metric': 'euclidean', # 사용할 거리 척도, str(default: ’euclidean’)
            'KDE_breadth_first': True, # breadth(너비) / depth(깊이) 중 우선순위 방식 정의, bool, True: breadth or False: depth
            'KDE_leaf_size': leaf_size} # tree 알고리즘에서의 leaf node 개수, int(default: 40)}
        
    elif algorithm == 'LOF':
        percentile = param['percentile'] # 1~100
        neighbors = param['neighbors']
        result ={ # Neighbors (1~100) 
            # percentile만 조절
            'LOF_neighbors': neighbors, # 가까운 이웃 개수, int(default: 20)
            'LOF_algorithm': 'auto', # 가까운 이웃을 정의하기 위한 알고리즘, {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}(default: ’auto’) 중 택 1
            'LOF_leaf_size': 30, # tree 알고리즘에서의 leaf node 개수, int(default: 30)
            'LOF_metric': 'minkowski', # 이웃을 정의하기 위한 거리 척도, str or callable(default: ’minkowski’)
            'LOF_contamination': (100-percentile)/100 # 오염 정도 (default: 0.2) (0~0.2]
        }
    elif algorithm =='MoG':
        component = param['component']
        result = { 
            'MoG_components': component, # mixture에 활용하는 component의 개수, int(default: 1) #Components(1~100)
            'MoG_covariance': 'full', # {‘full’, ‘tied’, ‘diag’, ‘spherical’}(default: ’full’) 중 택 1
            'MoG_max_iter': 100 # EM 방법론 반복 횟수, int(default: 100)
        }
    elif algorithm =='IQR':
        weight = param['weight']
        result = { 
                  'weight':weight # weight (1~100)
                  }
    elif algorithm == 'SD':
        period = param['period']
        limit = param['limit']
        result = {
            "period":period, 
            "limit":limit # limit (1~100)
            } 

    return result


def get_outlier_detection_param_meta_server(algorithm, algorithmParam):
    Parameter = {
        "IF": {  # Estimators (1~100)
            # ensemble에 활용하는 모델 개수, i(default: 100, 데이터 크기에 적합하게 설정)
            'IF_estimators': algorithmParam,
            # 각 모델에 사용하는 샘플 개수(샘플링 적용), int or float(default: 'auto')
            'IF_max_samples': 'auto',
            # 'auto', # 모델 학습시 활용되는 데이터의 outlier 비율, ‘auto’ or float(default: ’auto’, float인 경우 0 초과, 0.5 이하로 설정)
            'IF_contamination': 'auto',
            # 각 모델에 사용하는 변수 개수(샘플링 적용), int or float(default: 1.0)
            'IF_max_features': 1.0,
            'IF_bootstrap': False},  # bootstrap적용 여부, bool(default: False)
        "KDE": {  # leafSize (1~100)
            # Multivariable 가능
            'KDE_bandwidth': 1.0,  # kernel의 대역폭, float(default: 1.0)
            # 사용할 tree 알고리즘, {‘kd_tree’,‘ball_tree’,‘auto’}(default: ’auto’) 중 택 1
            'KDE_algorithm': 'auto',
            # kernel 종류, {'gaussian’, ‘tophat’, ‘epanechnikov’, ‘exponential’, ‘linear’, ‘cosine’}(default: ’gaussian’) 중 택 1
            'KDE_kernel': 'gaussian',
            'KDE_metric': 'euclidean',  # 사용할 거리 척도, str(default: ’euclidean’)
            # breadth(너비) / depth(깊이) 중 우선순위 방식 정의, bool, True: breadth or False: depth
            'KDE_breadth_first': True,
            'KDE_leaf_size': algorithmParam},  # tree 알고리즘에서의 leaf node 개수, int(default: 40)}
        "LOF": {  # Neighbors (1~100) , leafSize (1~100, Integer)
            'LOF_neighbors': algorithmParam,  # 가까운 이웃 개수, int(default: 20)
            # 가까운 이웃을 정의하기 위한 알고리즘, {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}(default: ’auto’) 중 택 1
            'LOF_algorithm': 'auto',
            'LOF_leaf_size': 30,  # tree 알고리즘에서의 leaf node 개수, int(default: 30)
            # 이웃을 정의하기 위한 거리 척도, str or callable(default: ’minkowski’)
            'LOF_metric': 'minkowski',
            'LOF_contamination': 0.2  # 오염 정도 (default: 0.2) (0~0.2]
        },
        "MoG": {  # Components(1~100)
            # mixture에 활용하는 component의 개수, int(default: 1)
            'MoG_components': algorithmParam,
            # {‘full’, ‘tied’, ‘diag’, ‘spherical’}(default: ’full’) 중 택 1
            'MoG_covariance': 'full',
            'MoG_max_iter': 100  # EM 방법론 반복 횟수, int(default: 100)
        },
        "SR": {
            # Multivariable 불가능해 보임
            # less than period, int, 데이터 크기에 적합하게 설정
            'SR_series_window_size': int(algorithmParam/2),
            # as same as period, int, 데이터 크기에 적합하게 설정
            'SR_spectral_window_size': algorithmParam,
            'SR_score_window_size': algorithmParam * 2},  # a number enough larger than period, int, period보다 충분히 큰 size로 설정
        "IQR": {  # weight (1~100)
            'weight': algorithmParam},
        "SD": {  # limit (1~100)
            "period": algorithmParam, "limit": 15}
    }

    return Parameter[algorithm]
