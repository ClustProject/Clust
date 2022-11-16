
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from tqdm import tqdm

import sranodec as anom
from sklearn.ensemble import IsolationForest   
from sklearn.neighbors import KernelDensity, LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from Clust.clust.preprocessing.dataPreprocessing import DataPreprocessing
class DataOutlier():
    """
    
    """
    def __init__(self, raw_data):
        """
        :param raw_data: train data whose shape is (num_index x num_variable)
        :type raw_data: dataframe

        AlgorithmList =[ 'IF', 'KDE', 'LOF', 'MoG', 'SR']
        # 1. Isolation Forest (IF)
        # 2. Kernel density estimation (KDE) 
        # 3. LOF:Local Outlier Factor (LOF)
        # 4. Mixture of Gaussian (MoG)
        # 5. SR: Spectral Residual(SR)

        """
        self.AlgParameter = {
            "IF":{
                'IF_estimators': 100, # ensemble에 활용하는 모델 개수, int(default: 100, 데이터 크기에 적합하게 설정)
                'IF_max_samples': 'auto', # 각 모델에 사용하는 샘플 개수(샘플링 적용), int or float(default: 'auto')
                'IF_contamination': 'auto', # 모델 학습시 활용되는 데이터의 outlier 비율, ‘auto’ or float(default: ’auto’, float인 경우 0 초과, 0.5 이하로 설정)
                'IF_max_features': 1.0, # 각 모델에 사용하는 변수 개수(샘플링 적용), int or float(default: 1.0)
                'IF_bootstrap': False}, # bootstrap적용 여부, bool(default: False)
            "KDE":{
                'KDE_bandwidth': 0.2, # kernel의 대역폭, float(default: 1.0)
                'KDE_algorithm': 'auto', # 사용할 tree 알고리즘, {‘kd_tree’,‘ball_tree’,‘auto’}(default: ’auto’) 중 택 1
                'KDE_kernel': 'gaussian', # kernel 종류, {'gaussian’, ‘tophat’, ‘epanechnikov’, ‘exponential’, ‘linear’, ‘cosine’}(default: ’gaussian’) 중 택 1
                'KDE_metric': 'euclidean', # 사용할 거리 척도, str(default: ’euclidean’)
                'KDE_breadth_first': True, # breadth(너비) / depth(깊이) 중 우선순위 방식 정의, bool, True: breadth or False: depth
                'KDE_leaf_size': 40}, # tree 알고리즘에서의 leaf node 개수, int(default: 40)}
            "LOF":{
                'LOF_neighbors': 5, # 가까운 이웃 개수, int(default: 20)
                'LOF_algorithm': 'auto', # 가까운 이웃을 정의하기 위한 알고리즘, {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}(default: ’auto’) 중 택 1
                'LOF_leaf_size': 30, # tree 알고리즘에서의 leaf node 개수, int(default: 30)
                'LOF_metric': 'minkowski',# 이웃을 정의하기 위한 거리 척도, str or callable(default: ’minkowski’)
                'LOF_contamination':0.1 # 오염 정도
                 }, 
            "MoG": {# EM 방법론 반복 횟수, int(default: 100)
                'MoG_components': 2, # mixture에 활용하는 component의 개수, int(default: 1)
                'MoG_covariance': 'full', # {‘full’, ‘tied’, ‘diag’, ‘spherical’}(default: ’full’) 중 택 1
                'MoG_max_iter': 100},
            "SR":{
                'SR_series_window_size': 24, # less than period, int, 데이터 크기에 적합하게 설정
                'SR_spectral_window_size': 24, # as same as period, int, 데이터 크기에 적합하게 설정
                'SR_score_window_size': 100}# a number enough larger than period, int, period보다 충분히 큰 size로 설정
        }
 
        self.data = raw_data.copy()
        self.imputedData = raw_data.copy()
    
    def refinmentForOutlierDetection(self):
        """    
        - Get refined data (same frequency, no redundency)
        - optional function for not cleaned data

        :return index_list: indices of detected outliers
        :type: json
        """ 
        refine_param = {
            "removeDuplication":{"flag":True},
            "staticFrequency":{"flag":True, "frequency":None}
            }
        self.data = DataPreprocessing().get_refinedData(self.data, refine_param)

        return self.data

    def imputationForOutlierDetection(self, imputation_param =None):
        """    
        1. Get self.originNaNIndex =  NaN data index before imputation
        2. Get imputed data before outlier detection

        :param imputation_param: Parameter for imputation
        :type imputation_param: json

        :return index_list: indices of detected outliers
        :type: json

        '''   example
         imputation_param = {
            "serialImputation":{
                "flag":True,
                "imputation_method":[{"min":0,"max":1000000,"method":"linear", "parameter":{}}
                ],"totalNonNanRatio":0}}
        '''

        """ 
        self.originNaNIndex = getNaNIndex(self.data)

        if imputation_param == None:
            self.imputedData = self.data.fillna(method='ffill')
            self.imputedData = self.imputedData.fillna(0)
        else:
            self.imputedData = DataPreprocessing().get_imputedData(self.data, imputation_param)
        
        return self.imputedData

    ### Outlier Detector
    def getOneDetectionResult(self, data, config):
        """    
        :param data: data for outlier Detetcion
        :type data: dataFrame

        :param config: config 
        :type config: dictionary

        :param percentile: # 예측시 활용되는 outlier 임계값
        :type percentile: integer/float

        :return index_list: indices of detected outliers
        :type: json

        example
            >>> config = { 
                    'algorithm': 'IF', # outlier detection에 활용할 알고리즘 정의, {'SR', 'LOF', 'MoG', 'KDE', 'IF'} 중 택 1            
                    'alg_parameter': AlgParameter['IF']      # option
                }
            >>> data_outlier = mod.DataOutlier(raw_data)
            >>> replaced_data, index_list = data_outlier.getOneDetectionResult(config, 95)

        # Output Example
        ``` json
        {'in_co2': array([  324,  1229,  1230, ..., 50274, 50275, 50276])}
        ```
        """
        self.percentile = config['percentile']
        self.algorithm = config['algorithm']
        if 'alg_parameter'in config:
            self.args = config['alg_parameter']
        else:
            self.args = self.AlgParameter[self.algorithm]

        self.columns_list = list(data.columns)
        index_list={}
        for col in tqdm(data.columns):
            if self.algorithm == "SR":
                data_col = data[col].values
            else:
                data_col = data[col].values.reshape(-1, 1)
            self.model = self.getModel(data_col)
            indexes = self.getIndexList(data_col, self.model)
            indexes = data[col].iloc[indexes].index
            index_list[col] = indexes

        return index_list
    
    def getModel(self, data_col):
        """
        :param data_col: data for each column
        :type: np.array
        
        :return model: fitted model of selected outlier detection algorithm
        :type: model
        
        """
        if self.algorithm == 'SR':
            model = anom.Silency(self.args['SR_spectral_window_size'], self.args['SR_series_window_size'],
                                 self.args['SR_score_window_size'])
        elif self.algorithm == 'LOF':
            model = LocalOutlierFactor(n_neighbors=self.args['LOF_neighbors'], novelty=True, 
                                       algorithm=self.args['LOF_algorithm'], leaf_size=self.args['LOF_leaf_size'], 
                                       metric=self.args['LOF_metric']).fit(data_col)
        elif self.algorithm == 'MoG':
            model =  GaussianMixture(n_components=self.args['MoG_components'], covariance_type=self.args['MoG_covariance'], 
                                     max_iter=self.args['MoG_max_iter'], random_state=0).fit(data_col)
        elif self.algorithm == 'KDE':
            model = KernelDensity(kernel=self.args['KDE_kernel'], bandwidth=self.args['KDE_bandwidth'], 
                                  algorithm=self.args['KDE_algorithm'], metric=self.args['KDE_metric'], 
                                  breadth_first=self.args['KDE_breadth_first'], 
                                  leaf_size=self.args['KDE_leaf_size']).fit(data_col)
        elif self.algorithm == 'IF':
            model = IsolationForest(n_estimators=self.args['IF_estimators'], max_samples=self.args['IF_max_samples'], 
                                    contamination=self.args['IF_contamination'], max_features=self.args['IF_max_features'], 
                                    bootstrap=self.args['IF_bootstrap']).fit(data_col)
        return model
    
    def getIndexList(self, data_col, model):
        """
        :param data_col: data for each column
        :type: np.array
        
        :param model: fitted model of selected outlier detection algorithm
        :type: model
        
        :return indexes: indices of detected outliers
        :type: list
        
        """
        if self.algorithm == 'SR':
            score = model.generate_anomaly_score(data_col)

        else:
            score = - 1.0 * model.score_samples(data_col)
     
        indexes = np.where(score > np.percentile(score, self.percentile))[0]
        return indexes
    
def showResult(data, result, outlierIndex):
    """
    :param data: data for each column
    :type: np.array
    
    :param result: fitted model of selected outlier detection algorithm
    :type: model

    :param outlierIndex: fitted model of selected outlier detection algorithm
    :type: model

    """
    import matplotlib.pyplot as plt
    for feature in data.columns:
        print(feature)
        index_list_column = outlierIndex[feature]
        plt.plot(data[feature])
        plt.plot(index_list_column, data[feature].loc[index_list_column].values, "x")
        plt.show()
        plt.plot(result[feature])
        plt.show()

def getMoreNaNDataByNaNIndex(data, NaNIndex):
    """
    :param data_col: data
    :type: dataFrame

    :param NaNIndex: NaNIndex
    :type: dictionary
    
    :return NaNData: data with NaN according to the NaNIndex
    :type: dataFrame
    """
    NaNData = data.copy()
    for column in data.columns:
        if column in NaNIndex.keys():
            indexes = NaNIndex[column]
            NaNData[column].loc[indexes] =np.nan 
    return NaNData

def getNaNIndex(data):
    """
    :param data_col: data
    :type: dataFrame
    
    :return NaNIndex: NaN Index of data
    :type: dictionary
    """

    NaNIndex={}
    for column in data.columns:
        NaNIndex[column]= data[data[column].isna()].index
    return NaNIndex