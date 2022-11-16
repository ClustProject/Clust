import pandas as pd 
import numpy as np
from scipy.sparse import data

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

class BasicImputation():
    """ This class supports basic imputation methods.
    """
    def __init__(self, data, method, max, parameter):
        """ Set data, imputation method, max imputation limit value, imputation parameter 
        """
        self.method = method
        self.data = data
        self.max = max
        self.columns = data.columns
        self.index = data.index
        self.parameter = parameter

    def makeDF(self, series_result):
        dfResult = pd.DataFrame(series_result, columns = self.columns, index = self.index)
        return dfResult

    def ScikitLearnMethod(self):
        """ Get imputed data from scikit library methods. (KNN, MICE)
        """
        data = self.data
        # TODO Extend parameter
        if self.method =='KNN':
            n_neighbors = self.parameter['n_neighbors']
            weights =self.parameter['weights']
            metric = self.parameter['metric']
            # https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html
            # n_neighbors # 대체에 참고하기 위한 이웃 개수, int(default: 5)
            # weights: 예측하는 과정에서 이웃에 부여할 가중치 여부, {‘uniform’, ‘distance’} or callable(default: ’uniform’)
            # metric: 이웃을 정의하기 위한 거리 척도, {‘nan_euclidean’} or callable(default: ’nan_euclidean’)
            series_result = KNNImputer(n_neighbors=n_neighbors, weights = weights, metric = metric).fit_transform(data)
            
        elif self.method =='MICE':
            #{‘mean’, ‘median’, ‘most_frequent’, ‘constant’}, default=’mean’
            # https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn-impute-iterativeimputer
            series_result = IterativeImputer(random_state=0, initial_strategy='mean', sample_posterior=True).fit_transform(data)
        
        else:
            series_result = data
            
        result = self.makeDF(series_result)
        return result

    def simpleMethod(self):
        """ Get imputed data from scikit SimpleImputer methods
        """
        series_result = SimpleImputer(strategy=self.method, missing_values = np.nan).fit_transform(self.data)
        result = self.makeDF(series_result)
        return result

    def fillNAMethod(self):
        """ Get imputed data from fillNA methods
        """
        result = self.data.fillna(method=self.method, limit=self.max)
        return result

    def simpleIntMethod(self):
        """ Get imputed data from simple other methods
        """
        result = self.data.interpolate(method=self.method, limit = self.max, limit_direction='both')
        return result

    def orderIntMethod(self):
        """ Get imputed data from interpolation methods
        """
        result = self.data.interpolate(method=self.method, limit = self.max, order = 2, limit_direction='both')
        return result

