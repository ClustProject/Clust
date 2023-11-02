import pandas as pd
from Clust.clust.transformation.splitDataByCycle import dataByCycle
import pandas as pd
import numpy as np
####Analysis #########################################################
##추후 analysis는 다른쪽으로 (다른 모듈, 다른 클래스) 빼는게 맞아 보임

class DataAnalysis():
    
    def get_max_correlation_table_with_lag(self, analysis_param, df):
        """
        lag 값을 적용하여 구한 상관 관계중 가장 큰 값 추출

        Args:
            analysis_param (_Dictionary_)
            df (_pd.dataFrame_)

        >>> 'analysis_param': {'feature_key': 'PM10', 'lag_number': '24'}
            
        Returns:
            pd.dataFrame : max_position_correlation_table

        """

        feature_key = analysis_param['feature_key']
        lag_number  = analysis_param['lag_number']
        # feature_key, lag_number
        
        from Clust.clust.tool.stats_table import timelagCorr
        CCT = timelagCorr.TimeLagCorr()
        result = CCT.df_timelag_crosscorr(df, feature_key, lag_number)
        max_position_correlation_table = CCT.get_absmax_index_and_values(result) 
        
        return max_position_correlation_table
   
    def scale_different_x_y_frequency(self, time_scale, data, sampling_flag = True):
        """        
        데이터를 서로 다른 X, y 축 프리컨시로 샘플링한 결과를 생성함

        Args
            data (_pd.Dataframe_) : Input data
            time_scale (_Dictionary_) : The time frequency scale of the x-axis and y-axis
            sampling_flag (_bool_) : 입력 데이터의 빈도와 탐색하고 싶은 데이터의 기준 빈도가 같아서 빈도 다운 샘플링이 필요 없는 경우 False를 입력
            
        Returns   
            pd.dataframe : result                              


        >>> time_scale = {"x_frequency" : {"unit":"H", "num":1}, 
                            "y_frequency" : {"unit":"D", "num":1}}
        """

        x_frequency_unit = time_scale["x_frequency"]['unit']
        x_frequency_num = time_scale["x_frequency"]['num']
        x_frequency = str(x_frequency_num)+x_frequency_unit
        y_frequency_unit = time_scale["y_frequency"]['unit']
        y_frequency_num = time_scale["y_frequency"]['num']
        y_frequency = str(y_frequency_num)+y_frequency_unit
        # down sampling by x_frequency
        if sampling_flag:
            downsampling_freq = x_frequency
        else:
            downsampling_freq = None

        try:
            split_dataset = dataByCycle.getCycleselectDataFrame(data, y_frequency_unit, y_frequency_num, downsampling_freq)
            
            if split_dataset:
                z = []
                for split_data in split_dataset:
                    value = split_data.T.to_numpy().reshape(-1)
                    z.append(value)

                x = [f"{i}_{x_frequency}" for i in range(1,len(split_dataset[0])+1)] 
                y = [f"{i}_{y_frequency}" for i in range(1, len(split_dataset)+1)]
                result = pd.DataFrame(z, index = y, columns = x) 
            else:
                result = pd.DataFrame()
            return result
        
        except ZeroDivisionError:
            print("The duration of the data is less than {}.".format(y_frequency))
