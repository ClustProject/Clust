from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(14, 4)
class DataRemoveByNaNStatus():
    
    def __init__(self):
        pass
    ## Select one remove method : (time, ratio, num)
    def removeNaNData(self, data, NanInfoForCleanData):
        type = NanInfoForCleanData['type']
        # 0,0 일 경우 NaN 이 하나 이상일 경우 모두 삭제
        if type=='time':
            #ConsecutiveNanLimitTime_second =   60 *60
            #totalNaNLimitTime_second =  1 * 60
            ConsecutiveNanLimitTime_second = NanInfoForCleanData['ConsecutiveNanLimit']
            totalNaNLimitTime_second =NanInfoForCleanData['totalNaNLimit']
            data = self.removeNaNDataByTime(data, ConsecutiveNanLimitTime_second, totalNaNLimitTime_second)
            
        elif type =='num':
            ConsecutiveNanLimitNum = NanInfoForCleanData['ConsecutiveNanLimit']
            totalNanLimitNum =NanInfoForCleanData['totalNaNLimit']
            data = self.removeNaNDataByNum(data, ConsecutiveNanLimitNum, totalNanLimitNum)
            
        elif type=='ratio':
            ConsecutiveNanLimitNum = NanInfoForCleanData['ConsecutiveNanLimit']
            totalNanLimitNum =NanInfoForCleanData['totalNaNLimit']
            data = self.removeNaNDataByRatio(data, ConsecutiveNanLimitNum, totalNanLimitNum)    
        return data
    
    def removeNaNDataByTime(self, data, ConsecutiveNanLimitTime_second, totalNaNLimitTime_second):
        if len(data)>0:
            totalNanLimitTime = timedelta(seconds = totalNaNLimitTime_second)
            frequency = data.index[-1]- data.index[-2]
            totalNaNLimitNum = int(totalNanLimitTime/frequency)
            data = self.removeNaNDataByTotalNaNLimitNum(totalNaNLimitNum, data)
            
            if len(data.columns)>0:
                ConsecutiveNanLimitTime = timedelta(seconds = ConsecutiveNanLimitTime_second)
                frequency = data.index[-1]- data.index[-2]
                ConsecutiveNanLimitNum = int(ConsecutiveNanLimitTime/frequency)
                data = self.removeNaNDataByConsecutiveNaNLimitNum(ConsecutiveNanLimitNum, data)
                
        return data
    
    def removeNaNDataByNum(self, data, ConsecutiveNanLimitNum, totalNanLimitNum):
        if len(data)>0:
            data = self.removeNaNDataByTotalNaNLimitNum(totalNanLimitNum, data)
            if len(data.columns)>0:
                data = self.removeNaNDataByConsecutiveNaNLimitNum(ConsecutiveNanLimitNum, data)            
        return data
    
    def removeNaNDataByRatio(self, data, ConsecutiveNanLimitRatio, totalNanLimitRatio):
        if len(data)>0:
            data_length = len(data)
            totalNanLimitNum = int(data_length * totalNanLimitRatio)
            data = self.removeNaNDataByTotalNaNLimitNum(totalNanLimitNum, data)
            if len(data.columns)>0:
                ConsecutiveNanLimitNum = int(data_length)  * ConsecutiveNanLimitRatio
                data = self.removeNaNDataByConsecutiveNaNLimitNum(ConsecutiveNanLimitNum, data)
        return data
 
    def consecutiveNaNCountMap(self, data):
        consecutiveNanCountMap = pd.DataFrame() 
        column_list = data.columns
        for column_name in column_list:
            consecutiveNanCountMap[column_name] = data[column_name].isnull().astype(int).groupby(
                data[column_name].notnull().astype(int).cumsum()).cumsum()
        return consecutiveNanCountMap
    
    def removeNaNDataByTotalNaNLimitNum(self, totalNanLimitNum, data):
        columnNaNCountSet = data.isnull().sum()
        for column_name in data.columns:
            columnNaNCount = columnNaNCountSet[column_name]
            if totalNanLimitNum < columnNaNCount:
                data = data.drop(column_name,axis=1)
        return data
    
    def removeNaNDataByConsecutiveNaNLimitNum(self, ConsecutiveNanLimitNum, data):
        self.consecutiveNanCountMap = self.consecutiveNaNCountMap(data)
        #plt.show()
        for column_name in data.columns:
            consecutiveNanCount_column = self.consecutiveNanCountMap[column_name]
            if (consecutiveNanCount_column > ConsecutiveNanLimitNum).any():
                data = data.drop(column_name,axis=1)
         
        return data