from Clust.clust.ingestion.DataToCSV import dfToCSV 
import matplotlib.pyplot as plt
import pandas as pd
def make_count_report_byBucket(resultSet, bucket_list, saveFlag = True):
    """
    make value counting report based on each feature and bucket_name
    
    Args:
        resultSet(pd.DataFrame): dataframe index=each bucket_name, column = features value = category number after clustering
        bucket_list(array): bucket_list , select all index including a specific sub name
        saveFlag (bool):save option, if True, save two files title.csv (count info DF), and title_p.csv (percent info DF)
        plotFlag (bool):plot count info DF (bar type)
        
    Returns:
        result_count(dict): dictionary: key <- feature, value <- pd.DataFrame, counting number information by feature and bucket_name
        result_percent(dict) dictionary: key <- feature, value <- pd.DataFrame, percent information, transformedfrom result_count 
    """
    result_count ={}
    result_percent ={}   
        
    feature_list = list(resultSet.columns)
    for feature_name in feature_list:
        #make dictionary by every feature
        count_DF = pd.DataFrame()
        for bucket_name in bucket_list:
            # make count DF with all bucket_name
            subSet = resultSet[feature_name].filter(like=bucket_name, axis=0)
            subSetCount = subSet.value_counts()
            temp= pd.DataFrame(subSetCount)
            temp.columns=[bucket_name]
            count_DF = pd.concat([count_DF, temp], axis=1)    
            
        # clean data
        count_DF = count_DF.sort_index().fillna(0).astype(int)
        percent_DF = round(count_DF/count_DF.sum()*100, 1).sort_index().fillna(0)
        
        # save data
        if saveFlag == True:
            dfToCSV.save_data(count_DF, feature_name+'.csv')
            dfToCSV.save_data(percent_DF, feature_name+'_p.csv')
        
            
        result_count[feature_name] = count_DF
        result_percent[feature_name] =percent_DF
            
    return result_count, result_percent