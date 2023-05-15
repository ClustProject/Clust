import sys
sys.path.append("../")
sys.path.append("../..")
sys.path.append("../../..")

def get_data_result(quality_type, data_input, quality_param):

    nan_processing_param = quality_param['nan_processing_param'] 
    if quality_type =='data_with_clean_feature':
        from Clust.clust.quality.NaN import clean_feature_data
        CMS = clean_feature_data.CleanFeatureData()
        result = CMS.get_cleanData_by_removing_column(data_input, nan_processing_param) 
         
    return result

