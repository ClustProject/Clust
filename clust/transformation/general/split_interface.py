import sys
sys.path.append("../")
sys.path.append("../../")


def get_data_result(split_type, data, split_param=None):
    """
    Split Data Set by split_type and parameter

    Args:
        split_type (string):split type 
        data(dataframe, dictionary)
        param: split parameter by split_type
    
    Returns:
        dictionary: Return value has measurements as keys and split result as values. 
                    split result composed of dataframes divided according to each label of holiday and notholiday.
    """
         
    if split_type=='holiday':
        from Clust.clust.transformation.splitDataByCondition import holiday
        result = holiday.split_data_by_holiday(data)
    elif split_type =='working':
        # split_param = {'workingtime_criteria': {'step': [0, 8, 18, 24], 'label': ['notworking', 'working', 'notworking']}}
        from Clust.clust.transformation.splitDataByCondition import working
        workingtime_criteria = split_param['workingtime_criteria']
        result = working.split_data_by_working(data, workingtime_criteria)
    elif split_type =='timestep':
        from Clust.clust.transformation.splitDataByCondition import timeStep
        result = timeStep.split_data_by_timestep(data, split_param)
    elif split_type =='cycle':
        # split_param = {'feature_cycle': 'Day', 'feature_cycle_times': 1}
        from Clust.clust.transformation.splitDataByCycle import dataByCycle
        feature_cycle = split_param['feature_cycle']
        feature_cycle_times = split_param['feature_cycle_times']
        result = dataByCycle.getCycleSelectDataSet(data, feature_cycle, feature_cycle_times)
        
    
    return result



