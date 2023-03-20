import sys
sys.path.append("../")
sys.path.append("../../")


def get_data_result(split_type, data_input, split_param=None):
    """
    Split Data Set by split_type and parameter

    Args:
        split_type (string):split type 
        data(dataframe, dictionary)
        param: split parameter by split_type
    
    Returns:
        dictionary: Return splited dictionary dataset with new key
    """
    
    if isinstance(data_input, dict):
        result = split_data_from_dataset(split_type, data_input, split_param)
    elif isinstance(data_input, pd.DataFrame):
        result = split_data_from_dataframe(split_type, data_input, split_param)
    return result

def split_data_from_dataset(split_type, dataset, split_param):
    """
    Split Data Set according to split_type and parameter

    Args:
        dataset (Dictionary): dataSet, dictionary of dataframe (ms data). A dataset has measurements as keys and dataframes(Timeseries data) as values.
    
    Returns:
        dictionary: Return value has measurements as keys and split result as values. 
                    split result composed of dataframes divided according to each condition
    """
    split_dataset = {}
    for i, data_name in enumerate(dataset):
        data = dataset[data_name]

        if not(data.empty):
            split_data_dict = split_data_from_dataframe(split_type, data, split_param)
            if split_data_dict:
                for condition_name in split_data_dict:
                    split_dataset[data_name + "/" + condition_name] = split_data_dict[condition_name]
                
    return split_dataset

# TODO Define
def split_data_from_dataframe(split_type, data, split_param):
    """
    Split Data Set according to split_type and parameter

    Args:
        dataset (Dictionary): dataSet, dictionary of dataframe (ms data). A dataset has measurements as keys and dataframes(Timeseries data) as values.
    
    Returns:
        dictionary: Return value has measurements as keys and split result as values. 
                    split result composed of dataframes divided according to each condition
                    
    split_param example:
        1. split_type: cycle
        >>> split_param = {'feature_cycle': 'Day', 'feature_cycle_times': 1}
        2. split_type: working
        >>> split_param = {'workingtime_criteria': {'step': [0, 8, 18, 24], 'label': ['notworking', 'working', 'notworking']}}
        3. split_type: holiday
        >>> split_param = {}
        4. split_type: timestep 
        >>> split_param = {'step': [0, 6, 12, 17, 20, 24], 'label': ['dawn', 'morning', 'afternoon', 'evening', 'night']}
    """
    
    if split_type=='holiday':
        from Clust.clust.transformation.splitDataByCondition import holiday
        result = holiday.split_data_by_holiday(data)
    elif split_type =='working':
        from Clust.clust.transformation.splitDataByCondition import working
        workingtime_criteria = split_param['workingtime_criteria']
        result = working.split_data_by_working(data, workingtime_criteria)
    elif split_type =='timestep':
        from Clust.clust.transformation.splitDataByCondition import timeStep
        result = timeStep.split_data_by_timestep(data, split_param)
    elif split_type =='cycle':
        from Clust.clust.transformation.splitDataByCycle import dataByCycle
        feature_cycle = split_param['feature_cycle']
        feature_cycle_times = split_param['feature_cycle_times']
        result = dataByCycle.getCycleSelectDataSet(data, feature_cycle, feature_cycle_times)
    return result




