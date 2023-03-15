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
    """
    
    if split_type =='keyword':
        keyword = split_param['keyword']
        result = select_only_specific_pair_inucluding_keyword(data, keyword)
    return result

def select_only_specific_pair_inucluding_keyword(data_dict, keyword):
    old_keys = list(set(data_dict.keys()))
    new_keys = [e for e in old_keys if '/working' in e]

    result = dict((k, data_dict[k]) for k in new_keys if k in data_dict)
    
    return result




