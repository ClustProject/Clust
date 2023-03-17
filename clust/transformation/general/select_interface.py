import sys
sys.path.append("../")
sys.path.append("../../")


def get_data_result(select_type, data_input, select_param=None):
    """
    select data by select_type and parameter

    Args:
        select_type (string):select type 
        data (dictionary)
        param: select parameter
    
    Returns:
        dictionary: Return selected dictionary dataset
    """
    
    if select_type =='keyword':
        keyword = select_param['keyword']
        old_keys = list(set(data_input.keys()))
        new_keys = [e for e in old_keys if keyword in e]
        result = dict((k, data_input[k]) for k in new_keys if k in data_input)

    return result
