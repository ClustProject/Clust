
###
def get_dict_from_two_array(input_key, input_value):
    """make dictionary from two array (key array, value array)

    Args:
        input_key (array): input array for key
        input_value (array): input array for value

    Returns:
        dict_result(dict): dictionary type result -> key: input_key, value: input_value
    """
    dict_result = dict(zip(input_key, map(str, input_value)))
    return dict_result


# data processing
def count_label_info(labels):
    """
    count numbers by label
    
    Args:
        labels (numpy.array): label data

    Returns:
        count (list of tuple) : number of label

    example > [(0, 93), (1, 24), (2, 35)]
    
    """
    import collections 
    count = sorted(collections.Counter(labels).items())
    return count

