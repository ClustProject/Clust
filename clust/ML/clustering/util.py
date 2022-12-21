def count_label_info(labels):
    """
    count numbers by label
    
    Args:
        labels (numpy.array): label data
    Returns:
        count (list of tuple) : number of label
        example> [(0, 93), (1, 24), (2, 35)]
    
    """
    import collections 
    count = sorted(collections.Counter(labels).items())
    return count

