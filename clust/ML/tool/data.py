def DF_to_series(data):
    """make input data for clustering. 
    Args:
        data(np.dataFrame): input data
    Return:
        series_data(series): transformed data for training, and prediction

    """
    series_data = data.to_numpy().transpose()
    return series_data


