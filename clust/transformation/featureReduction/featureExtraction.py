import pandas as pd
def featureReduction(data, n_components = 3, method ="TSNE"):
    """ 
    Feature Reduction

    Args:
        data (DataFrame): input data
        n_components (integer): umber of features to be reduced
        method (string ["TSNE", "PCA"]): feature reduction method

    Returns:
        DataFrame: feature_extraction - reduced data result

    Example:
        >>> inputData = data
        >>> n_components = 4
        >>> method = "PCA" #['PCA', 'TSNE']
        >>> inputData = standardScale(inputData)
        >>> FE_data = featureExtraction.featureReduction(inputData, n_components, method)

    """

    colList = MakeColumnName(n_components, method)
    if method == 'TSNE':
        from sklearn.manifold import TSNE
        model = TSNE(n_components=n_components, learning_rate='auto', init='random')
        FEData = model.fit_transform(data)

    if method =='PCA':
        from sklearn.decomposition import PCA
        model = PCA(n_components = n_components)
        FEData = model.fit_transform(data)

    feature_extraction = pd.DataFrame(data = FEData, columns = colList)
    return feature_extraction

def MakeColumnName(num = 3, prefix ="col"):
    """ 
    Make new string list 

    Args:
        num (integer): length of list
        prefix (string): prefix of string

    Returns:
        string array: colList - new string list 

    Example:
        >>> n_components = 3
        >>> method = "PCA" 
        >>> colList = MakeColumnName(n_components, method)
        >>> colList <-['PCA_0', 'PDA_1', 'PCA_2']
    """
    colList =[]
    for i in range(num):
        colList.append(prefix+'_'+str(i))
    return colList