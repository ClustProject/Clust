import pandas as pd

def processing_for_clustering(data, ewm_parameter=0.3):

    data = data.ewm(com=ewm_parameter).mean()
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=[data.columns], index = data.index)      
    
    return data