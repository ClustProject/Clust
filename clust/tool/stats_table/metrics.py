from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

def get_RMSE_with_original_and_processedData(original,  processed, partial_number, feature_name):
    raw_value = original[partial_number][feature_name].values
    df = pd.DataFrame(data = {'raw':raw_value})
    for i, raw_clean in enumerate(processed):
        value = processed[i][partial_number][feature_name].values
        df['processed_'+str(i)] =  value
        
    df = df.dropna()
    df_len = len(df.columns)
    for i in range(df_len-1): 
        a = df.raw
        b = df['processed_'+str(i)]
        print("RMSE of raw & processed_",i,":", mean_squared_error(a, b, squared=False))
        print("The number of different values ",i,":", np.sum(a != b), "(Total number: ", len(a),")")



def calculate_metrics_df(df):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {'mae' : mean_absolute_error(df.value, df.prediction),
            'rmse' : mean_squared_error(df.value, df.prediction) ** 0.5,
            'r2' : r2_score(df.value, df.prediction),
           'mape' : np.mean(np.abs((df.value-df.prediction)/df.value))*100 }