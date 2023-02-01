import pandas as pd
import os
# 

## 2. DataSelection
def getSavedIntegratedData(data_save_mode, data_name, data_folder_path=None, db_name=None, db_client = None):
    if data_save_mode =='CSV':
        file_name = os.path.join(data_folder_path, data_name +'.csv')
        try:
            data = pd.read_csv(file_name, index_col='datetime', infer_datetime_format=True, parse_dates=['datetime'])
        except ValueError:
            data = pd.read_csv(file_name, index_col='Unnamed: 0', infer_datetime_format=True, parse_dates=['Unnamed: 0'])

    elif data_save_mode =='influx':
        ms_name = data_name
        data = db_client.get_data(db_name, ms_name)
        
    return data
