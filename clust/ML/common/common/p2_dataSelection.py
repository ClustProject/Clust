import pandas as pd
import os
# 

## 2. DataSelection
def getSavedIntegratedData(dataSaveMode, dataName, dataFolderPath=None, db_name=None, db_client = None):
    if dataSaveMode =='CSV':
        fileName = os.path.join(dataFolderPath, dataName +'.csv')
        try:
            data = pd.read_csv(fileName, index_col='datetime', infer_datetime_format=True, parse_dates=['datetime'])
        except ValueError:
            data = pd.read_csv(fileName, index_col='Unnamed: 0', infer_datetime_format=True, parse_dates=['Unnamed: 0'])

    elif dataSaveMode =='influx':
        ms_name = dataName
        data = db_client.get_data(db_name, ms_name)
        
    return data
