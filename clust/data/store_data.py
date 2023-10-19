import os

def save_csv_data(data_folder_path, data_name, data):
    """    
        # Description
            - 데이터를 csv 파일로 저장하는 기능

        # Args
            - data_folder_path (_String_) : folder path
            - data_name (_String_) : file name
            - data (_pd.dataFrame_) : data to be saved as CSV

        # Returns
            - file_name (_String_)

    """
    if not os.path.exists(data_folder_path):
        os.makedirs(data_folder_path)

    file_name = os.path.join(data_folder_path, data_name + '.csv')
    data.to_csv(file_name)

    return file_name


def save_influx_data(db_name, data_name, data, db_client):
    """    
        # Description
            - 데이터를 influxDB에 저장하는 기능

        # Args
            - data_folder_path (_String_) : folder path
            - data_name (_String_) : file name
            - data (_pd.dataFrame_) : data to be saved as CSV
            - db_client (_instance_) : influxDB instance    
        
        # Returns
            - No Returns

    """
    bk_name = db_name
    ms_name = data_name
    db_client.write_db(bk_name, ms_name, data)
    
    
