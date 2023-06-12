import os

def save_csv_data(data_folder_path, data_name, data):
    if not os.path.exists(data_folder_path):
        os.makedirs(data_folder_path)

    file_name = os.path.join(data_folder_path, data_name + '.csv')
    data.to_csv(file_name)
    return file_name


def save_influx_data(db_name, data_name, data, db_client):
    bk_name = db_name
    ms_name = data_name
    db_client.write_db(bk_name, ms_name, data)
    
    
