import os

def save_data(df, file_name):
    """
    - delete old file
    - save DF into CSV file
    Args:
         df(pd.DataFrame): input DF
         file_name(string): file name to be saved
        
    """
    if os.path.exists(file_name):
        os.remove(file_name)
        print("The file: {} is deleted!".format(file_name))
    else:
        print("The file: {} does not exist!".format(file_name))
        
    with open(file_name, 'w') as f:
        df.to_csv(f,  index=True, header=True)