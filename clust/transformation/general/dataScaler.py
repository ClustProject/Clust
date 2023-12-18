import pandas as pd
import os
import joblib
import json
# 2022 New Code

class DataScaler():
    def __init__(self, scaling_method, rootPath):
        """
        This class generates a scaler and transforms the data. 
        - All information should be described in [rootPath]/scaler_list.json. Before use this class, you should make the empty json file.
        - Initially, only {} should be written in the json file.
        --- setNewScaler(data) function: it makes a new Scaler, even if there is previous scaler. And
        Checks whether the scaler file is already saved, and if it exists, it is loaded and used. 
        If it does not exist, a new scaler is created based on the input data and saved .

        The scaler can be set to scale only a limited columns. (use self.setScaleColumns)
        Unless otherwise specified, scalers are created for scalable numeric data.

        
        Example:
            >>> make new scaler and scale data
            ... 
            >>> from clust.transformation.general.dataScaler import DataScaler
            >>> scalerRootpath = = os.path.join('/home','keti','CLUST_KETI','Clust','KETIAppTestCode','scaler','VIBES')
            >>> DS = DataScaler('minmax', scalerRootpath )
            >>> #feature_col_list = dataScaler.get_scalable_columns(train_o)
            >>> feature_col_list= ['CO/value', 'H2S/value', 'NH3/value',  'O2/value', 'sin_hour']
            >>> DS.setScaleColumns(feature_col_list)
            >>> DS.setNewScaler(trainval_o)
            >>> train = DS.transform(train_o)


        Example:
            >>> load scaler and scale data
            ... 
            >>> from clust.transformation.general.dataScaler import DataScaler
            >>> scalerRootpath = = os.path.join('/home','keti','CLUST_KETI','Clust','KETIAppTestCode','scaler','VIBES')
            >>> DS = DataScaler('minmax', scalerRootpath )
            >>> feature_col_list= ['CO/value', 'H2S/value', 'NH3/value',  'O2/value', 'sin_hour']
            >>> DS.setScaleColumns(feature_col_list)
            >>> DS.loadScaler()
            >>> train = DS.transform(train_o)

        Args:
            scaling_method (one of ['minmax','standard','maxabs','robust']): scaling method 
            rootPath (String): Root path where the scaler will be stored 

        """
        self.scaling_method = scaling_method #
        #self.scale_columns = get_scalable_columns(data)
        self.rootPath = rootPath
        self.scalerListJsonFilePath = os.path.join(self.rootPath, "scaler_list.json")
        

    def _setScalerInfo(self):
        """
        This function set scalerListJsonFilePath and update it. and describes detail information in [rootpath]/scaler_list.json

        Args:
            scaling_method (one of ['minmax','standard','maxabs','robust']): scaling method 
            rootPath (String): Root path where the scaler will be stored 

        """
        scaler_list = self.readJson(self.scalerListJsonFilePath)
        encoded_scaler_list = encode_hash_style(self.scale_columns)
        self.scalerFilePath = os.path.join(self.rootPath, self.scaling_method, encoded_scaler_list, "scaler.pkl")
        return scaler_list, encoded_scaler_list

    def _setNewScalerInfo(self):
        scaler_list, encoded_scaler_list = self._setScalerInfo()

        scaler_list[encoded_scaler_list] = self.scale_columns
        print(self.scale_columns)
        self.writeJson(self.scalerListJsonFilePath, scaler_list)
        
    def setScaleColumns(self, scaleColumns):
        """
        The function can be set to scale only a limited columns. (use self.setScaleColumns)
        Unless otherwise specified, scalers are created for scalable numeric data.

        Args:
            scaleColumns (string list): limited column list to be scaled

        Example:
            >>> scaleColumns=['a','b']
            >>> DS.setScaleColumns(scaleColumns) # can skip
        """
        self.scale_columns = scaleColumns

    def readJson(self, jsonFilePath):
        """
        The function can read json file.  It can be used to find out column list of scaler file.
        
        Example:
            >>> from clust.transformation.general.dataScaler import DataScaler
            >>> scalerRootpath = os.path.join('/Users','scaler')
            >>> DS = DataScaler('minmax',scalerRootpath )
            >>> scaler = DS.setNewScaler(trainval_o)
            >>> df_features = DS.transform(data)
            >>> y = os.path.split(os.path.dirname(DS.scalerFilePath))
            >>> scalerList = DS.readJson(DS.scalerListJsonFilePath)
            >>> scalerList[y[-1]] # print column list of scaler

        Returns:
            json: scaler
        """
        if os.path.isfile(jsonFilePath):
            pass
        else: 
            directory = os.path.dirname(jsonFilePath)
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(jsonFilePath, 'w') as f:
                data={}
                json.dump(data, f, indent=2)
                print("New json file is created from data.json file")

        if os.path.isfile(jsonFilePath):
            with open(jsonFilePath, 'r') as json_file:
                jsonText = json.load(json_file)
        
        return jsonText

    def writeJson(self, jsonFilePath, text):
        if os.path.isfile(jsonFilePath):
            pass
        else: 
            directory = os.path.dirname(jsonFilePath)
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(jsonFilePath, 'w') as f:
                data={}
                json.dump(data, f, indent=2)
                print("New json file is created from data.json file")
                
        with open(jsonFilePath, 'w') as outfile:
            outfile.write(json.dumps(text))

    def transform(self, data):
        """
        The function transform data by scaler

        Args:
            data (dataFrame): input data to be scaled

        Returns:
            DataFrame: transformed Data
        """
        self.data = data
        dataTobeScaled = self.data[self.scale_columns]
        scaledData = self.scaler.transform(dataTobeScaled)
        self.scaledData= pd.DataFrame(scaledData, index =dataTobeScaled.index, columns =dataTobeScaled.columns)
        return self.scaledData

    def setNewScaler(self, dataForScaler):
        """
        The function makes new scaler and saves it

        Args:
            dataForScaler (dataFrame): data to make a new scaler

        Returns:
            scaler: scaler
        """
        self._setNewScalerInfo()
        dataForScaler = dataForScaler[self.scale_columns]

        scaler = self._get_BasicScaler(self.scaling_method) 
        self.scaler = scaler.fit(dataForScaler)
        self.save_scaler(self.scalerFilePath, scaler)
        print("Make New scaler File")
        return self.scaler

    def loadScaler(self):
        
        """
        The function loads scaler. 
        
        Returns:
            scaler: scaler
        """
        self._setScalerInfo()
        if os.path.isfile(self.scalerFilePath):
            self.scaler = joblib.load(self.scalerFilePath)      
            print("Load scaler File")
        else:
            print("No Scaler")
            self.scaler= None

        return self.scaler
        
    def save_scaler(self, scalerFilePath, scaler):
        import os
        dir_name = os.path.dirname(scalerFilePath)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        joblib.dump(scaler, scalerFilePath)
        
    def _get_BasicScaler(self, scaler):
        from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
        scalers = {
            "minmax": MinMaxScaler,
            "standard": StandardScaler,
            "maxabs": MaxAbsScaler,
            "robust": RobustScaler,
        }
        return scalers.get(scaler.lower())()

def get_scalable_columns(data):
    integer_columns = list(data.select_dtypes(include=['int64', 'int32']).columns)
    float_columns = list(data.select_dtypes(include=['float64', 'float32']).columns)
    object_columns = list(data.select_dtypes(include=['object']).columns)
    scale_columns = integer_columns + float_columns
    return scale_columns
    
class DataInverseScaler():
    def __init__(self, scaling_method, rootPath):
        """
        This class makes inverse scaled data.

        Example:
            >>> from clust.transformation.general.dataScaler import DataScaler
            >>> scalerRootpath = os.path.join('/Users','jw_macmini','CLUSTGit','KETIAppMachineLearning','scaler')
            >>> DIS = DataInverseScaler(df_features, 'minmax',scalerRootpath )
            >>> DIS.setScaleColumns(df_features) 
            >>> result = DIS.transform()

        Args:
            scaling_method (one of ['minmax','standard','maxabs','robust']): scaling method 
            rootPath (String): Root path where the scaler will be stored 
        """
        self.scaling_method = scaling_method #
        self.rootPath = rootPath
        
    def setScaleColumns(self, column_list):
        """
        Set scale columns and get scalerFilePath

        Args:
            column_list (list): column_list

        """
        
        self.scale_columns = column_list
        encoded_scaler_list = encode_hash_style(column_list)
        self.scalerFilePath = os.path.join(self.rootPath, self.scaling_method, encoded_scaler_list, "scaler.pkl")
        print(self.scalerFilePath)

    def _setScaler(self):
        """
        The function set scaler. (generation or load based on root_path info, scale columns)
        
        Args:
            data (dataFrame): input data to be inverse-scaled

        Returns: scaler
            scaler: scaler
        """
        if os.path.isfile(self.scalerFilePath):
            self.scaler = joblib.load(self.scalerFilePath)      
            print("Load scaler File")
        else:
            print("No proper scaler")
            self.scaler=None

        return self.scaler
    
    def transform(self, data):
        """
        The function transform data by inverse-scaler

        Args:
            data (dataFrame): input data to be inverse-scaled

        Returns:
            Dataframe: transformed Data
        """
        self.data = data
        self.dataToBeScaled = self.data[self.scale_columns]
        self.scaler = self._setScaler()
        inverseScaledData = self.scaler.inverse_transform(self.dataToBeScaled)
        self.inverseScaledData= pd.DataFrame(inverseScaledData, index =self.dataToBeScaled.index, columns =self.dataToBeScaled.columns) 
        return self.inverseScaledData






def encode_hash_style(text):
    import hashlib
    hash_object = hashlib.md5(str(text).encode('utf-8'))
    hashedText= hash_object.hexdigest()
    return hashedText


def get_scaled_data(data, feature_col_list, scalerRootpath, scale_method):
    """This method makes scaled data.

        This function finds scaler based and scale numpy array.

        Args:
            data (np.array): data array, size = (past_step, len(feature_col_list))
            feature_col_list (list[string]): feature list for scaler
            scalerRootpath (string): rootPath for scaler

        Returns:
            DataFrame: scaledData - scaled Output Data
        
        Note
        ---------
        Original scaler was generated and stored before using this method for feature_col_list features.


    """
    data_df = pd.DataFrame(data, columns=feature_col_list, index=range(len(data)))
    
    DS = DataScaler(scale_method, scalerRootpath)
    DS.setScaleColumns(list(data_df.columns))
    DS.loadScaler()
    scaledData = DS.transform(data_df)
    return scaledData
