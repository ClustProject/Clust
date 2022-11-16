
# clust.preprocessing
> Preprocessing module for one single dataset. 

It includes cleaning, imputation, outlier detection modules.
And It also has dataRemoveByNaN module which remove a part of data according to the NaN status.

## 1. main.py (+ main.ipynb)
> This is the test code full data_preprocessing pipeline.
> Input can be both file or data from inlxlufDB 
> If you want to change db and measurement name or add another data ingestion methods, modify data_manager/multipleSourceIngestion.py

## 2. data_preprocessing.py
### 2-1. function allPartialProcessing(input_data, refine_param, outlier_param, imputation_param)
> This function gets cleaner data by all possible data preprocessing modules from clust.preprocessing packages.
> Refinging, Outlier Detction, Imputation

### 2-2. function MultipleDatasetallPartialProcessing(multiple_dataset, process_param)
> This function make multiple-Datasets through All preprocessing method.

### 2-2. DataPreprocessing (class)
> This class provdies several preprocessing Method from this package.

- So far, data refining, outlier removal, imputation module are available.
- There is a plan to expand more preprocessing modules.

### 2-2-1. get_refinedData(self, data, refine_param)
- input: data, refine_param
```json
         refine_param = {
        "removeDuplication":{"flag":True},
        "staticFrequency":{"flag":True, "frequency":None}
    }
```
1) clust.preprocessing.data_cleaning.RefineData.RemoveDuplicateData: Remove duplicated data
2) clust.preprocessing.data_cleaning.RefineData.make_staticFrequencyData: Let the original data have a static frequency
- output: datafrmae type

### 2-2-2. get_errorToNaNData(self, data, outlier_param)
- errorToNaN.errorToNaN:Let outliered data be.
```json
    outlier_param  = {
        "certainErrorToNaN":{"flag":True},
        "unCertainErrorToNaN":{
            "flag":True,
            "param":{"neighbor":0.5}
        },
        "data_type":"air"
    }
```

### 2-2-3. get_imputedData(self, data, impuation_param)
- Replace missing data with substituted values according to the imputation parameter.
```json
     imputation_param = {
        "serialImputation":{
            "flag":True,
            "imputation_method":[{"min":0,"max":10,"method":"linear" , "parameter":{}},{"min":11,"max":20,"method":"mean" , "parameter":{}}],
            "totalNonNanRatio":80
        }
    }
```
