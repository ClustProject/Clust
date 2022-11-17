```bash

├── clust
│   ├── README.md
│   ├── data
│   │   └── ingestion
│   │       ├── __init__.py
│   │       ├── __pycache__
│   │       │   └── __init__.cpython-38.pyc
│   │       ├── influx
│   │       │   ├── __pycache__
│   │       │   │   └── influx_Client_v2.cpython-38.pyc
│   │       │   ├── config.ini
│   │       │   ├── influx_Client.py
│   │       │   ├── influx_Client_v2.py
│   │       │   └── influx_Module.py
│   │       └── mongo
│   ├── integration
│   │   ├── ML
│   │   │   ├── RNNAEAlignment.py
│   │   │   ├── RNN_AE
│   │   │   │   ├── model.py
│   │   │   │   └── train_model.py
│   │   │   ├── Test.ipynb
│   │   │   └── __init__.py
│   │   ├── clustDataIntegration.py
│   │   ├── meta
│   │   │   ├── __init__.py
│   │   │   ├── data_integration.py
│   │   │   └── partialDataInfo.py
│   │   └── utils
│   │       ├── __init__.py
│   │       └── param.py
│   ├── preprocessing
│   │   ├── LICENSE
│   │   ├── __pycache__
│   │   │   └── data_preprocessing.cpython-38.pyc
│   │   ├── dataPreprocessing.py
│   │   ├── data_refine
│   │   │   └── __pycache__
│   │   │       ├── __init__.cpython-38.pyc
│   │   │       ├── frequency.cpython-38.pyc
│   │   │       └── redundancy.cpython-38.pyc
│   │   ├── errorDetection
│   │   │   ├── README.md
│   │   │   ├── certainError.py
│   │   │   ├── dataOutlier copy.py
│   │   │   ├── dataOutlier.py
│   │   │   ├── dataRangeInfo_manager.py
│   │   │   ├── errorToNaN.py
│   │   │   └── unCertainError.py
│   │   ├── imputation
│   │   │   ├── DLMethod.py
│   │   │   ├── Imputation.py
│   │   │   ├── __init__.py
│   │   │   ├── basicMethod.py
│   │   │   └── nanMasking.py
│   │   ├── refinement
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   └── frequency.cpython-38.pyc
│   │   │   ├── frequency.py
│   │   │   └── redundancy.py
│   │   └── sampleData
│   │       ├── data_miss_original.csv
│   │       ├── imputated_data.csv
│   │       ├── original_data.csv
│   │       ├── result.csv
│   │       ├── test_imputed_data.csv
│   │       └── test_original_data.csv
│   ├── quality
│   │   ├── NaN
│   │   │   ├── __init__.py
│   │   │   ├── clean_feature_data.py
│   │   │   └── data_remove_byNaN.py
│   │   ├── README.md
│   │   └── cycle
│   │       ├── __init__.py
│   │       ├── __pycache__
│   │       │   ├── __init__.cpython-38.pyc
│   │       │   ├── cycleData.cpython-38.pyc
│   │       │   └── cycle_Module.cpython-38.pyc
│   │       ├── cycleData.py
│   │       └── cycle_Module.py
│   ├── tool
│   │   ├── plot_graph
│   │   │   ├── __init__.py
│   │   │   ├── plot_TwoData.py
│   │   │   ├── plot_correlation.py
│   │   │   └── plot_features.py
│   │   └── stats_table
│   │       ├── __init__.py
│   │       ├── corr_table.py
│   │       └── metrics.py
│   └── transformation
│       ├── entropy
│       │   ├── LICENSE
│       │   ├── README.md
│       │   ├── entropy
│       │   │   ├── DisEn_NCDF.py
│       │   │   ├── DisEn_NCDF_ms.py
│       │   │   ├── MCRDE
│       │   │   │   ├── MCRDE.py
│       │   │   │   └── cumulativeFunc.py
│       │   │   ├── MDE
│       │   │   │   └── MDE.py
│       │   │   └── Multi.py
│       │   ├── main.py
│       │   ├── requirement.txt
│       │   ├── results
│       │   │   ├── avg_mcrde_chf.txt
│       │   │   ├── avg_mcrde_healthy.txt
│       │   │   ├── std_mcrde_chf.txt
│       │   │   └── std_mcrde_healthy.txt
│       │   ├── sample_data
│       │   │   ├── RRIs_AF_1000.mat
│       │   │   ├── RRIs_CHF_1000.mat
│       │   │   └── RRIs_HEALTHY_1000.mat
│       │   ├── test_keti.ipynb
│       │   ├── test_kw.ipynb
│       │   ├── test_old.ipynb
│       │   └── utils
│       │       ├── plot_Entropy.py
│       │       └── write_txt.py
│       ├── featureExtension
│       │   ├── __init__.py
│       │   ├── encodedFeatureExtension.py
│       │   ├── feature_extension_old.py
│       │   ├── periodicFeatureExtension.py
│       │   ├── timeFeatureExtension.py
│       │   └── timeLagFeatureExtension.py
│       ├── featureReduction
│       │   ├── __init__.py
│       │   └── featureExtraction.py
│       ├── general
│       │   ├── __init__.py
│       │   ├── basicTransform.py
│       │   ├── dataScaler.py
│       │   └── data_scaling.py
│       ├── purpose
│       │   ├── __init__.py
│       │   ├── machineLearning.py
│       │   ├── trans_for_LSTMLearning.py
│       │   └── transformForDataSplit.py
│       ├── sampling
│       │   ├── __init__.py
│       │   └── data_up_down.py
│       └── type
│           ├── DFToNPArray.py
│           └── NPArrayToDF.py


```
        

