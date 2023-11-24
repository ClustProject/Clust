
```bash
clust/
  ├── README.md
  │
  ├── analysis
  │   ├── bucketReport.py
  │   ├── dataAnalysis.py
  │   ├── dataSetAnalysis.py
  │   ├── statisticAnalyzer
  │   │   └── statistics.py
  │   └── timeAnalyzer
  │       ├── mean_by_holiday.py
  │       ├── mean_by_timeStep.py
  │       └── mean_by_working.py
  │
  ├── data
  │   ├── data_interface.py - (문/완료)   
  │   ├── df_data.py - (문/완료)  
  │   ├── df_set_data.py - (문/완료)  
  │   └── store_data.py - (문/완료) 
  │
  ├── ingestion
  │   ├── influx (진행중)
  │   │   ├── influx_client.py - (이)정리 필요  
  │   │   └── influx_client_v2.py - (이)수정 필요  
  │   ├── mongo (진행중)
  │   │   ├── custom_modules.py
  │   │   └── mongo_client.py 
  │   ├── DatatoCSV
  │   │   └── dfToCSV.py 
  │   └── CSVtoInflux
  │       ├── cleanDataByType.py
  │       └── csvCollector.py 
  │
  ├── integration
  │   ├── integration_by_method.py
  │   ├── integration_interface.py
  │   ├── meta 
  │   │   ├── data_integration.py 
  │   │   └── partialDataInfo.py
  │   ├── utils
  │   │   └── param.py 
  │   ├── DatatoCSV
  │   │   └── dfToCSV.py 
  │   └── ML
  │       ├── RNNAEAlignment.py
  │       └── RNN_AE
  │             ├── model.py 
  │             └── train_model.py 
  │
  ├── meta
  │   ├── ingestion_meta_exploration.py
  │   ├── metaDataManager
  │   │   ├── bucketMeta.py
  │   │   ├── collector.py
  │   │   ├── descriptor.py
  │   │   └── wizMongoDbApi.py
  │   ├── metaFormatCheck
  │   │   └── metaInfo.py
  │   └── metaGenerator
  │       ├── analysisDBMetaGenerator.py
  │       ├── analysisMSMetaGenerator.py
  │       └── fileMetaGenerator.py
  │
  ├── pipeline
  │   ├── data_pipeline.py
  │   └── param.py
  │
  ├── preprocessing
  │   ├── custom
  │   ├── errorDetection
  │   ├── imputatation
  │   ├── refinement
  │   └── sampleData
  │
  ├── quality
  │   ├── quality_interface.py
  │   └── NaN
  │       ├── clean_feature_data.py
  │       └── data_remove_byNaN.py
  │
  ├── tool
  │   ├── file_module
  │   │   └── file_common.py
  │   ├── plot
  │   │   ├── plot_echart.py
  │   │   ├── plot_features.py
  │   │   ├── plot_image.py
  │   │   ├── plot_interface.py
  │   │   ├── plot_plt.py
  │   │   └── plot_two_data.py
  │   └── stats_table
  │       ├── correlation.py
  │       ├── metrics.py
  │       └── timelagCorr.py
  │
  ├── preprocessing
  │   ├── dataPreprocessing.py
  │   ├── processing_interface.py
  │   ├── errorDetection
  │   │   ├── certainError.py
  │   │   ├── dataOutlier.py
  │   │   ├── dataRangeInfo_manager.py
  │   │   ├── errorToNaN.py
  │   │   └── unCertainError.py
  │   ├── imputation
  │   │   ├── DLMethod.py
  │   │   ├── Imputation.py
  │   │   ├── basicMethod.py
  │   │   └── nanMasking.py
  │   ├── refinement
  │   │   ├── frequency.py
  │   │   └── redundancy.py
  │   └── sampleData
  │
  │
  ├── transformation
  │   ├── entropy : Entropy를 위한 테스트 코드 (TBD)
  │   │   ├── main.py
  │   │   ├── entropy
  │   │   │     ├── DisEn_NCDF.py 
  │   │   │     ├── DisEn_NCDF_ms.py
  │   │   │     ├── Multi.py 
  │   │   │     ├── MCRDE
  │   │   │     │   ├── MCRDE.py 
  │   │   │     │   └── cumulativeFunc.py 
  │   │   │     └── MDE
  │   │   │         └── MDE.py 
  │   │   ├── utils
  │   │   │     ├── model.py 
  │   │   │     └── train_model.py 
  │   │   ├── results
  │   │   └── sample_data
  │   ├── featureExtension :
  │   │   ├── encodedFeatureExtension.py
  │   │   ├── feature_extension_old.py
  │   │   ├── periodicFeatureExtension.py
  │   │   ├── timeFeatureExtension.py
  │   │   └── timeLagFeatureExtension.py
  │   ├── featureReduction
  │   │   ├── featureExtraction.py
  │   │   └── pca.py
  │   │── general
  │   │   ├── basicTransform.py
  │   │   ├── dataScaler.py
  │   │   ├── data_scaling.py
  │   │   ├── select_interface.py
  │   │   └── split_interface.py
  │   ├── purpose
  │   │   ├── machineLearning.py
  │   │   ├── trans_for_LSTMLearning.py
  │   │   └── transformForDataSplit.py
  │   ├── sampling
  │   │   └── data_up_down.py
  │   ├── splitDataByCondition
  │   │   ├── holiday.py
  │   │   ├── timeStep.py
  │   │   └── working.py
  │   ├── splitDataByCycle
  │   │   ├── cycleModule_tobedeleted.py
  │   │   └── dataByCycle.py
  │   └── type
  │       ├── DFToNPArray.py
  │       └── NPArrayToDF.py
  │
  └── ML # forecasting 삭제 예상하고 만듬
      ├── clustering
      │   ├── interface.py :clustering 모듈을 활용하기 위한 인터페이스 
      │   ├── cluetring.py :train, test를 위한 추상화 클래스 기술
      │   ├── kMeans.py    :clustering algorithm 1
      │   ├── som.py       :clustering algorithm 2
      │   └── etc.py       :clustering algorithm 3 (모듈화 어려운)
      ├── tool
      │   ├── clean.py                  : data clean 관련 모듈
      │   ├── data.py                   : data get, transform 관련 모듈 
      │   ├── meta.py                   : meta data 사용 관련 모듈
      │   ├── model.py                  : model 저장 관련 모듈
      │   ├── scaler.py                 : scaling 관련 모듈
      │   └── util.py                   : ML 데이터/모델 이외 관련 공통 모듈
      ├── common
      │   ├── ML_api.py
      │   ├── ML_pipeline.py
      │   ├── echart.py
      │   ├── model_parameter_setting.py
      │   ├── model_path_setting.py           
      │   └── common    
      │         └── p1_integratedDataSaving.py                    
      ├── regression
      │   ├── interface.py      : abstract class & method
      │   ├── train.py          : train class
      │   ├── test.py           : test class
      │   ├── inference.py      : abstract class & method
      │   ├── models
      │   │     ├── cnn_1d.py 
      │   │     ├── fc.py 
      │   │     ├── lstm_fcns.py 
      │   │     └── rnn.py 
      │   └── clust_models
      │         ├── cnn1d_clust.py 
      │         ├── fc_clust.py 
      │         ├── lstm_fcns_clust.py 
      │         └── rnn_clust.py 
      └── classification
          ├── interface.py      : abstract class & method
          ├── train.py          : train class
          ├── test.py           : test class
          ├── inference.py      : abstract class & method
          ├── models
          │     ├── cnn_1d.py 
          │     ├── fc.py 
          │     ├── lstm_fcns.py 
          │     └── rnn.py 
          └── classification_model
                ├── cnn_1d_model.py 
                ├── fc_model.py 
                ├── lstm_fcns_model.py 
                └── rnn_model.py 

   
```

