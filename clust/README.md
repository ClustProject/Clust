
```bash
clust/
  ├── README.md
  │
  ├── analysis
  │   ├── bucket
  │   ├── statisticAnalyzer
  │   ├── timeAnalyzer
  │   └── bucketReport.py
  │
  ├── data
  │   ├── data_interface - (문/완료)   
  │   ├── df_set_data - (문/완료)   
  │   └── df_data - (문/완료) 
  │
  ├── ingestion
  │   └── influx (진행중)
  │       ├── influx_client.py - (이)정리 필요  
  │       └──  influx_client_v2.py - (이)수정 필요  
  │   ├── mongo (진행중)
  │       └── mongo_client.py 
  │   ├── DatatoCSV
  │   └── CSVtoInflux
  │
  ├── integration
  │   ├── ML
  │   ├── util
  │   └── meta
  │
  ├── meta
  │   ├── metaDataManager
  │       └── bucketMeta.py (소/완료) - (문/검토 완료)
  │   ├── metaFormatCheck
  │   └── metaGenerator
  │
  ├── ML
  │   ├── app
  │       ├── clustering_app1.py : 응용 어플리케이션을 위함
  │   ├── clustering
  │       ├── interface.py :clustering 모듈을 활용하기 위한 인터페이스 
  │       ├── cluetring.py :train, test를 위한 추상화 클래스 기술
  │       ├── kMeans.py    :clustering algorithm 1
  │       ├── som.py       :clustering algorithm 2
  │       └── etc.py       :clustering algorithm 3 (모듈화 어려운)
  │   └── tool
  │       ├── clean.py                  : data clean 관련 모듈
  │       ├── data.py                   : data get, transform 관련 모듈 
  │       ├── meta.py                   : meta data 사용 관련 모듈
  │       ├── model_path_setting.py     : model별 path 위치 관련 정보
  │       ├── model.py                  : model 저장 관련 모듈
  │       ├── scaler.py                 : scaling 관련 모듈
  │       └── util.py                   : ML 데이터/모델 이외 관련 공통 모듈
  │   └── common
  │       │   └── common
  │       │       └── p1_integratedDataSaving.py
  │       ├── ML_pipeline.py            : 
  │       ├── model_path_setting.py     : 
  │       └── tool.py                   : 
  │   └── regression
  │       │   └── regression_model
  │       │      ├── cnn_id_model.py       : custom CNN1D model
  │       │      ├── fc_model.py           : custom FC model
  │       │      ├── lstm_fcns_model.py    : custom LSTMFCNs model
  │       │      └── rnn_model.py          : custom RNN model
  │       │   └── models
  │       │      ├── cnn_1d.py     : standard CNN1D model
  │       │      ├── fc.py         : standard FC model
  │       │      ├── lstm_fcns.py  : standard LSTMFCNs model
  │       │      └── rnn.py        : standard RNN model
  │       ├── interface.py      : abstract class & method
  │       ├── train.py          : train class
  │       ├── test.py           : test class
  │       └── inference.py      : inference class
  │   └── classification
  │       │   └── classification_model
  │       │      ├── cnn_id_model.py       : custom CNN1D model
  │       │      ├── fc_model.py           : custom FC model
  │       │      ├── lstm_fcns_model.py    : custom LSTMFCNs model
  │       │      └── rnn_model.py          : custom RNN model
  │       │   └── models
  │       │      ├── cnn_1d.py     : standard CNN1D model
  │       │      ├── fc.py         : standard FC model
  │       │      ├── lstm_fcns.py  : standard LSTMFCNs model
  │       │      └── rnn.py        : standard RNN model
  │       ├── interface.py      : abstract class & method
  │       ├── train.py          : train class
  │       ├── test.py           : test class
  │       └── inference.py      : inference class
  │
  ├── preprocessing
  │   ├── custom
  │   ├── errorDetection
  │   ├── imputatation
  │   ├── refinement
  │   └── sampleData
  │
  ├── quality
  │   └── NaN
  │
  ├── tool
  │   ├── plot
  │   ├── stats
  │   └── etc   
  │
  └── transformation
      ├── entropy : Entropy를 위한 테스트 코드 (TBD)
      ├── featureExtension :
      ├── featureReduction
      ├── general
      ├── purpose
      ├── sampling
      ├── splitDataByCondition
      ├── splitDataByCycle
      └── type
```

