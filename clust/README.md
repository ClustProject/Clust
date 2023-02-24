
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
  │       └── mongoClient.py 
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
  │       ├── usecase.py : 응용 어플리케이션을 위함
  │   ├── clustering
  │       ├── interface.py :clustering 모듈을 활용하기 위한 인터페이스 
  │       ├── cluetring.py :train, test를 위한 추상화 클래스 기술
  │       ├── kMeans.py    :clustering algorithm 1
  │       ├── som.py       :clustering algorithm 2
  │       └── etc.py       :clustering algorithm 3 (모듈화 어려운)
  │   └── tool
  │       ├── data.py      : ML 인풋 데이터 처리 관련 모듈
  │       ├── model.py     : ML 모델 입출력 관련 모듈
  │       └── util.py      : ML 데이터/모델 이외 관련 공통 모듈
  │   └── common
  │       │   └── common
  │       │       ├── p1_integratedDataSaving.py
  │       │       ├── p2_dataSelection.py
  │       │       ├── p3_training.py
  │       │       └── p4_testing.py
  │       ├── trainer.py            : train abstract class
  │       ├── inference.py          : inference abstract class
  │       ├── model_manager.py      : model save 관련 모듈
  │       ├── model_info.py         : model 저장하기 위한 path function
  │       └── model_path_setting.py : 각 model 별 path name 설정
  │   └── brits
  │       ├── brits_model.py    : brits model 관련 class, 모듈
  │       ├── brits_trainer.py  : brits train class
  │       ├── train.py          : brits training class
  │       └── inference.py      : brits inference class
  │   └── forecasting
  │       │   └── models    : forecasting 관련 사용 model
  │       ├── train.py      : forecasting train class
  │       ├── test.py       : forecasting test 모듈
  │       ├── inference.py  : forecasting inference class
  │       ├── app.py        : test & inference application
  │       └── optimizer.py  : forecasting train optimization class
  │   └── regression
  │       │   └── models    : regression 관련 사용 model
  │       ├── train.py      : regression train class
  │       ├── inference.py  : regression inference class
  │       └── app.py        : test & inference application
  │   └── classification
  │       │   └── models    : classification 관련 사용 model
  │       ├── train.py      : classification train class
  │       ├── test.py       : classification test 모듈
  │       ├── inference.py  : classification inference class
  │       └── app.py        : test & inference application
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

