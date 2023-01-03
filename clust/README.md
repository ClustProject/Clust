
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
  ├── ingestion
  │   └── influx
  │       ├── influx_client.py (이) - 정리 필요  ----> 문 Review
  │       ├── influx_client_v2.py (이) - 수정 필요  -------> 문 Review
  │       ├── ms_data (문, 완료)   ---------> 황 Review
  │       └── bucket_data (문, 완료)  -------> 황 Review
  │   ├── mongo
  │       ├── customModules.py 
  │       └── mongo_Client.py 
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
  │   ├── metaFormatCheck
  │   └── metaGenerator
  │
  ├── ML
  │   ├── clustering
  │   └── model
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
      ├── entropy
      ├── featureExtension
      ├── featureReduction
      ├── general
      ├── purpose
      ├── sampling
      ├── splitDataByCondition
      ├── splitDataByCycle
      └── type
```

