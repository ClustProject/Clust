
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
  │       ├── influx_client (이) - _v2, _v1 에 대한 수정 필요
  │       ├── ms_data (문, 완료)
  │       └── bucket_data (문, 완료)
  │   ├── mongo
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

