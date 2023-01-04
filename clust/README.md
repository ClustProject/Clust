
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
  │   └── influx (진행중)
  │       ├── influx_client.py - (이)정리 필요  
  │       ├── influx_client_v2.py - (이)수정 필요  
  │       ├── ms_data - (문/완료)   
  │       └── bucket_data - (문/완료) 
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
  │       └── bucketMeta.py (소/완료) - (문/검완료)
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

