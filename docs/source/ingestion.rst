Clust: ingestion
=================================
해당 Repository는 Data Ingestion 기능들을 모아두었다.
CSV에서 추출한 데이터를 InfluxDB, MongoDB 또는 반대로 Database에 있는 데이터들을 CSV로 저장하는 기능들을 소개한다.

(기능별 설명)

(데이터베이스 or 현재 데이터 사용 구조 설명, 이미지)


|

InfluxDB
---------------------------------
시간 특성을 가진 시계열 데이터의 효율적인 운용을 위하여 일반적인 RDB(Relational Database)가 아닌 시계열에 특화된 TSDB(Time Series Database)를 사용하였다.
일반 RDB와 다르게 TSDB인 InfluxDB는 Database = Bucket , table = measurement 라고 명명한다. 
InfluxDB 구버전인 1.x 이하의 버전에서는 InfluxQL를 사용하며 일반적인 SQL Query와 유사한 모습을 보여준다. 하지만 2.x 이상 버전부터 독자적인 flux Query를 사용하여 시계열 데이터에 특화된 언어를 사용한다.
본 Documents의 Ingetion/Influx에서 두가지의 Query를 사용법을 확인할 수 있다.

.. image:: ../image/influx_aa.png
   :scale: 50%
   :alt: influx structure
   :align: center



Authentication Information: ::

   Authentication={
         "url" : url,
         "token" : token,
         "org" : org
         }


|

MongoDB
----------------------------------
InfluxDB에 저장되는 Data와 트레이닝된 Model의 메타 정보를 저장하기 위하여 MongoDB를 사용하였다.
MongoDB는 NoSql구조로서 Database, colletion, document로 구성되어 있다. 

Authentication Information: ::

   Authentication={
            "username" : username,
            "password" : password,
            "host" : host,
            "port" : port
            }


.. image:: ../image/mongo_aa.png
   :scale: 50%
   :alt: mongo structure
   :align: center


|



CSV & DataFrame
-----------------------------
- dataframe save to CSV
- CSV to dataframe


|


Packages
-----------------------------

.. toctree::
   :maxdepth: 2

   ingestion/ingestion.influx
   ingestion/ingestion.mongo
   ingestion/ingestion.DataToCSV
   ingestion/ingestion.CSVtoInflux
   ingestion/ingestion.interface
   