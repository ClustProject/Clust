Clust: ingestion
=================================
해당 Repository는 Data Ingestion 기능들을 모아두었다.
현재 CLUST의 플랫폼 및 활용 코드에서 데이터 입출력을 위해 시계열 데이터를 저장하는 데이터베이스인 InfluxDB,
메타 정보를 저장하는 MongoDB를 사용하고 있다. 더하여 데이터를 주고 받을 때 많이 사용하는 CSV를 생성하거나 InfluxDB에 시계열 데이터 포멧으로
저장하는 방법들을 소개한다.

|

.. figure:: ../image/ingestion/ingestion_influx_mongo_save.png
   :scale: 50%
   :alt: influx mongo save
   :align: center
   :class: with-border

   InfluxDB & MongoDB Data Structure




|

InfluxDB
---------------------------------
시간 특성을 가진 시계열 데이터의 효율적인 운용을 위하여 일반적인 RDB(Relational Database)가 아닌 시계열에 특화된 TSDB(Time Series Database)인 InfluxDB를 사용하였다.
시간 정보가 인덱스로 설정이 되어 있어 일반 데이터베이스를 사용했을 때보다 입출력에서 좋은 성능을 보여준다. 일반적인 데이터베이스에서 수행하는 CURD가 모두 가능하지만 Create와 Read에 특화되어 있다.


Influx Ver.1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
InfluxDB Ver.1은 Database, Measurement, Feature의 구조를 가지고 있으며, 일반적인 SQL Query와 유사한 InfluxQL를 사용하여 CRUD를 진행한다.


**InfluxQL**
::

   # Data Read
   Example Query ==> select * from {measurement}


**Authentication Information**

InfluxDB에 접근하기 위한 정보는 host, port, user, password로 일반 접속 정보와 비슷한 구조를 가지고 있다.

::

   Authentication={
         "host" : host,
         "port" : port,
         "user" : user,
         "password" : password
         }

|

Influx Ver.2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
InfluxDB Ver.2로 업그레이드 되면서 InfluxDB의 Query, Structure, Authentication 및 UI 통한 데이터 관리 등 여러 부분등이 변경되었다.


아래의 이미지는 InfluxDB의 구조이며, Bucket(Database)하위의 Measurement에 Data가 저장되며, User는 인증 정보 및 Query를 통해 데이터를 출력할 수 있다.

.. figure:: ../image/ingestion/influx_structure.png
   :scale: 50%
   :alt: influx structure
   :align: center
   :class: with-border

   InfluxDB 2.0 version Structure



**Flux Query**

기존의 ver.1과 가장 다른 ver.2의 부분은 Query이다. 새롭게 Flux Query를 사용하게 되면서 Query 구조 및 출력 포멧도 달라지게 되었다.
아래의 ``Original ver.2 View Raw Data`` 와 같이 _start, _stop, _field, _measuremt라는 컬럼을 기준으로 값이 출력이 된다.

::

   from(bucket:"bucket_name")
   |> range(start: start_time, stop: end_time)
   |> filter(fn: (r) => r._measurement == "measurement_name")


.. figure:: ../image/ingestion/original_table.PNG
   :scale: 25%
   :alt: Original ver.2 View Raw Data
   :align: center
   :class: with-border

   Original ver.2 View Raw Data



**change result of Influx 2.x to Influx 1.x**

기존의 ver.1의 결과와 같은 데이터프레임의 형태로 만들어주기 위하여 아래의 Query를 추가하였다. 
그 결과 ``custom ver.2 View Raw Data`` 와 같이 각 Feature가 컬럼으로, _time이 시간 인덱스로 지정되면서 식별이 편리한 형태로 결과가 출력된다.

::

   |> drop(columns: ["_start", "_stop", "_measurement"])
   |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")


.. figure:: ../image/ingestion/custom_table.PNG
   :scale: 25%
   :alt: Custom ver.2 View Raw Data
   :align: center
   :class: with-border

   Custom ver.2 View Raw Data



**Authentication Information**

ver.2부터 인증 정보가 달라진다. 

- ``url`` 은 현재 서버에 등록된 InfluxDB의 IP:Port이다. 예를 들어 IP가 127.0.0.1이라면 InfluxDB 기본 Port인 8086과 합해진 127.0.0.1:8086이 ``url`` 이 된다.
- ``org`` 는 ``Organization`` 을 뜻하며, InfluxDB를 사용하는 사용자 그룹을 위한 작업공간이다. 모든 Bucket과 사용자는 이 organization에 속한다.
- ``token`` 은 각 Organization에 부여된 고유의 값이다. 이 ``token`` 값을 통해 선택된 Organization과 그 안의 데이터에 접근할 수 있다.


::

   Authentication={
         "url" : url,
         "org" : org,
         "token" : token
         }


|




MongoDB
----------------------------------
InfluxDB에 저장되는 Data와 트레이닝된 Model의 메타 정보를 저장하기 위하여 MongoDB를 사용한다.
MongoDB는 NoSql구조로서 Database, colletion, document로 구성되어 있다. 




.. figure:: ../image/ingestion/mongo_structure.png
   :scale: 50%
   :alt: mongo structure
   :align: center
   :class: with-border

   MongoDB Structure



**Authentication Information**

::

   Authentication={
            "username" : username,
            "password" : password,
            "host" : host,
            "port" : port
            }



|



Save CSV Data
-----------------------------
해당 파트에서는 시간 정보를 가진 CSV 데이터를 시계열 데이터베이스에 저장하기 위해 데이터프레임으로 생성하는 방법과 반대로 시계열 데이터베이스의 데이터 
또는 데이터프레임으로 만들어진 시계열 데이터를 CSV로 저장하는 방법에 대하여 설명한다.

- CSV Data to TSDB
   - CSV Data를 데이터 프레임으로 만든 후, TSDB에 저장
   - 시간 정보가 중요하기 때문에 중복된 시간 또는 년/월/일/시/초 형식이 맞는지 체크
   - 컬럼의 중복, 컬럼명 변경, 컬럼 삭제 등 사전에 설정 후 저장 수행

- Dataframe to CSV
   - Dataframe을 CSV 데이터로 저장
   - 미리 저장하는 path와 naming 필요
   - 일반적인 CSV 저장 과정과 동일


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
   