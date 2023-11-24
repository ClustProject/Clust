Clust: analysis
=================================
CLUST 플랫폼은 특정 방법 기준으로 시계열 데이터를 분석하고 활용할 수 있다. 이때 분석 결과를 분석 메타라고 부르며,
Clust Analysis 패키지는 분석 메타 생성에 활용되는 도구로써 사용자 지정 파라미터에 따라 다양한 분석 기능을 제공한다.

|
Analyzer
----------------------------------------------------------
Analyzer는 지정 라벨(Statistic Analyzer) 또는 지정 시간(Time Analyzer)에 
의거하여 데이터를 분석하는 모듈이다. 
분석 모듈로는 Statistic Analyzer, MeanByHoliday, MeanByWorking, MeanByTimeStep가 있으며, 
CLUST Platform Feature exploration에서 각 모듈을 이용하여 분석 및 시각화 결과를 제공한다. 


Statistic Analyzer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Statistic Analyzer는 지정 라벨을 이용하여 시계열 데이터를 분석하는 모듈이다.

.. figure:: ../image/analysis/docs_analysis_img_1.png
   :scale: 50%
   :alt: Visual Result of Statistic Analyzer
   :align: center
   :class: with-border

   Visual Result of Statistic Analyzer


|


Time Analyzer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
timeAnalyzer 패키지는 시간 기준에 따라 시계열 데이터를 분석하는 모듈이다. 
시간 기준은 세 가지이며 이에 따라 아래처럼 모듈을 분류했다.
평일, 휴일 기준으로 시계열 데이터를 분석하는 mean_by_holiday, 
지정 시간을 기준으로 분석하는 mean_by_timeStep, 근무 시간 기준으로 분석하는 mean_by_working이다. 

.. figure:: ../image/analysis/docs_analysis_img_3.png
   :scale: 30%
   :alt: Visual Result of mean_by_timeStep Time Analyzer
   :align: center
   :class: with-border

   Visual Result of mean_by_timeStep Time Analyzer

|

Analysis Interface
----------------------------------------------------------
Analysis Interface는 사용자 지정 파라미터를 확인한 후, 
그에 따라 단일 데이터 또는 데이터셋 분석을 진행하는 인터페이스이다.

- Functions by Input parameter
   - get_analysis_result
   - get_analysis_by_data
   - get_analysis_by_data_set

|
bucketReport
----------------------------------------------------------
bucket 이름과 feature에 의거한 리포트 정보를 생성하는 모듈이다.

|
Analysis
----------------------------------------------------------
단일 데이터 또는 데이터셋 분석과 관련한 함수를 모아놓은 패키지이다.


Data Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
사용자 지정 파라미터에 의거하여 단일 데이터를 분석하는 기능이다. Clust EDA Single 메뉴에서 활용한다.

**user parameter**

::

  
   ["original", 'correlation', 'scaling', 'max_correlation_value_index_with_lag','scale_xy_frequency']
    
.. figure:: ../image/analysis/docs_data_analysis_img.png
   :scale: 30%
   :alt: Single Data scaling 분석 시각화 결과
   :align: center
   :class: with-border

   Single Data scaling 분석 시각화 결과

DataSet Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
데이터셋을 분석하는 기능이다. Clust EDA Multiple 메뉴에서 활용한다.

**user parameter**

::

 ['multiple_maxabs_correlation_value_table_with_lag', 'multiple_maxabs_correlation_index_table_with_lag']

.. figure:: ../image/analysis/docs_dataSet_analysis_img.png
   :scale: 30%
   :alt: Data Set scaling 분석 시각화 결과
   :align: center
   :class: with-border

   Data Set scaling 분석 시각화 결과
|
Packages
-----------------------------

.. toctree::
   :maxdepth: 2

   analysis/analysis.statisticAnalyzer
   analysis/analysis.timeAnalyzer
   analysis/analysis.analysis_py
