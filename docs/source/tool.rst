Clust: tool
=================================
Clust tool은 특정 메뉴에서 사용하는 툴을 모아놓은 패키지라기 보단, 모든 모듈 곳곳에서 
활용하는 툴을 모아놓은 패키지이다.
크게 file 관련, plot 관련, 분석 테이블 생성 관련 모듈이 존재한다.

-
tool 설명
plot, stats table 이미지, 설명 추가


Plot EDA

.. image:: ../image/tool/plot_eda.png
   :scale: 70%
   :alt: Plot EDA
   :align: center

|


File
----------------------------------------------------------
파일 입출력 관련 기능을 제공하는 툴이다.

|


Plot
----------------------------------------------------------
데이터 분석 결과를 그래프로 시각화 할 때 사용하는 툴이다. 

[간단한 설명 + 이미지]


Plot Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
결과 시각화를 위한 사용자 지정 파라미터는 해당 인터페이스를 통과한다.
시용자 지정에 따라 echart tool, plot tool, image tool 등 활용 툴이 달라진다.


Plot Echart
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
javascript E-chart 그래프 생성 관련 툴이다.
사용자 지정 graph type에 따라 데이터 프레임을 json 형태로 가공하여 리턴하는 기능을 제공한다.

**graph type**

::
   ['heat_map' | 'line_chart' | 'bar_chart' | 'scatter' | 'box_plot' | 'histogram' | 'area' | 'density'] 


Plot Plt
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
pyplot 생성을 위한 툴이다. 사용자 지정 graph type에 따라 plt를 생성한 후 리턴하는 기능을 제공한다.

**graph type**

::
   ['heat_map' | 'line_chart' | 'bar_chart' | 'scatter' | 'box_plot' |'histogram'| 'area'|'density'] 


Plot Image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
pyplot 이미지 관련 툴이다. 사용자 지정 graph type에 따라 plt 이미지를 생성하고, 이미지를
byte string으로 변환 후 리턴하는 기능을 제공한다.


Plot feature
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
pyplot feature 관련 툴이다.


Plot seriesDataSet
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
pyplot seriesDataSet 관련 툴이다.
(현재 사용하지 않는 툴 2023.11.23 기준)


Plot two data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
pyplot 데이터 예측 관련 툴이다.


|


Stats Table
----------------------------------------------------------
데이터의 상관관계를 구하고 테이블을 제공하는 기능을 모아둔 패키지이다.


Correlation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Correlation 기능을 모아둔 클래스이다.


Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Metrics 관련 함수들을 모아두었다.


timelagCorr
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TimeLag Correlation 기능을 모아둔 클래스이다.


|


Packages
-----------------------------

.. toctree::
   :maxdepth: 2

   tool/tool.file_module
   tool/tool.plot
   tool/tool.stats_table