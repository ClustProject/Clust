Time Analyzer
=================================
Time Analyzer는 시간 기준에 따라 시계열 데이터를 분석하는 모듈이다. 
시간 기준은 세 가지로,
평일, 휴일 기준으로 시계열 데이터를 분석하는 mean_by_holiday, 
지정 시간을 기준으로 분석하는 mean_by_timeStep, 근무 시간 기준으로 분석하는 mean_by_working 모듈로 분류된다. 
분석 기준과 분석된 정보를 활용하여 시각화 자료를 생성할 수 있다.

.. figure:: ../image/analysis/docs_analysis_img_3.png
   :scale: 30%
   :alt: Visual Result of mean_by_timeStep Time Analyzer
   :align: center
   :class: with-border

   Visual Result of mean_by_timeStep Time Analyzer

.. toctree::
   :maxdepth: 2
   :caption: Subpackages:

   timeAnalyzer/analysis.timeAnalyzer_py

