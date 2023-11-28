Clust: pipeline
=================================




Pipeline
-----------------------------------------------------

Purpose
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
데이터 파이프라인은 CLUST에서 제공하는 주요 모듈을 pipeline을 활용해 랜덤한 순서로 쉽게 활용하기 위한 기능을 제공한다. 
원 데이터가 들어가면  파이프라인을 타고 데이터를 처리한 후 마지막 처리된 데이터를 제공한다. 
이때 현재 총 10가지 모듈을 활용할 수 있다.

   - DataPreprocessing-Refinement
   - DataPreprocessing-OutlierDetection
   - DataPreprocessing-Imputation
   - DataPreprocessing-Smoothing
   - DataPreprocessing-Scaling
   - Data-split
   - Data-Selection
   - Data-Integration
   - Data-Quality Check
   - Data-Flatten


.. figure:: ../image/pipeline/pipeline_module.png
   :scale: 50%
   :alt: pipeline module image 
   :align: center
   :class: with-border

   pipeline module



data optimization module input
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
데이터 파이프라인을 활용하기 위해서는 데이터와 관련 모듈 파라미터를 입력으로 받아야 한다. 
각 데이터 처리 모듈은 1개의 데이터 혹은 다수의 데이터 혹은 그 두개 모두 받는 경우로 세가지 타입의로 정의 된다.


.. figure:: ../image/pipeline/pipeline_optimization_module.png
   :scale: 50%
   :alt: pipeline optimization module 
   :align: center
   :class: with-border



.. figure:: ../image/pipeline/pipeline_format.png
   :scale: 50%
   :alt: pipeline format
   :align: center
   :class: with-border




data optimization module input
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
아래 표는 각 데이터 처리 모듈에 대한 인풋과 모듈에 대한 정의를 기술한 것이다. 
예를들어 data_refinement 모듈의 input은 데이터 프레임 (DF 1개 데이터)과 데이터 프레임 셋트 (DFSet, 다수 데이터)를 모두 받을 수 있으며 각 인풋의 포맷에 따라 같은 결과 포맷을 갖는다.



.. figure:: ../image/pipeline/pipeline_in_out.png
   :scale: 50%
   :alt: pipeline in out
   :align: center
   :class: with-border

   pipeline Input Output Type



|


Use Case
------------------------------------


|



Parameter
--------------------------------------------



Default Parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Outlier Parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^




|


Packages
-----------------------------

.. toctree::
   :maxdepth: 2

   pipeline/pipeline.pipeline_py

