Clust: preprocessing
=================================
preprocessing 전처리 패키지는 불규칙한 데이터의 기술 문제를 해결하고 에러 데이터를 제거한다.
또한 유실된 데이터에 대해서는 파라미터에 의거하여 필요한 만큼만 유실 데이터를 보완할 수 있도록 한다.

|


preprocessing module
-------------------------------
.. image:: ../image/preprocessing_module.png
   :scale: 50%
   :alt: preprocessing module image
   :align: center

|

Error Detection
--------------------------------


|


Imputation
-----------------------------------
유실 데이터에 대해서 보완하는 모듈로 연속된 에러의 길이가 짧거나 긴 구간에 대해 적층적으로 처리할 수 있도록 구성하였다.
시계열 데이터를 활용하기에 앞서 존재하는 유실 데이터 전체를 보완하게 되면 활용하는 단계의 데이터가 활용하기 이전의 데이터와 달리 정상 데이터로 인식할 수 있는 문제점이 있다.
따라서, 해당 Imputation에서 파라미터로 조건을 기입 받아 보완 정도를 조절한다.

- Features
   - 사용자가 정의한 범위에 해당하는 필수 NaN 데이터에 대해서만 처리하고 보완
   - 장기간 연속된 유실 값을 갖는 데이터에 대해서는 사용자 파라미터를 기반하여 보완 진행 여부 결정
   - 보완 알고리즘과 이를 적용할 데이터 범위 결정 가능
   - 다수의 알고리즘을 동시에 순차적으로 적용 가능


|



Refinement
-------------------------------------
Refinement는 데이터를 균일하게 하는 모든 전처리 과정을 진행한다. 
중복된 데이터가 있거나, 데이터 중간에 유실 구간이 있는데 이를 처리하지 않은 채로 분석 및 모델링을 한다면 오류 데이터가 정상 데이터로 간주되게 되어 데이터가 편향된다. 
제안하는 Refinement는 여러 문재로 인해 생긴 중복된 군더더기 데이터를 삭제하고, 데이터가 없는 구간에 대한 시간 스탬프를 원하는 시간 주기로 기술하여 사건에 따른 데이터의 표현을 명확하게 한다.
원 데이터보다 유실 처리한 데이터가 많아지더라도 시간에 따른 데이터의 표현은 더욱 명확해질 수 있다.

- Feature
   - 파라미터 기반 데이터 정리
   - 중복 데이터 제거
   - 데이터의 기술 간격의 불규칙성을 균일하게 조정
   - 시간 순차적으로 데이터 재배열
   - 기타 데이터 클린징을 수행


|



Preprocessing Interface
-----------------------------------------
전처리를 쉽게 처리하기 위한 대표 인터페이스를 활용한다.


|


preprocessing pipeline
----------------------------------
시계열 데이터 처리 파이프라인은 데이터를 활용하고자 하는 목적에 따라 서로 다른 파이프라인 흐름을 활용해야 한다.
아래 그림은 전처리 파이프라인이 활용될 수 있는 어플리케이션에 대한 데이터 흐름도를 나타낸다.
데이터는 하나의 데이터 혹은 여러 개의 데이터 묶음에 대해서 일괄 처리가 가능하다. 데이터 전처리는 파라미터의 기술 내용에 따라 취사 선택하여 활용된다.
만약 데이터 전처리 파라미터가 정의되지 않은 경우 processing parameter를 생성하는 함수를 통해 자동으로 생성이 가능하다.

.. image:: ../image/preprocessing_pipeline.png
   :scale: 50%
   :alt: preprocessing pipeline image
   :align: center

|



Packages
---------------------

.. toctree::
   :maxdepth: 2

   preprocessing/preprocessing.errorDetection

   preprocessing/preprocessing.imputation
   
   preprocessing/preprocessing.refinement

   preprocessing/preprocessing.preprocessing_py