```bash

  ML/
  ├── app
      ├── usecase.py : 응용 어플리케이션을 위함
  ├── clustering
      ├── interface.py :clustering 모듈을 활용하기 위한 인터페이스 
      ├── cluetring.py :train, test를 위한 추상화 클래스 기술
      ├── kMeans.py    :clustering algorithm 1
      ├── som.py       :clustering algorithm 2
      └── etc.py       :clustering algorithm 3 (모듈화 어려운)
  └── tool
      ├── data.py      : ML 인풋 데이터 처리 관련 모듈
      ├── model.py     : ML 모델 입출력 관련 모듈
      └── util.py      : ML 데이터/모델 이외 관련 공통 모듈
  └── common
      │   └── common
      │       ├── p1_integratedDataSaving.py
      │       ├── p2_dataSelection.py
      │       ├── p3_training.py
      │       └── p4_testing.py
      ├── trainer.py            : train abstract class
      ├── inference.py          : inference abstract class
      ├── model_manager.py      : model save 관련 모듈
      ├── model_info.py         : model 저장하기 위한 path function
      └── model_path_setting.py : 각 model 별 path name 설정
  └── brits
      ├── brits_model.py    : brits model 관련 class, 모듈
      ├── brits_trainer.py  : brits train class
      ├── train.py          : brits training class
      └── inference.py      : brits inference class
  └── forecasting
      ├── gru_forecasting.py        : GRU model forecasting train & Test & Inference class
      ├── lstm_forecasting.py       : LSTM model forecasting train & Test & Inference class
      ├── rnn_forecasting.py        : RNN model forecasting train & Test & Inference class
      └── app.py                    : test & inference application
  └── regression
      ├── cnn_1d_regression.py      : 1D CNN model regression train & Test & Inference class
      ├── fc_regression.py          : FC model regression inference class
      ├── lstm_fcns_regression.py   : LSTM FCNs model regression inference class
      ├── rnn_regression.py         : RNN model regression inference class
      └── app.py        : test & inference application
  └── classification
      ├── cnn_1d_classification.py      : 1D CNN model classification train & Test & Inference class
      ├── fc_classification.py          : FC model classification inference class
      ├── lstm_fcns_classification.py   : LSTM FCNs model classification inference class
      ├── rnn_classification.py         : RNN model classification inference class
      └── app.py        : test & inference application

```

