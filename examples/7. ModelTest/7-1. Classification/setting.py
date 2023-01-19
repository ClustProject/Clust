##########################
# Default List Information

modelConfig={
    "LSTM_cf":{# Case 1. LSTM model (w/o data representation)
        'num_layers': 2,  # recurrent layers의 수, int(default: 2, 범위: 1 이상)
        'hidden_size': 64,  # hidden state의 차원, int(default: 64, 범위: 1 이상)
        'dropout': 0.1,  # dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
        'bidirectional': True,  # 모델의 양방향성 여부, bool(default: True)   
        "lr":0.0001
    },
    "GRU_cf":{# Case 2. GRU model (w/o data representation)
        'num_layers': 2,  # recurrent layers의 수, int(default: 2, 범위: 1 이상)
        'hidden_size': 64,  # hidden state의 차원, int(default: 64, 범위: 1 이상)
        'dropout': 0.1,  # dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
        'bidirectional': True,  # 모델의 양방향성 여부, bool(default: True)
        "lr":0.0001
        
    },
    "CNN_1D_cf":{# Case 3. CNN_1D model (w/o data representation)
        'output_channels': 64, # convolution layer의 output channel, int(default: 64, 범위: 1 이상, 2의 지수로 설정 권장)
        'kernel_size': 3, # convolutional layer의 filter 크기, int(default: 3, 범위: 3 이상, 홀수로 설정 권장)
        'stride': 1, # convolution layer의 stride 크기, int(default: 1, 범위: 1 이상)
        'padding': 0, # padding 크기, int(default: 0, 범위: 0 이상)
        'drop_out': 0.1, # dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
        "lr":0.0001
    },
    "LSTM_FCNs_cf":{#Case 4. LSTM_FCNs model (w/o data representation)
        'num_layers': 1,  # recurrent layers의 수, int(default: 1, 범위: 1 이상)
        'lstm_drop_out': 0.4, # LSTM dropout 확률, float(default: 0.4, 범위: 0 이상 1 이하)
        'fc_drop_out': 0.1, # FC dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
        "lr":0.0001
    },
    "FC_cf":{# Case 5. fully-connected layers (w/ data representation)
        'drop_out': 0.1, # dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
        'bias': True,# bias 사용 여부, bool(default: True)
        "lr":0.0001}  
}

###########################
# Get Data From Files
import pickle
import pandas as pd
def getTrainDataFromFilesForClassification(folderAddress, model_name):
    if model_name in ["LSTM_cf","GRU_cf", "CNN_1D_cf","LSTM_FCNs_cf"]:
        # raw time series data
        train_x = pickle.load(open(folderAddress+'x_train.pkl', 'rb'))
        train_y = pickle.load(open(folderAddress+'y_train.pkl', 'rb'))
        test_x = pickle.load(open(folderAddress+'x_test.pkl', 'rb'))
        test_y = pickle.load(open(folderAddress+'y_test.pkl', 'rb'))

        print(train_x.shape)  #shape : (num_of_instance x input_dims x window_size) = (7352, 9, 128)
        print(train_y.shape) #shape : (num_of_instance) = (7352, )
        print(test_x.shape)  #shape : (num_of_instance x input_dims x window_size) = (2947, 9, 128)
        print(test_y.shape)  #shape : (num_of_instance) = (2947, )
        print("inputSize(train_x.shape[1]):", train_x.shape[1]) # input size
        print("sequenceLenth (train_x.shape[2]):", train_x.shape[2] )# seq_length

    if model_name in ["FC_cf"]:
        # representation data
        train_x = pd.read_csv(folderAddress+'ts2vec_repr_train.csv')
        train_y = pickle.load(open(folderAddress+'y_train.pkl', 'rb'))

        test_x = pd.read_csv(folderAddress+'ts2vec_repr_test.csv')
        test_y = pickle.load(open(folderAddress+'y_test.pkl', 'rb'))

        print(train_x.shape)  #shape : (num_of_instance x representation_dims) = (7352, 64)
        print(train_y.shape) #shape : (num_of_instance) = (7352, )
        print(test_x.shape)  #shape : (num_of_instance x representation_dims) = (2947, 64)
        print(test_y.shape)  #shape : (num_of_instance) = (2947, )
    
    return train_x, train_y,test_x, test_y