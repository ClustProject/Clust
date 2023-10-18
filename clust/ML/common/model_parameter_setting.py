
def set_model_parameter(model_info):
    """
    set model parameter

    Args:
        model_info (dict): model information    
        seq_len (_type_): _description_
        input_size (_type_): _description_

    Returns:
        model_parameter(dict):model_parameter

    """
    model_method = model_info['model_method']
    seq_len = model_info['seq_len']
    input_size = model_info['input_size']
    regression_model_list = ['LSTM_rg' , 'GRU_rg',  'CNN_1D_rg', 'LSTM_FCNs_rg',  'FC_rg']
    classification_model_list = [ 'LSTM_cf', 'GRU_cf', 'LSTM_cf', 'CNN_1D_cf', 'LSTM_FCNs_cf', 'FC_cf']
    anomaly_detection_model_list = ['BeatGAN_ad', 'LSTM_VAE_ad']

    if model_method in regression_model_list:
        model_parameter = get_regression_model_parameter(model_method, model_info['model_parameter'], seq_len, input_size)
    elif model_method in classification_model_list:
        model_parameter = get_classification_model_parameter(model_method, model_info['model_parameter'], seq_len, input_size)
    elif model_method in anomaly_detection_model_list:
        model_parameter = get_anomaly_detection_model_parameter(model_method, model_info['model_parameter'], seq_len, input_size)
    else:
        model_parameter = None
        
    return model_parameter

def get_regression_model_parameter(model_method, model_info, seq_len, input_size):
    """ 
    regression & forecasting model method parameter 
    
    Args:
        model_method (string): model_method
        model_info (dict): model_info
        seq_len (int): seq_len
        input_size (int): input_size

    """

    if model_method == 'LSTM_rg' or model_method == 'GRU_rg':
        print(model_info)
        model_parameter = {
            'rnn_type': 'lstm',
            'input_size': input_size, 
            'hidden_size': model_info['hidden_size'],
            'num_layers': model_info['num_layers'],
            'output_dim': model_info['output_dim'],
            'seq_len': seq_len,
            'dropout': model_info['dropout'], 
            'bidirectional': model_info['bidirectional']
        }
        if model_method == 'LSTM_cf':
            model_parameter['rnn_type'] = 'lstm'
        elif model_method == 'GRU_cf':
            model_parameter['rnn_type'] = 'gru'
                
    # CNN_1D model parameters
    elif model_method == 'CNN_1D_rg':
        model_parameter = {
        'input_size': input_size,
        'seq_len': seq_len,
        'output_channels': model_info['output_channels'],
        'kernel_size': model_info['kernel_size'],
        'stride': model_info['stride'],
        'padding': model_info['padding'], 
        'dropout': model_info['dropout']
        }
    # LSTM_FCNs model parameters
    elif model_method == 'LSTM_FCNs_rg':
        model_parameter = {
        'input_size': input_size,
        'num_layers': model_info['num_layers'],
        'lstm_dropout': model_info['lstm_dropout'],
        'fc_dropout': model_info['fc_dropout']
        }
    # FC model parameters
    elif model_method == 'FC_rg':
        model_parameter = {
        'input_size': input_size,
        'dropout': model_info['dropout'],
        'bias': model_info['bias']
        }

    return model_parameter

def get_classification_model_parameter(model_method, model_info, seq_len, input_size):
    """ classification model method parameter 
    
    Args:
        model_method (string): model_method
        model_info (dict): model_info
        seq_len (int): seq_len
        input_size (int): input_size
    """

    if model_method == 'LSTM_cf' or model_method == 'GRU_cf':
        model_parameter = {
            'input_size': input_size,
            'seq_len': seq_len,
            'output_dim': model_info['output_dim'],
            'num_classes': model_info['num_classes'],
            'num_layers': model_info['num_layers'],  # recurrent layers의 수, int(default: 2, 범위: 1 이상)
            'hidden_size': model_info['hidden_size'],  # hidden state의 차원, int(default: 64, 범위: 1 이상)
            'dropout': model_info['dropout'],  # dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
            'bidirectional': model_info['bidirectional']  # 모델의 양방향성 여부, bool(default: True)   
        }
        if model_method == 'LSTM_cf':
            model_parameter['rnn_type'] = 'lstm'
        elif model_method == 'GRU_cf':
            model_parameter['rnn_type'] = 'gru'
            
    # CNN_1D model parameters
    elif model_method == 'CNN_1D_cf':
        model_parameter = {
        'input_size': input_size,
        'seq_len': seq_len,
        'num_classes': model_info['num_classes'],
        'output_channels': model_info['output_channels'], # convolution layer의 output channel, int(default: 64, 범위: 1 이상, 2의 지수로 설정 권장)
        'kernel_size': model_info['kernel_size'], # convolutional layer의 filter 크기, int(default: 3, 범위: 3 이상, 홀수로 설정 권장)
        'stride': model_info['stride'], # convolution layer의 stride 크기, int(default: 1, 범위: 1 이상)
        'padding': model_info['padding'], # padding 크기, int(default: 0, 범위: 0 이상)
        'dropout': model_info['dropout'] # dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
        }
        
    # LSTM_FCNs model parameters
    elif model_method == 'LSTM_FCNs_cf':
        model_parameter = {
        'input_size': input_size,
        'seq_len': seq_len,
        'num_classes': model_info['num_classes'],
        'num_layers': model_info['num_layers'],  # recurrent layers의 수, int(default: 1, 범위: 1 이상)
        'lstm_dropout': model_info['lstm_dropout'], # LSTM dropout 확률, float(default: 0.4, 범위: 0 이상 1 이하)
        'fc_dropout': model_info['fc_dropout'] # FC dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
        }
        
    # FC model parameters
    elif model_method == 'FC_cf':
        model_parameter = {
        'input_size': input_size,
        'num_classes': model_info['num_classes'],
        'drop_out': model_info['drop_out'], # dropout 확률, float(default: 0.1, 범위: 0 이상 1 이하)
        'bias': model_info['bias']# bias 사용 여부, bool(default: True)
        }

    return model_parameter

def get_anomaly_detection_model_parameter(model_method, model_info, seq_len, input_size):
    """ anomaly detection model method parameter 
    
    Args:
        model_method (string): model_method
        model_info (dict): model_info
        seq_len (int): seq_len
        input_size (int): input_size
    """

    # BeatGAN model parameter
    if model_method == 'BeatGAN_ad':
        model_parameter = {
            'input_size': input_size,
            'seq_len': seq_len,
            'in_c': model_info['in_c'],
            'hidden_c': model_info['hidden_c'],
            'latent_c': model_info['latent_c']
        }
    elif model_method == 'LSTM_VAE_ad':
        pass
                
    return model_parameter

