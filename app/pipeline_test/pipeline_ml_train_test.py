##############################################################
# TODO # select model ## 'GRU_rg', 'LSTM_rg', 'CNN_1D_rg', 'LSTM_FCNs_rg'
model_num = 0

# TODO # set data info
task_name = "air_quality"
feature_list = ["in_co2"]

############# Train (Model info)
# TODO # set train data info
train_ms_name = "task_air_quality_case_2_level_13_pre_param_3_clustering_False_Train"

# TODO # (Train) set predict step (int)
past_step = 24
future_step = 12

############ Test
# TODO # set test data info
test_ms_name = "task_air_quality_case_4_level_12_pre_param_3_clustering_False_Test"


##############################################################
## set data info
bucket_name = "task_" + task_name

## set model param 
### forecasting model parameter samples
GRU_rg_model_info = {'hidden_size': 64, 'num_layers': 2, 'output_dim': 1, 'dropout': 0.1, 'bidirectional': 'True'}
LSTM_rg_model_info = {'hidden_size': 64, 'num_layers': 2, 'output_dim': 1, 'dropout': 0.1, 'bidirectional': 'True'}
CNN_1D_rg_model_info = {'output_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'dropout': 0.1}
LSTM_FCNs_rg_info = {'num_layers': 2, 'lstm_dropout': 0.4, 'fc_dropout': 0.1}

model_list = ['GRU_rg', 'LSTM_rg', 'CNN_1D_rg', 'LSTM_FCNs_rg']
model_param_dict = {
    "GRU_rg" : GRU_rg_model_info,
    "LSTM_rg" : LSTM_rg_model_info,
    "CNN_1D_rg" : CNN_1D_rg_model_info,
    "LSTM_FCNs_rg" : LSTM_FCNs_rg_info
}

model_method = model_list[model_num]
model_param = model_param_dict[model_method]

model_name = train_ms_name+"_"+str(past_step)+"_"+str(future_step)+"_"+model_method



##############################################################
## Train : forecasting
# set train param
train_params = {
    "ingestion_param_X" :{
        "bucket_name": bucket_name,
        "ms_name" : train_ms_name,
        "feature_list":feature_list
    },
    "ingestion_param_y":{
        "bucket_name": bucket_name,
        "ms_name" : train_ms_name,
        "feature_list":feature_list
    },
    'data_y_flag' : 'false',
    'scaler_param':{
        'scaler_flag':'scale', #scale_param,
        'scale_method' :'minmax',
        'scaler_path' :'./scaler/'
    },
    "transform_param":{
        'data_clean_option' : "false",
        'split_mode' : 'step_split', # 현재 data_y_flag=Ture --> 모두 window_split # data_y = False --> step_split
        'past_step':past_step, #step_split일 경우만 past_step과 future_step이 존재
        'future_step':future_step
    },
    
    "model_info" :{
        'model_purpose' : 'regression',
        'model_method' : model_method,    # 'GRU_rg', 'LSTM_rg', 'CNN_1D_rg', 'LSTM_FCNs_rg'
        'model_name' : model_name,
        'model_tags' : task_name+"_forecasting",
        'train_parameter' : {"lr":0.0001,"weight_decay":0.000001,"n_epochs":100,"batch_size":16},
        'model_parameter' : model_param
    }
}

##############################################################
## Test : forecasting
# set test param
test_params = {
    "ingestion_param_X" :{
        "bucket_name": bucket_name,
        "ms_name" : test_ms_name,
        "feature_list": feature_list
    },
    "ingestion_param_y":{
        "bucket_name": bucket_name,
        "ms_name" : test_ms_name,
        "feature_list": feature_list
    },
    'data_y_flag' : "None",
    'model_name':model_name
}
##############################################################
## save model meta
db_name ='model'
collection_name ='JS_kweather_airquality_test'

def save_model_meta_data(mongodb_client, model_meta):
    try :
        result = mongodb_client.insert_document(db_name, collection_name, model_meta)
        print("======== OK ========")
        print(result)
        return 200
    except Exception as e : 
        print("======== Fail ========")
        print(e)
        return 500