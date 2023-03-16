# Modify this Path to make adaptive model path environment
my_model_root_path =['./Models']
#my_model_root_path = ['/home','keti', 'CLUST_KETI', 'Clust', 'KETIAppTestCode','MLModelTest','Models']
my_model_info_list = {
            "brits":{
                "model_root_path": my_model_root_path, 
                "model_info_path": ["brits"],
                "model_file_names":['model.json', 'model.pkl']},
            "lstm":{
                "model_root_path": my_model_root_path, 
                "model_info_path": ["lstm"],
                "model_file_names":['model_state_dict.pkl']},
            "gru":{
                "model_root_path": my_model_root_path, 
                "model_info_path": ["gru"],
                "model_file_names":['model_state_dict.pkl']},
            # TODO 아래 클래시피케이션 중복 아닌지 확인해야함.
            "LSTM_cf":{
                "model_root_path": my_model_root_path, 
                "model_info_path": ["LSTM_cf"],
                "model_file_names":['model.pkl']},
            "GRU_cf":{
                "model_root_path": my_model_root_path, 
                "model_info_path": ["GRU_cf"],
                "model_file_names":['model.pkl']},
            "CNN_1D_cf":{
                "model_root_path": my_model_root_path, 
                "model_info_path": ["CNN_1D_cf"],
                "model_file_names":['model.pkl']},
            "LSTM_FCNs_cf":{
                "model_root_path": my_model_root_path, 
                "model_info_path": ["LSTM_FCNs_cf"],
                "model_file_names":['model.pkl']},
            "FC_cf":{
                "model_root_path": my_model_root_path, 
                "model_info_path": ["FC_cf"],
                "model_file_names":['model.pkl']},
            # Regression Model
            "LSTM_rg":{
                "model_root_path": my_model_root_path, 
                "model_info_path": ["LSTM_rg"],
                "model_file_names":['model.pkl']},
            "GRU_rg":{
                "model_root_path": my_model_root_path, 
                "model_info_path": ["GRU_rg"],
                "model_file_names":['model.pkl']},
            "CNN_1D_rg":{
                "model_root_path": my_model_root_path, 
                "model_info_path": ["CNN_1D_rg"],
                "model_file_names":['model.pkl']},
            "LSTM_FCNs_rg":{
                "model_root_path": my_model_root_path, 
                "model_info_path": ["LSTM_FCNs_rg"],
                "model_file_names":['model.pkl']},
            "FC_rg":{
                "model_root_path": my_model_root_path, 
                "model_info_path": ["FC_rg"],
                "model_file_names":['model.pkl']}
}
