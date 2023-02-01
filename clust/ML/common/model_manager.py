import sys
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
from Clust.clust.ML.common import model_info as MI
import pickle

# save model meta - mongodb
def update_model_meta_data(mongodb_client, model_meta):
    db_name = 'model'
    collection_name = 'meta'
    mongodb_client.insert_document(db_name, collection_name, model_meta)
    print(model_meta)



# get model path
# global 선언으로 train_save_pickle_model에 변수 사용 가능
# 함수안에 global을 선언했으므로 해당 함수가 호출될때마다 global 변수 값도 변경
# BUT, 해당 함수를 실행해야지만 model_file_path 값 할당 ---> 너무 종속적이다, 대안 고려
def get_model_file_path(train_data_path_list, model_method):
    global model_file_path
    model_file_path_list = MI.get_model_file_path(train_data_path_list, model_method)
    model_file_path = ''.join(model_file_path_list)
    print(model_file_path)

    return model_file_path


# save model .pkl   
def save_pickle_model(model):
    with open(model_file_path, 'wb') as file:
        pickle.dump(model, file)


# load model .pkl
def load_pickle_model(model_file_path):
    with open(model_file_path, 'rb') as file:
        model = pickle.load(file)

    return model