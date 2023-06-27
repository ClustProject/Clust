import json
import os

def read_model_meta_by_db_style(model_meta_db_style, meta_json_path, model_name): 
    """db style에 따라 model meta를 읽어옴, 아직 모듈화 안되어 있음 우선 여기에..
    # TODO 수정해야함

    Args:
        model_meta_db_style (str): document/influxdb
        meta_json_path (str): json의 meta path
        model_name (str): model name

    Returns:
        model_meta(str): model meta
    """
    if  model_meta_db_style == "document":
        from Clust.clust.ML.tool import meta
        model_meta = read_model_meta_from_local(meta_json_path, model_name)
    else:
        # TODO 나중에 고쳐야 함
        model_meta = read_model_meta_from_mongodb(mongodb_client, 'model', 'meta', model_name)
        model_meta = model_meta[0]
    return model_meta

##################Mongo DB
def save_model_meta_into_mongodb(mongodb_client, model_meta,db_name,collection_name):
    """model meta 를 몽고디비에 저장함

    Args:
        mongodb_client (instance): mongodb instnace
        model_meta (dict): model meta
        db_name (str): db_name
        collection_name (str): collection_name

    Returns:
        answer(int): 200(OK), 500(fail)
    """
    try :
        result = mongodb_client.insert_document(db_name, collection_name, model_meta)
        print("======== OK ========")
        print(result)
        return 200
    except Exception as e : 
        print("======== Fail ========")
        print(e)
        return 500



def read_model_meta_from_mongodb(mongodb_client, db_name, collection_name, model_name):
    """read mongo db meta

    Args:
        mongodb_client (mongodb instance): meta mongodb instance
        db name (string) : name
        collection name (string): collection
        model_name(string) : model or meta name

    Returns:
        model_meta (dict): mongo meta result
        
    """
    model_meta = mongodb_client.get_document_by_json(db_name, collection_name,{'model_info.model_name':model_name} )
    
    return model_meta 

##################Local Document 

def read_model_meta_from_local(json_file_path, model_name=None):
    """model_name에 따른 메타를 읽음

    Args:
        json_file_path (str): model info가 적힌 file path
        model_name (str): 정보를 읽으려는 model name

    Returns:
        model_meta: 해당 model name의 모델 메타
    """
    model_meta = read_json(json_file_path)
    
    if model_name:
        model_meta = model_meta[model_name]
    else:
        pass
    return model_meta

def read_model_list_from_local(json_file_path):
    """model_info를 통해 model list를 읽어옴

    Args:
        json_file_path (str): model info가 적힌 file path

    Returns:
        model_list[array]: 모델 리스트
    """
    model_meta = read_json(json_file_path)
    model_list = list(model_meta.keys())
    
    return model_list

def save_model_meta_into_local(json_file_path, new_model_meta):
    """model meta의 인포를 local에 기록함

    Args:
        json_file_path (str): model info json file의 패스
        new_model_meta (dict): 정보를 새롭게 기록하려는 메타 데이터

    Returns:
        _type_: _description_
    """
    meta_info = read_json(json_file_path)
    meta_info[new_model_meta['model_info']["model_name"]] = new_model_meta
    
    try :
        write_json(json_file_path, meta_info)
        print("======== OK ========")
        return 200
    except Exception as e : 
        print("======== Fail ========")
        print(e)
        return 500
    
def read_json(json_file_path):
    """
    The function can read json file.  
    Args:
        json_file_path(string): json file path

    Returns:
        json_result(json): json file text
    """
    check_json_file (json_file_path)
    if os.path.isfile(json_file_path):
        with open(json_file_path, 'r') as json_file:
            json_result = json.load(json_file)
    return json_result

def write_json(json_file_path, text):
    """
    The function writes text into json_file 
    Args:
        json_file_path(string): json file path
        text(dict): text to be written 
    """
    check_json_file (json_file_path)
    with open(json_file_path, 'w') as outfile:
        outfile.write(json.dumps(text))
        
    
def check_json_file (json_file_path):
    """해당 json 파일이 있는지, 없다면 생성함 초기 정보는 {}
    
    Args:
        json_file_path(string): json file path
        
    """
    
    if os.path.isfile(json_file_path):
        pass
    else: 
        directory = os.path.dirname(json_file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(json_file_path, 'w') as f:
            data={}
            json.dump(data, f, indent=2)
            print("New json file is created")
