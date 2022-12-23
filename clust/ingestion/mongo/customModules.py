def manufacture_min_max_limit(db_client, db_name) :    
         
    """
    db에서 데이터를 가져와 민맥스 만들고 리턴하는 함수 

    Args:
        db_client (mongoClient) : mongoDB client
        db_name (String) : domain name ( ex. 'air_indoor_경로당' )

    Returns:
        data_min_max_limit (Dict) : {
            "max_num":{...}, "min_num":{...}
        }   
    
    """


    bucket_meta = db_client.get_document_by_table('bucket', 'meta_info', db_name)        
    data_min_max_limit ={"max_num":{}, "min_num":{}}    
    
    for x in bucket_meta[0]['columnInformation'] :
        columnName  = x['columnName']
        data_min_max_limit['max_num'][columnName] = x['max']
        data_min_max_limit['min_num'][columnName] = x['min']
        
    return data_min_max_limit

def update_old_dict_to_new_dict(oldDict, newDict) :   
    """
    기존 Dict를 새로운 Dict로 업데이트하는 함수

    Args:
        oldDict (Dict) 
        newDict (Dict) 

    Returns:
        oldDict (Dict) : updated data
    
    """

    oldDict['max_num'].update(newDict['max_num'])
    oldDict['min_num'].update(newDict['min_num'])
    
    #oldDict는 업데이트 된 상태
    return oldDict