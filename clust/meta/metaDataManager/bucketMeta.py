def get_min_max_info_from_bucketMeta(db_client, db_name) :    

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
    
    if len(bucket_meta) != 0 :
        if ('columnInformation' in bucket_meta[0]) :
            for x in bucket_meta[0]['columnInformation'] :
                column_name  = x['columnName']
                data_min_max_limit['max_num'][column_name] = x['max']
                data_min_max_limit['min_num'][column_name] = x['min']
        
    return data_min_max_limit

def update_old_dict_with_new_dict(oldDict, newDict) :   
    """
    기존 Dict를 새로운 Dict로 업데이트하는 함수

    Args:
        oldDict (Dict) :원 Dict
        newDict (Dict) :Old key와 동일 Key값이 있는 경우 업데이트 하고, 새로운 Key 값이 있는 경우 생성한다.

    Returns:
        oldDict (Dict) : updated data
    
    """

    oldDict['max_num'].update(newDict['max_num'])
    oldDict['min_num'].update(newDict['min_num'])
    
    #oldDict는 업데이트 된 상태
    return oldDict