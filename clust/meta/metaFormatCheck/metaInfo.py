def get_ms_meta_info_inBucket(ms_meta_info):

    basic_a_keys    = ['domain', 'subDomain', 'table_name', 'fields', 'startTime', 'endTime', 'pointCount', 'frequency', 'numberOfColumns']            
    basic_b_keys    = ['keyword', 'location'] #'description', 'sourceAgency', 'sourceType' 추후 추가
    analysis_a_keys = ['analysisResult']

                  
    basic_a_flag = check_including_keyList(ms_meta_info, basic_a_keys)
    basic_b_flag = check_including_keyList(ms_meta_info, basic_b_keys)
    analysis_a_flag = check_including_keyList(ms_meta_info, analysis_a_keys)

    ms_meta_array={}
    ms_meta_array['basic_a_flag'] = basic_a_flag
    ms_meta_array['basic_b_flag'] = basic_b_flag
    ms_meta_array['analysis_a_flag'] = analysis_a_flag
    return ms_meta_array

def get_ms_meta_info_inBucket(bucket_meta_info):
    
    bucket_meta_array ={}
    #key 변경 시 에러남. key 고정 또는 db에 저장 필요
    basic_a_keys= ['description', 'domain', 'subDomain', 'table_name']
    basic_b_keys =['columnInformation']
    analysis_a_keys=['analysisResult']

    bucket_meta_array['basic_a_flag'] = check_including_keyList(bucket_meta_info, basic_a_keys)
    bucket_meta_array['basic_b_flag'] = check_including_keyList(bucket_meta_info, basic_b_keys)
    bucket_meta_array['analysis_a_flag'] = check_including_keyList(bucket_meta_info, analysis_a_keys)
    bucket_meta_array['meta_info'] = bucket_meta_info

    return bucket_meta_array


def seperate_bucket_name(bucketName) :

    #bucket name => domain, sub_domain
    domain = bucketName.split("_", maxsplit=1)[0]
    subDomain = bucketName.split("_", maxsplit=1)[1]

    return domain, subDomain
    
def check_including_keyList(dict_input, key_list) :
     
    """
        dict_input이 key_list의 모든 key값을 포함 하는지 확인
        

        Args:
            (Dict) dict_input = {'tags': [], 'fields': ...
            (list) key_list  =['a', 'b' ...]

        Returns:
            (Boolean) True|False
    
    """

    a = set(dict_input.keys())
    b = set(key_list)
    result = b.issubset(a)

    return result

