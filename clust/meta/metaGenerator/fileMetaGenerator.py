import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))\

"""
def DBNameToDomainSubDomainName(dbname):
    domain = dbname.split("_", maxsplit=1)[0]
    sub_domain = dbname.split("_", maxsplit=1)[1]
    return domain, sub_domain
"""
class FileMetaGenerator():
    """
    fileMetaInfo에 따라서 file의 meta 정보를 읽고, meta 형식에 따라 additionalMeta가 있을 경우 추가하여 생성함

    1. get_file_meta()
        - fileName 유효한 경우 읽고 fileMeta 생성
        - fileName 유효하지 않는 경우 fileMeta {}로 초기화
            - fileMeta는 필수 table_name을 포함하고 있다고 가정함

    2. get_file_and_additionalMeta(additional)
        - get_file_meata() 로 fileMeta 생성
        - fileMeta가 dictionary인 경우 (single)
            - fileMeta가 + additonal meta 정보를 합쳐 생성
        - fileMeta가 list인 경우 (multi)
            - fileMeta의 개별 dictionary에 대해 +additional meta  합치고 반환

    """
    # Algorithm에 대해서 한번 더 컨펌
    # get_metadata_by_condition 교체해야 함
    def __init__(self, fileMetaInfo):
        """
       :param fileMetaInfo: file에 대한 정보
       :type fileMetaInfo: dictionary

        >>> file_meta_info = {
        ... "filePath" : "../Clust.clust.meta/metaSchema",
        ... "fileName" : "MSLocationMeta_Air_Indoor_체육시설.json" }
        """

        self.file_path = fileMetaInfo["filePath"]
        self.file_name = fileMetaInfo["fileName"]
    
    def get_file_and_additionalMeta(self, additional_meta):
        """

        :param additional_meta: add할 정보
       :type additional_meta: dictionary

    
        >>> additional_meta = {
            "keyword": [
                "kweather", "고등학교", "indoor","air", "co2", "pm10", "pm25", "voc", "temp", "humi", "pm01", "noise",
                "실내", "공기질", "환경", "미세먼지", "날씨", "온도", "습도", "이산화탄소", "소음"
                , "temperature", "humidity", "air , "high school", "fine dust"
            ],
            "description" : "This is weather data",
            }

        :returns: 최종 메타 정보
        :rtype: dictionary or arrary 
        """

        file_meta = self.get_file_meta()
        file_and_custom_meta = self._add_custom_meta(file_meta, additional_meta)
        return file_and_custom_meta

    def get_file_meta(self):
        """
        - file만 사용하여 meta를 생성함

        :returns: file에서 읽어드린 meta 정보
        :rtype: dictionary or arrary 
        """
        if self.file_name:
            with open(os.path.join(self.file_path, self.file_name), "r", encoding="utf-8") as meta:
                fileMeta = json.load(meta)
        else:
            fileMeta = {}
        return fileMeta


        return self.file_meta

    def _add_custom_meta(self, metaOrigin, additional_meta):
        """
        - 기존 meta에 custom 메타를 dicionary단위로 붙임
        :param meta: 기존 메타
        :type meta: dictionary or list

        :param additional_meta: 덧붙이고 싶은 Meta 정보
        :type additional_meta: dictionary
        
        :returns: 최종 메타 정보
        :rtype: dictionary or arrary 
        """
        print("start add_custom_meta")
        if type(metaOrigin) is dict: #one single meta
            metaOrigin.update(additional_meta)
            meta = metaOrigin

        elif isinstance(metaOrigin, list): #multiple dictonary list meta
            meta =[]
            for oneMeta in metaOrigin:
                oneMeta.update(additional_meta)
                meta.append(oneMeta)

        return meta
        
    
