import os
##########################################
# 파일 입출력 관련 기능 모음
# os에서 루트 경로를 clust로 읽는 경우가 있어 KETIAppDataServer2에 따로 사용합니다. 
##########################################


def check_path(directory, file_name) :               
    """           
    # Description
        - 파일을 저장할 폴더 없으면 생성
        - 기존에 폴더에 저장된 파일 삭제

    # Input Data
        - String
        ```
            /home/jianso/문서/Code/CLUST/KETIAppDataServer/static/img/eda/
        ```

    """
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    if os.path.exists(file_name):
        os.remove(file_name)

def get_user_file_path(user_directory, file_name) :   
    """           
    # Description
        root 경로와 user 경로를 합쳐서 반환

    # Input Data
        user_diretory(_string_)
        file_name(_string_)

    # Returns       
        file_path(_string_)

    """

    root_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    directory = os.path.join(root_dir, user_directory)    
    file_path = os.path.join(directory,file_name)

    return file_path
