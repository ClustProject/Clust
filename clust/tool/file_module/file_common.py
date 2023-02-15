import os
##########################################
# 파일 입출력 관련 기능 모음
##########################################

def check_path(directory, file_name) :               
    """           
    # Description
    - 파일을 저장할 폴더 없으면 생성
    - 기존에 폴더에 저장된 파일 삭제

    # Input Data
    *String
    ```
        /home/jianso/문서/Code/CLUST/KETIAppDataServer/static/img/eda/
    ```

    """
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    if os.path.exists(file_name):
        os.remove(file_name)
