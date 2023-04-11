import sys
import io, base64
sys.path.append("../")
sys.path.append("../../")

from Clust.clust.tool.plot import plot_plt

def get_img_result(graph_type, df):
    """ 
    # Description    
     graph_type에 따라 생성한 plt이미지를 byte string으로 변환하여 리턴함.    

    # Args
     * graph_type(_str_) = [ heat_map | line chart | bar chart ]

    # Returns
     * df(_pandas.dataFrame_)        
            
    """
    #TODO 명확히 정의할 것 프로그램이 independent 하도록
    
    plt_ = plot_plt.get_img_result(graph_type, df)
    jpg_data = plt_to_image(plt_)
    
    return jpg_data

def plt_to_image(plt):
    """
    Convert plt into real image
    Args:
        plt (matplotlib.pyplot):
    Return:
        image_base64 (image_base64)
    """
    # send images
    buf = io.BytesIO()
    plt.savefig(buf, format='jpg')
    #image_base64 = base64.b64encode(buf.read())
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
    return image_base64
