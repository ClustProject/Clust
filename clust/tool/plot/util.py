from io import BytesIO
import os
import base64
import matplotlib.pyplot as pyplt
from Clust.clust.tool.file_module import file_common

def plt_to_image(plt):
    """
    Convert plt into real image
    Args:
        plt (matplotlib.pyplot):
    Return:
        image_base64 (image_base64)
    """
    # send images
    buf = BytesIO()
    plt.savefig(buf, format='png')
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
    return image_base64


def savePlotImg(plot_name, plot_obj, img_directory) :     
    """
    - Save plot image to image directory
    - 원하는 디렉토리에 플롯 이미지를 저장한다.

    Args :
        plot_name(str)
        plot_obj(pyplot object)
        img_directory(str)
    
    Return :
        None
         
    """    
	
    file_common.check_path(img_directory,(plot_name+'.png'))
    plot_obj.savefig(os.path.join(img_directory,(plot_name+'.png')))  
    plot_obj.clf()

