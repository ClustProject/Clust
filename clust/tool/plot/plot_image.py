

import os
import sys
import io, base64
sys.path.append("../")
sys.path.append("../../")

from Clust.clust.tool.plot import plot_plt

def get_plt_result(graph_type, df):

    plt_ = plot_plt.img_graph_by_graph_type(graph_type, df)

    return plt_
    
    
def get_img_result(graph_type, df):
    #TODO 명확히 정의할 것 프로그램이 independent 하도록
    
    plt_ = get_plt_result(graph_type, df)
    my_stringIObytes = io.BytesIO()
    plt_.savefig(my_stringIObytes, format='jpg')
    my_stringIObytes.seek(0)    
    my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode()

    
    return my_base64_jpgData

