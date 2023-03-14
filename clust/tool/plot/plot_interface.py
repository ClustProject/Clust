import os
import sys
sys.path.append("../")
sys.path.append("../../")

from Clust.clust.tool.plot import plot_echart, plot_image

def get_graph_result(graph_format, graph_type, df):
    
    if graph_format =='web':
        result = plot_echart.get_echart_json_result(graph_type, df) # return echart style json
    elif graph_format =='plt':
        result = plot_image.get_plt_result(graph_type, df)
    elif graph_format =='img': 
        result = plot_image.get_img_result(graph_type, df)        
        
    return result


