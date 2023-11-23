import sys
sys.path.append("../")
sys.path.append("../../")
import numpy as np

from Clust.clust.tool.plot import plot_echart, plot_image, plot_plt

def get_graph_result(graph_format, graph_type, df, param = None):
    """       
    graph_type에 따라 그래프 생성에 필요한 정보를 가공하고, graph_format 방법으로 결과를 리턴하는 함수

    Args:
        graph_type(_str_) :  graph type
        graph_format : ['web' | 'plt' | 'img']
        df(_pandas.dataFrame_) : dataframe
        param (json) : parameter

    >>>     graph_type = 
    ...     ['heat_map' | 'line_chart' | 'bar_chart' | 'scatter' | 'box_plot' |'histogram'| 'area'|'density'] 

    Returns:
        img or plt or json: result

            
    """

    # by graph_type
    if graph_type =='area':
        df = abs(df)

    # by graph_format
    if graph_format =='web':
        result = plot_echart.get_echart_json_result(graph_type, df) # return echart style json
    elif graph_format =='plt':
        result = plot_plt.get_plt_result(graph_type, df)
    elif graph_format =='img': 
        result = plot_image.get_img_result(graph_type, df)        
        
    return result

#plot_analised_graph

from Clust.clust.analysis import analysis_interface 
def scale_xy_frequency(data, analysis_method, analysis_param, plot_format, plot_type_list):
    """       
    한개의 데이터를 분석후 여러 plot type으로 그림을 그리는 plot 함수

    Args:
        data(_pandas.dataFrame_) :  input data
        analysis_method(str) : one of analysis_methods
        analysis_param(json) : parameter
        plot_format(str): plot_format
        plot_type_list (array) : multiple (or one) plot type list

    Returns:
        img or plt or json: result

            
    """
    analysis_data = np.round(analysis_interface.get_analysis_result(analysis_method, analysis_param, data), 2)
    for plot_type in plot_type_list:
        result = get_graph_result(plot_format, plot_type, analysis_data)
    return result

