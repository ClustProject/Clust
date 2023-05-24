import sys
sys.path.append("../")
sys.path.append("../../")

from Clust.clust.tool.plot import plot_echart, plot_image, plot_plt

def get_graph_result(graph_format, graph_type, df):
    """ 
    # Description       
     graph_type에 따라 그래프 생성에 필요한 정보를 가공하고, graph_format 방법으로 결과를 리턴하는 함수

    # Args
     * graph_type(_str_)    = [ heat_map | line chart || bar chart ]
     * graph_format         = ['web' | 'plt' | 'img']
     * df(_pandas.dataFrame_)

    # Returns         
     * result_json(_json_)
            
    """
        
    if graph_format =='web':
        result = plot_echart.get_echart_json_result(graph_type, df) # return echart style json
    elif graph_format =='plt':
        result = plot_plt.get_plt_result(graph_type, df)
    elif graph_format =='img': 
        result = plot_image.get_img_result(graph_type, df)        
        
    return result


