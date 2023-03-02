
import os
import sys
sys.path.append("../")
sys.path.append("../../")

from Clust.clust.tool.plot import plot_plt

def get_plt_result(graph_type, df):

    #TODO: 
    plt_ = plot_plt.img_graph_by_graph_type(graph_type, df, data='value')

    print("===============================get_plt_result")
    print(plt_)

    return plt_
    
    
def get_img_graph_result(graph_type, df):
    #TODO 명확히 정의할 것 프로그램이 independent 하도록
    plt_ = get_plt_result(graph_type, df)

    #util.savePlotImg(graph_type, plt_, img_directory )
    result = [graph_type] 
    
    # ??????? 함수에 대한 Return이 무엇인지? byte 이미지 바로 리턴하는 것(선호), 혹은 파일 경로를 리턴해야하나 현재 그렇게 보이지가 않음
    
    # 바이트 스트링 코드 리턴 
    return result
    