
def get_img_result(self, cart_name, df):
    plt_ = plot_plt.img_graph_by_graph_type(graph_type, df)
    util.savePlotImg(graph_type, plt_, img_directory )
    result = [graph_type] # ??????? 함수에 대한 Return이 무엇인지? byte 이미지 바로 리턴하는 것(선호), 혹은 파일 경로를 리턴해야하나 현재 그렇게 보이지가 않음
    return result
    