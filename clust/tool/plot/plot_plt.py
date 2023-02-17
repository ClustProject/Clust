import matplotlib.pyplot as plt

def img_graph_by_chart_name(chart_name, df):
    # TODO plt인 경우 바깥, 안에서 무분별하게 param을 설정하는 경우가 많은데. .이부분을 공부해서 어떻게 해야 원하는 사이즈로 이미지를 뽑을 수 있는지
    # 그렇게 하려면 외부 변수를 어떤 식으로 받아들여야 하는지 정리 필요함
    
    if chart_name == 'heat_map' :            
        plt_ = plot_heatmap(df)
    elif chart_name == 'line_chart' :
        plt_ = plot_all_feature_line_chart(df)
    elif chart_name =='bar_chart':
        plt_ = plot_bar_chart(df)  
    
    return plt_
 
 
def plot_heatmap(data):
    """
    plot heatmap plt

    Args:
        data (dataFrame): input data 

    Returns:
        
    """
    import seaborn as sns    
    
    ax = sns.heatmap(data, xticklabels = data.columns.values, yticklabels = data.columns.values, annot =True, annot_kws ={'size': 8})
    bottom, top = ax.get_ylim() 
    heat_map = plt.gcf()
    ax.set_ylim(bottom+0.5, top-0.5)
    heat_map.set_size_inches(10, 6)
   
    return plt

#plot_features->plot_plt
def plot_all_feature_line_chart(data):
    """
    This function plots all column data by index. graphs are lines.

    Args:
        data (dataFrame): input dataframe
    """
    plot_cols = data.columns
    plot_features = data[plot_cols]
    _ = plot_features.plot(subplots=True)
    plt.legend()
    
    return plt

def plot_bar_chart(data):
    """
    This function plots bar chart

    Args:
        data (dataFrame): input dataframe
    """
    plt.figure()
    data.plot.bar(subplots=True)
    
    return plt
    
    
