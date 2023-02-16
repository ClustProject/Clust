import matplotlib.pyplot as plt

# def plot_correlation_chart(data):

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

def plot_bar_chart_with_line(data, line):
    """
    This function plots bar chart with red line (x = data.index, y = data['value'])

    Args:
        data (dataFrame): input dataframe
    """
    plt.axhline(y = line, color = 'r', linestyle = '-')
    data.plot.bar(subplots=True)
    
    return plt
    
    
