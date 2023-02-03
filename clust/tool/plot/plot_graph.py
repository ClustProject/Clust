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

    
