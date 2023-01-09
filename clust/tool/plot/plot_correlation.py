import matplotlib.pyplot as plt

def plot_correlation_chart(data):
    """
    Plot Correlation Chart

    Args:
        data (dataFrame): input data 
    """
    import seaborn as sns
    corr = data.corr()
    ax = sns.heatmap(corr, xticklabels = corr.columns.values, yticklabels = corr.columns.values, annot =True, annot_kws ={'size': 8})
    bottom, top = ax.get_ylim() 
    heat_map = plt.gcf()
    ax.set_ylim(bottom+0.5, top-0.5)
    heat_map.set_size_inches(10, 6)
    
    return plt
    
