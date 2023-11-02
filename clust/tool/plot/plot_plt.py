import matplotlib.pyplot as plt
import seaborn as sns 

def get_plt_result(graph_type, data):
    """ 
    Description         
    graph_type에 따라 plt을 생성하여 리턴함.

    Args:
        graph_type(_str_) :  graph type
        data(dataframe) : input data
      
    >>>     graph_type = 
    ...     ['heat_map' | 'line_chart' | 'bar_chart' | 'scatter' | 'box_plot' |'histogram'| 'area'|'density'] 

    Returns:      
        plot : plt 
            
    """
    
    pp = PlotPlt()

    if graph_type == 'heat_map' :            
        plt_ = pp.plot_heatmap(data)
    elif graph_type == 'line_chart' :
        plt_ = pp.plot_all_feature_line_chart(data)
    elif graph_type =='box_plot':
        plt_ = pp.plot_box_plot(data) 
    elif graph_type =='scatter':
        plt_ = pp.plot_scatter(data)  
    elif graph_type =='histogram':
        plt_ = pp.plot_histogram(data)      
    elif graph_type =='bar_chart':
        plt_ = pp.plot_bar_chart(data)  
    elif graph_type == 'area':
        plt_ = pp.plot_area_chart(data)
    elif graph_type == 'density':
        plt_ = pp.plot_density(data)
    
    return plt_
 
 
class PlotPlt():
    def plot_heatmap(self, data):
        """
        plot heatmap plt

        Args:
            data(dataframe) : Input data

        Returns:
            plot : plt
            
        """
        plt.figure()
        ax = sns.heatmap(data, xticklabels = data.columns.values, yticklabels = data.index.values, 
                         cmap="Blues", annot =True, annot_kws ={'size': 8})
        bottom, top = ax.get_ylim() 
        heat_map = plt.gcf()
        ax.set_ylim(bottom+0.5, top-0.5)
        #heat_map.set_size_inches(10, 6)
    
        return plt

    #plot_features->plot_plt
    def plot_all_feature_line_chart(self, data):
        """
        This function plots all column data by index. graphs are lines.
        
        Args:
            data(_pandas.dataFrame_) :  Input data

        Returns:
            plot : plt

        """
        plt.figure()
        plot_cols = data.columns
        plot_features = data[plot_cols]
        _ = plot_features.plot(subplots=True)

        plt.legend()
        
        return plt

    def plot_bar_chart(self, data):
        """
        This function plots bar chart
        
        Args:
            data(_pandas.dataFrame_) :  Input data

        Returns:
            plot : plt

        """
        plt.figure()
        data.plot.bar(subplots=True)
        return plt
    
    def plot_scatter(self, data):
        """
        This function plots scatter chart with only the front two columns
               
        Args:
            data(_pandas.dataFrame_) :  Input data, 입력은 두개의 컬럼만 가져야 함
            param(dict) : feature_list

        Returns:
            plot : plt

        """
        feature_list = list(data.columns)
        plt.figure()
        data.plot.scatter(x=feature_list[0], y=feature_list[1], c='DarkBlue')

        return plt
    

    def plot_box_plot(self, data):
        """
        This function plots scatter chart
    
        Args:
            data(_pandas.dataFrame_) :  Input data

        Returns:
            plot : plt

        """
        plt.figure()
        data.boxplot()

        return plt
    
    def plot_histogram(self, data):
        """
        Show histogram result 

        Args:
            data(_pandas.dataFrame_) :  Input data  , 입력은 하나의 column만 가져야 함
            param(dict) : feature_list

        Returns:
            plot : plt(histogram plt instance)

        """
            
        data_h = data.astype(float)
        plt.figure() 
        data_h.plot.hist(bins=20, alpha = 0.5)
        
        return plt
    
    def plot_area_chart(self, data):
        """
        This function plots area plot
    
        Args:
            data(_pandas.dataFrame_) :  Input data

        Returns:
            plot : plt

        """
        plt.figure()
        data.plot.area()
        
        return plt
    
    def plot_density(self, data):
        """
        This function plots area plot
    
        Args:
            data(_pandas.dataFrame_) :  Input data

        Returns:
            plot : plt

        """
        plt.figure()
        data.plot.density()
        
        return plt
        
