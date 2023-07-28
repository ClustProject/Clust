import matplotlib.pyplot as plt
import seaborn as sns 

def get_plt_result(graph_type, df, param):
    """ 
    # Description         
     graph_type에 따라 plt을 생성하여 리턴함.

    # Args
     * graph_type(_str_) = [ heat_map | line chart | bar chart ]
     * df : input data
     * param: 필요 파람
      
    # Returns      
     * df(_pandas.dataFrame_) 
            
    """
    # TODO plt인 경우 바깥, 안에서 무분별하게 param을 설정하는 경우가 많은데. .이부분을 공부해서 어떻게 해야 원하는 사이즈로 이미지를 뽑을 수 있는지
    # # 그렇게 하려면 외부 변수를 어떤 식으로 받아들여야 하는지 정리 필요함
    pp = PlotPlt()
    
    if graph_type == 'heat_map' :            
        plt_ = pp.plot_heatmap(df, param)
    elif graph_type == 'line_chart' :
        plt_ = pp.plot_all_feature_line_chart(df, param)
    elif graph_type =='box_plot':
        plt_ = pp.plot_box_plot(df, param) 
    elif graph_type =='scatter':
        plt_ = pp.plot_scatter(df, param)  
    elif graph_type =='histogram':
        plt_ = pp.plot_histogram(df, param)      
    elif graph_type =='bar_chart':
        plt_ = pp.plot_bar_chart(df, param)  
    
    return plt_
 
 
class PlotPlt():
    def plot_heatmap(self, data, param):
        """
        # Description 
         plot heatmap plt

        # Args
         * data(_pandas.dataFrame_) =  Input data

        # Returns
         * plt(_pyplot module_)
            
        """
        ax = sns.heatmap(data, xticklabels = data.columns.values, yticklabels = data.index.values, annot =True, annot_kws ={'size': 4})
        bottom, top = ax.get_ylim() 
        heat_map = plt.gcf()
        ax.set_ylim(bottom+0.5, top-0.5)
        heat_map.set_size_inches(10, 6)
    
        return plt

    #plot_features->plot_plt
    def plot_all_feature_line_chart(self, data, param):
        """
        # Description 
        This function plots all column data by index. graphs are lines.
        
        # Args
         * data(_pandas.dataFrame_) =  Input data

        # Returns
         * plt(_pyplot module_)

        """
        plot_cols = data.columns
        plot_features = data[plot_cols]
        _ = plot_features.plot(subplots=True)

        plt.legend()
        
        return plt

    def plot_bar_chart(self, data, param):
        """
        # Description 
         This function plots bar chart
        
        # Args
         * data(_pandas.dataFrame_) =  Input data

        # Returns
         * plt(_pyplot module_)

        """
        plt.figure()
        data.plot.bar(subplots=True)
        return plt
    
    def plot_scatter(self, data, param):
        """
        # Description 
         This function plots scatter chart with only the front two columns
               
        # Args
         * data(_pandas.dataFrame_) =  Input data

        # Returns
         * plt(_pyplot module_)

        """
        if param: 
            first_col = param["first_col"]
            second_col = param['second_col']
        else:
            first_col = data.columns[0]
            second_col = data.columns[1]
            
        first_col
        plt.figure()
        data.plot.scatter(x=first_col, y=second_col, c='DarkBlue')

        return plt
    
    # TODO
    def plot_histogram(self, y, param):
        """
            Show histogram result 
            
            Args:
                y (numpy.ndarray): 1d array (label result)    
            Returns:
                histogram plt instance
        """
            
        plt.figure() 
        y.hist(bins=20)
        
        return plt
    
    def plot_box_plot(self, data, param):
        """
        # Description 
         This function plots scatter chart
    
        # Args
         * data(_pandas.dataFrame_) =  Input data

        # Returns
         * plt(_pyplot module_)

        """
        plt.figure()
        data.boxplot()

        return plt
        
        
