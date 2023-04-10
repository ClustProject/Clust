import matplotlib.pyplot as plt

def get_img_result(graph_type, df):
    """ 
    # Description         
     graph_type에 따라 plt을 생성하여 리턴함.

    # Args
     * graph_type(_str_) = [ heat_map | line chart | bar chart ]
      
    # Returns      
     * df(_pandas.dataFrame_) 
            
    """
    # TODO plt인 경우 바깥, 안에서 무분별하게 param을 설정하는 경우가 많은데. .이부분을 공부해서 어떻게 해야 원하는 사이즈로 이미지를 뽑을 수 있는지
    # # 그렇게 하려면 외부 변수를 어떤 식으로 받아들여야 하는지 정리 필요함
    pp = PlotPlt()
    
    if graph_type == 'heat_map' :            
        plt_ = pp.plot_heatmap(df)
    elif graph_type == 'line_chart' :
        plt_ = pp.plot_all_feature_line_chart(df)
    elif graph_type =='bar_chart':
        plt_ = pp.plot_bar_chart(df)  
    elif graph_type =='scatter':
        plt_ = pp.plot_scatter(df)  
    elif graph_type =='histogram':
        plt_ = pp.plot_histogram(df)      
    elif graph_type =='box_plot':
        plt_ = pp.plot_box_plot(df) 
    
    return plt_
 
 
class PlotPlt():
    def plot_heatmap(self, data):
        """
        # Description 
         plot heatmap plt

        # Args
         * data(_pandas.dataFrame_) =  Input data

        # Returns
         * plt(_pyplot module_)
            
        """
        import seaborn as sns    
        
        ax = sns.heatmap(data, xticklabels = data.columns.values, yticklabels = data.index.values, annot =True, annot_kws ={'size': 4})
        bottom, top = ax.get_ylim() 
        heat_map = plt.gcf()
        ax.set_ylim(bottom+0.5, top-0.5)
        heat_map.set_size_inches(10, 6)
    
        return plt

    #plot_features->plot_plt
    def plot_all_feature_line_chart(self, data):
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

    def plot_bar_chart(self, data):
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
    
    def plot_scatter(self, data):
        """
        # Description 
         This function plots scatter chart with only the front two columns
               
        # Args
         * data(_pandas.dataFrame_) =  Input data

        # Returns
         * plt(_pyplot module_)

        """
        plt.figure()
        data.plot.scatter(x=data.columns[0], y=data.columns[1], c='DarkBlue')

        return plt
    
    # TODO
    def plot_histogram(self, y):
        """
            Show histogram result 
            
            Args:
                y (numpy.ndarray): 1d array (label result)    
            Returns:
                histogram plt instance
        """
        """
    
        bins = np.arange(0, y.max()+1.5)-0.5
        fig, ax = plt.subplots()
        _ = ax.hist(y, bins)
        ax.set_xticks(bins+0.5)
        """
        plt.figure() 
        y.hist()
        
        return plt
    
    def plot_box_plot(self, data):
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
        
        
