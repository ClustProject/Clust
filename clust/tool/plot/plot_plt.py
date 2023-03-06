import matplotlib.pyplot as plt

class PlotPlt():
    def plot_heatmap(self, data):
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
    def plot_all_feature_line_chart(self, data):
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

    def plot_bar_chart(self, data):
        """
        This function plots bar chart
        Args:
            data (dataFrame): input dataframe
        """
        plt.figure()
        data.plot.bar(subplots=True)
        
        return plt
    
    def plot_scatter(self, data):
        """This function plots scatter chart with only the front two columns
        
         
    
        Args:
            data (dataframe): Input data

        """
        plt.figure()
        data.plot.scatter(x=data.columns[0], y=data.columns[1], c='DarkBlue')
        return plt
    
    def plot_box_plot(self, data):
        """This function plots scatter chart
    
        Args:
            data (dataframe): Input data

        """
        plt.figure()
        data.boxplot()
        return plt
        
        
def img_graph_by_graph_type(graph_type, df):
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
    elif graph_type =='box_plot':
        plt_ = pp.plot_box_plot(df) 
    
    return plt_
 