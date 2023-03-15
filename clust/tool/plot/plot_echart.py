def get_echart_json_result(graph_type, df)  :
    """ 
    # Description       
     graph_type에 따라 df를 가공한 후 리턴함.

    # Args
     * graph_type(_str_) : [heat_map | line chart || bar chart]
     * df(_pandas.dataFrame_)

    # Returns         
     * result_json(_json_)
            
    """
    if graph_type == 'heat_map' :            
        result_json = df.to_json()
        
    elif graph_type == 'line_chart' :
        result_json = df.to_json()
    
    elif graph_type == 'bar_chart' :
        result_json = df.to_json()
        
    return result_json
