def get_echart_json_result(self, graph_type, df):
    if graph_type == 'heat_map' :            
        result_json = df.to_json()
        
    elif graph_type == 'line_chart' :
        result_json = df.to_json()
    
    elif graph_type == 'bar_chart' :
        result_json = df.to_json()
        
    return result_json
