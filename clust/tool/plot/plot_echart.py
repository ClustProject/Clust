
def get_echart_json_result(self, chart_name, df):
    if chart_name == 'heat_map' :            
        result_json = df.to_json()
        
    elif chart_name == 'line_chart' :
        result_json = df.to_json()
    
    elif chart_name == 'bar_chart' :
        result_json = df.to_json()
        
    return result_json
