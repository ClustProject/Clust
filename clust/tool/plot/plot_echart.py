import json
import numpy as np

def get_echart_json_result(graph_type, df)  :
    """ 
    # Description       
     graph_type에 따라 df를 echart에서 쓰일 x_arr, y_arr, data_arr로 가공 후 리턴함.


    # Args
     * graph_type(_str_) : ['heat_map' | 'line_chart' | 'bar_chart' | 'scatter' | 'box_plot'] 
     * df(_pandas.dataFrame_)

    # Returns         
     * result_json(_json_)
            
    """
    

    if graph_type == 'heat_map' :  
        '''
                CO2     Noise      PM10      PM25      Temp      VoCs     humid
        CO2    1.000000  0.498797  0.316684  0.368846 -0.095152  0.275486  0.228707
        Noise  0.498797  1.000000  0.157772  0.185372  0.468728  0.350617 -0.288541
        PM10   0.316684  0.157772  1.000000  0.994489 -0.059075  0.361747  0.609859
        
        '''   
        index_value = df.to_json()
                
    elif graph_type == 'line_chart' :
        index_value = get_index_value_by_columns(df)
    
    elif graph_type == 'bar_chart' :
        index_value = get_index_value_by_columns(df)

    elif graph_type == 'box_plot' :
        index_value = get_index_value_by_columns(df)
    
    elif graph_type == 'scatter' :
        index_value = get_index_value_by_columns(df)

    result  = json.dumps(index_value)

    return result


def get_index_value_by_columns(df):
    """
    This is the function to create result in echart format when input parameter is dataframe.

    :param df: This is Time Series Dataframe.
    :type df: DataFrame

    :returns: echart format result
    
    :rtype: dictionary
    
    Output Example :
                    {
                        "value" : {
                            "column_name1" : [value1, value2, value3],
                            "column_name2" : [value_x, value_y, value_z]
                            },
                        "index" : ["2021-02-03 17:18", "2021-02-03 17:19", "2021-02-03 17:20"]
                    }
    """
    result ={}
    df = df.replace({np.nan:None})
   
    #index라는 컬럼이 없다. 
    result['index'] = list(df.index.strftime('%Y-%m-%d %H:%M'))
    result['value']={}
    for column in df.columns:
        value = df.loc[:, column].values.tolist()
        result['value'][column] = value
    
    return result
