import json
import numpy as np

def get_echart_json_result(graph_type, df)  :
    """ 
    # Description       
     graph_type에 따라 df를 echart에서 쓰일 x_arr, y_arr, data_arr로 가공 후 리턴함.

    # Args
     * graph_type(_str_) : ['heat_map' | 'line_chart' | 'bar_chart' | 'scatter' | 'box_plot' | 'histogram' | 'area' | 'density'] 
     * df(_pandas.dataFrame_)

    # Returns         
     * result_json(_json_)
            
    """    
    print(df)

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
        index_value = get_bar_data(df)

    elif graph_type == 'box_plot' :
        index_value = get_index_value_by_columns(df)
    
    elif graph_type == 'scatter' :
        index_value = get_scatter_data(df)
    
    elif graph_type == 'area' :
        index_value = get_index_value_by_columns(df)
    
    elif graph_type == 'histogram' : 
        #echart 제공 안 됨
        index_value = get_index_value_by_columns(df)    
    
    elif graph_type == 'density' : 
        #echart 제공 안 됨
        index_value = get_index_value_by_columns(df)
    
    else   :
        index_value = {}
    
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
    print(df.index)
    result['value']={}
    for column in df.columns:
        value = df.loc[:, column].values.tolist()
        result['value'][column] = value
    
    return result

def get_scatter_data(df):
    """
    
    This is the function to create result for scatter graph 
    in echart format when input parameter is dataframe.

    :param df: This is Time Series Dataframe.
    :type df: DataFrame

    :returns: echart format result
    
    :rtype: dictionary
    
    Output Example :
                    {
                        "value" : {
                            "scatter" : [value1, value2, value3]                            
                            },
                        "index" : ["2021-02-03 17:18", "2021-02-03 17:19", "2021-02-03 17:20"]
                    }
    """

    result = {}

    df = df.replace({np.nan:None})     
    
    result['value'] = {'scatter' : [], 'keys' : [df.columns[0], df.columns[1]]}

    size = len(df[df.columns[0]])   

    for x in range(size):        
        result['value']['scatter'].append([  df[df.columns[0]][x], df[df.columns[1]][x]] )       

    
    return result

def get_bar_data(df):
    """
    
    This is the function to create result for scatter graph 
    in echart format when input parameter is dataframe.

    :param df: This is Time Series Dataframe.
    :type df: DataFrame

    :returns: echart format result
    
    :rtype: dictionary
    
    Output Example :
                    {
                        "value" :  [['product', '2015', '2016', '2017'],
                                    ['Matcha Latte', 43.3, 85.8, 93.7],
                                    ['Milk Tea', 83.1, 73.4, 55.1],
                                    ['Cheese Cocoa', 86.4, 65.2, 82.5],
                                    ['Walnut Brownie', 72.4, 53.9, 39.1]]
                     
                    }
    """

    result = {}
    df = df.replace({np.nan:None})     
    
    result['value'] = []

    for column in df.columns:      
        value = df.loc[:, column].values.tolist()           
        value.insert(0, column)
        result['value'].append(value)
    

    return result