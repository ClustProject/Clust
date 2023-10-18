import json
import numpy as np
def getEChartFormatResult(data):
    """
    This is the function to create result in echart format when input parameter is dataframe.

    :param data: This is Time Series Dataframe.
    :type data: DataFrame

    :returns: echart format result
    
    :rtype: dictionary
    
    Output Example ::

                    {
                        "value" : {
                            "column_name1" : [value1, value2, value3],
                            "column_name2" : [value_x, value_y, value_z]
                            },
                        "index" : ["2021-02-03 17:18", "2021-02-03 17:19", "2021-02-03 17:20"]
                    }
    """
    result ={}
    data = data.replace({np.nan:None})

    print("=========================data")
    print(data)

    #result['index'] = list(data.index.strftime('%Y-%m-%d %H:%M'))
    result['index'] = list(data.index)
    result['value']={}
    for column in data.columns:
        value = data.loc[:, column].values.tolist()
        result['value'][column] = value
    
    return result

def getEchartFormatResultForJsonInput(data_json):
    """
    This is the function to create result in echart format when input parameter is json.


    :param data: This is json data. (data.to_json(orient="split"))
    :type data: json

    :returns: echart format result
    
    :rtype: dictionary
    
    Input Example ::

                    {
                        "columns":["column_name1","column_name2","column_name3"],
                        "index":["2021-02-03 17:18", "2021-02-03 17:19", "2021-02-03 17:20"],
                        "data":[[value1, value_x, value_a], [value2, value_y, value_b], [value_3, value_z, value_c]]
                    }
     
    Output Example ::
    
                    {
                        "value" : {
                            "column_name1" : [value1, value2, value3],
                            "column_name2" : [value_x, value_y, value_z],
                            "column_name3" : [value_a, value_b, value_c]
                            },
                        "index" : ["2021-02-03 17:18", "2021-02-03 17:19", "2021-02-03 17:20"]
                    }
    """
    result ={}
    data_dict = json.loads(data_json)
    result["index"] = data_dict["index"]
    result["value"] ={}
    for column in data_dict["columns"]:
        column_idx = data_dict["columns"].index(column)
        value = []
        for data in data_dict["data"]:
            value.append(data[column_idx])
        result["value"][column] = value
        
    return result