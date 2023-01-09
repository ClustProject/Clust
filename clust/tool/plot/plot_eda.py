import os
import sys
sys.path.append("../")
sys.path.append("../../")

import re
import plotly.graph_objects as go
import plotly.express as px
from Clust.clust.transformation.splitDataByCycle import dataByCycle
from Clust.clust.transformation.sampling.data_up_down import DataUpDown


def show_one_comaparing_distributions_of_multiple_data(dataset, feature):
    """
    Args:
        dataset (dictionary of dataframe): Input dataset
        feature (string): A column of data as one of column name to be shown
    """
    layout = go.Layout(title = feature)
    fig = go.Fiqure(layout = layout)
    for data_name in dataset.keys():
        trace = go.Box(y = dataset[data_name][feature], name = data_name)
        fig.add_trace(trace)
    fig.show()

def show_one_feature_data_based_on_two_times(data, feature, time_criteria):
    """
    A function that visualizes the amount of change in a specific feature of one data as a heatmap based on two time standards.

    Args:
        data (dataframe): Input data
        #feature (string): A column of data as one of column name to be shown
        time_criteria (dictionary): The time frequency of the x-axis and y-axis of the graph
                                    

    Example:
        >>> time_criteria = {"x_frequency" : "1Hour", "y_frequency" : "1Day"}
    """
    x_frequency = time_criteria["x_frequency"]
    y_frequency = time_criteria["y_frequency"]

    # get down samping frequency
    string_time = re.findall('\D', x_frequency)[0]
    numeric_time = re.findall('\d+', x_frequency)[0]
    downsampling_freq = numeric_time+string_time

    # down sampling by donwsampling_freq (x_frequency)
    data = DataUpDown().data_down_sampling(data, downsampling_freq)

    try:
        # split data by y_frequency
        feature_cycle = re.findall('\D+', y_frequency)[0]
        feature_cycle_times = int(re.findall('\d+', y_frequency)[0])
        split_dataset = dataByCycle.getCycleselectDataFrame(data[[feature]], feature_cycle, feature_cycle_times, downsampling_freq)

        # get heatmap x-axis, y-axis value
        x = list(range(1,len(split_dataset[0])+1))
        y = list(range(1, len(split_dataset)+1))

        # get heatmap z(data) value
        z = []
        for split_data in split_dataset:
            z.append(split_data[feature].values.tolist())
        
        fig = go.Figure(data = go.Heatmap(z=z, x=x, y=y, hoverongaps=False))

        fig.update_layout(
            title = feature,
            xaxis = dict(
                tickmode = 'array',
                tickvals = x,
                ticktext = [str(num)+string_time for num in x]
            ),
            yaxis = dict(
                tickmode = 'array',
                tickvals = y,
                ticktext = [str(num)+feature_cycle for num in y]
            ))

        fig.show()
    except ZeroDivisionError:
        print("The duration of the data is less than {}.".format(y_frequency))

def show_one_data_distribution_of_two_feature(data, feature_list):
    """
    
    Args:
        data (dataframe): Input data
        feature_list (string): columns of data as two of column name to be shown
    """
    
    try:
        x_feature = feature_list[0]
        y_feature = feature_list[1]
        
        fig = go.Figure(
            data=go.Scatter(
                y = data[y_feature], # y축 값 sepal_length 값에 따라 배치
                x = data[x_feature],
                mode='markers', # Scatter Plot을 그리기 위해 Markers 
                marker=dict(    # Marker에 대한 세부적은 설정을 지정
                    size=10,    # 점 크기
                    line_width=1, # 마커 라인 두께 설정
                )
            )
        )
        fig.update_layout(
            title='distribution of {} and {}'.format(x_feature, y_feature),
            xaxis_title = "{} Feature".format(x_feature),
            yaxis_title = "{} Feature".format(y_feature)
            )
        fig.show()
    except:
        print("Data is None")