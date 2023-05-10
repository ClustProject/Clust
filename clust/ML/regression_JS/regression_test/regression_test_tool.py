import os
import sys
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")

import missingno as msno
from Clust.clust.preprocessing import processing_interface
from Clust.clust.transformation.general import split_interface, select_interface
from Clust.clust.transformation.purpose import machineLearning
from matplotlib import pyplot as plt
import numpy as np

def nan_plot_refine_data_plot(data, refine_param, target):
    """
    원본 데이터를 입력 받아서 데이터의 Nan 값을 체크하고 이를 입력한 refine_param에 맞춰서 refine 함.
    refine 한 데이터 또한, Nan값을 체크함.
    원본 데이터와 refine 한 데이터의 target (feature) 값의 시간 변화 그래프를 출력

    refine 된 데이터를 반환
    """
    print("======= Original Data Nan =======")
    print(data.isna().sum())

    ## refine
    refinement_data = processing_interface.get_data_result('refinement', data , refine_param)

    print("======= Refine Data Nan =======")
    print(refinement_data.isna().sum())

    ## data plot
    print("======= Original Data Plot =======")
    plt.plot(data[target])
    plt.show()

    print("======= Refine Data Plot =======")
    plt.plot(refinement_data[target])
    plt.show()

    return refinement_data

def split_data_by_day_workingtime(data, split_param, select_param):
    """
    split_param에 맞춰서 데이터 cycle를 자르고 working time 을 자름.
    자른 데이터의 plot 을 출력
    cycle 단위의 데이터가 없을 경우 빈 데이터를 출력

    cycle, working time 에 따라 잘라진 데이터 반환
    """
    split_data_day = split_interface.get_data_result("cycle", data, split_param)
    split_data_working = split_interface.get_data_result("working", split_data_day,  split_param)
    split_data_day_working = select_interface.get_data_result("keyword_data_selection", split_data_working, select_param)

    data_empty_name = []
    for data_name, data in split_data_day_working.items():
        print(data_name)
        if not(data.empty):
            plt.plot(data.in_co2)
            plt.show()
        else:
            data_empty_name.append(data_name)
    print("Data Empty List : " , data_empty_name)

    return split_data_day_working

def get_y_data_by_split_dataset(split_dataset, get_y_data_transformParameter, transform_clean):
    """
    일별 working time으로 나뉜 데이터 셋으로 부터 y 값을 구하는 함수 
    """
    for idx, data in enumerate(split_dataset.values()):
        if not(data.empty):
            LSTMD = machineLearning.LSTMData()
            data_x_arr, data_y_arr = LSTMD.transform_Xy_arr(data, get_y_data_transformParameter, transform_clean)

            if idx == 0:
                data_x = data_x_arr
                data_y = data_y_arr
            else:
                data_x = np.append(data_x, np.array(data_x_arr), axis=0)
                data_y = np.append(data_y, np.array(data_y_arr), axis=0)

            print(data_x.shape, data_y.shape)
            print("*******************")
    return data_x, data_y