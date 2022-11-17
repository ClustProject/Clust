import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from io import BytesIO
import base64
from tslearn.barycenters import dtw_barycenter_averaging
import math
plt.switch_backend('Agg')

def somTrain(feature_dataset, feature_datasetName):

    import math
    result = {}; figdata=""
    
    from KETIPreDataTransformation.general import basicTransform
    seriesData_SS_DF = basicTransform.scalingSmoothingDF(feature_dataset, ewm_parameter=0.3 )
    seriesData_SS_series = basicTransform.DFSetToSeries(seriesData_SS_DF)
    som_x = som_y = math.ceil(math.sqrt(math.sqrt(len(seriesData_SS_series))))
    som_x = som_y = 3
    from minisom import MiniSom
    from clust.ML.clustering import som_visual
    som = MiniSom(som_x, som_y,len(seriesData_SS_series[0]), sigma=0.3, learning_rate = 0.1)
    som.random_weights_init(seriesData_SS_series)
    som.train(seriesData_SS_series, 50000)
    # Returns the mapping of the winner nodes and inputs
    win_map = som.win_map(seriesData_SS_series)
    center_type = 'dtw_barycenter_averaging'
    figdata1 = som_visual.plot_som_series_center_return_image(som_x, som_y, win_map, center_type)
    figdata2 = som_visual.drawSomClusteringResult(win_map, som_x, som_y, 25, 5)
    result = som_visual.getsomClustNumberDict(som, seriesData_SS_series, feature_datasetName, som_y)
    # result = som_visual.getsomClustNumber(som, seriesData_SS_series, feature_datasetName, som_y) #return DF
    print(win_map.keys())

    return result, figdata1, figdata2

def plot_som_series_center(som_x, som_y, win_map, center_type):
    plt.rcParams.update({'font.size': 25})
    if(len(win_map)==1):
        fig = plt.figure(figsize=(25,25))
        axs = fig.add_subplot(1,1,1)
        cluster = (0,0)
        if cluster in win_map.keys():
            for series in win_map[cluster]:
                axs.plot(series,c="gray",alpha=0.5)
            # nan to zero for plotting
            """
            for m in win_map[cluster]:
                m = np.nan_to_num(m, copy=False)
            """
            if center_type == 'dtw_barycenter_averaging':
                axs.plot(dtw_barycenter_averaging(np.vstack(win_map[cluster])),c="red") 
            else:
                axs.plot(np.average(np.vstack(win_map[cluster]),axis=0),c="red")
            
            axs.set_title(f"Cluster {1}")
    else:
        size_x = math.ceil((math.sqrt(len(win_map))))
        size_y = math.ceil(len(win_map)/size_x)
        fig, axs = plt.subplots(size_y,size_x,figsize=(25,25))
        cnt = 0
        for x in range(som_x):
            for y in range(som_y):
                cluster = (x,y)
                if cluster in win_map.keys():
                    if(size_y==1): pos = cnt
                    else : pos = (int(cnt/size_x),cnt%size_x)
                    cnt = cnt + 1
                    for series in win_map[cluster]:
                        axs[pos].plot(series,c="gray",alpha=0.5)
                    
                    if center_type == 'dtw_barycenter_averaging':
                        axs[pos].plot(dtw_barycenter_averaging(np.vstack(win_map[cluster])),c="red") 
                    else:
                        axs[pos].plot(np.average(np.vstack(win_map[cluster]),axis=0),c="red")
                    cluster_number = x*som_y+y+1
                    axs[pos].set_title(f"Cluster {cluster_number}")

    fig.suptitle('Clusters')
    return plt

def plot_som_series_center_return_image(som_x, som_y, win_map, center_type):
    plt = plot_som_series_center(som_x, som_y, win_map, center_type)
    # send images
    buf = BytesIO()
    plt.savefig(buf, format='png')
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8').replace('\n', '')
    return image_base64


def drawSomClusteringResult(win_map, som_x, som_y, width, height):
    cluster_c = []
    cluster_n = []
    for x in range(som_x):
        for y in range(som_y):
            cluster = (x,y)
            if cluster in win_map.keys():
                cluster_c.append(len(win_map[cluster]))
            else:
                cluster_c.append(0)
            cluster_number = x*som_y+y+1
            cluster_n.append(f"Cluster {cluster_number}")

    plt.figure(figsize=(width,height))
    plt.title("Cluster Distribution for SOM")
    plt.bar(cluster_n,cluster_c)

    return plt

def getsomClustNumber(som, seriesData, seriesDataName, som_y):
    # return dataframe
    cluster_map = []
    for idx in range(len(seriesData)):
        winner_node = som.winner(seriesData[idx])
        cluster_map.append((seriesDataName[idx],f"Cluster {winner_node[0]*som_y+winner_node[1]+1}"))

    result = pd.DataFrame(cluster_map,columns=["Series","Cluster"]).sort_values(by="Cluster").set_index("Series")
    return result


def getsomClustNumberDict(som, seriesData, seriesDataName, som_y):
    # return dataframe
    cluster_map = {}
    for idx in range(len(seriesData)):
        winner_node = som.winner(seriesData[idx])
        cluster_map[seriesDataName[idx]]=str(winner_node[0]*som_y+winner_node[1]+1)

    return cluster_map