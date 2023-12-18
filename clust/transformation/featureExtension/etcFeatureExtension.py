import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

class ETCFeatureExtension():
    
    def __init__(self):
        pass
    
    def add_level_features(self, data, feature_limit):
        for feature_name in feature_limit:
            level = feature_limit[feature_name]['level']
            label = feature_limit[feature_name]['label']
            data[feature_name] = pd.to_numeric(data[feature_name], errors='coerce')
            #feature_notnull = data[~data[feature_name].isnull()][feature_name]
            feature_notnull = data[data[feature_name].notnull()][feature_name]
            print(feature_name)
            data[feature_name+'_level'] = pd.cut(feature_notnull, level, labels = label, right=False).astype(int)
        return data

    def add_vector_features(self, data, vector_feature_list):
        plt.rcParams['figure.figsize'] =(5, 5)
        for speed_feature in vector_feature_list:
            velocity = data[speed_feature]
            direction = data[vector_feature_list[speed_feature][0]] 
            velocity_x = vector_feature_list[speed_feature][1][0]
            velocity_y= vector_feature_list[speed_feature][1][1]
            direction_radian = direction * np.pi /180
            data[velocity_x] = velocity * np.cos(direction_radian)
            data[velocity_y] = velocity * np.sin(direction_radian)
            plt.hist2d(data[velocity_x].dropna(), data[velocity_y].dropna(), bins = (30, 30), vmax= 100)
            #plt.colorbar()
            ax = plt.gca()
            ax.axis('tight')
        return data

    def add_ratio_features(self, data):
        import itertools
        tag ='_ratio'
        data_features = list(data.columns)
        combination = list(itertools.combinations(data_features, 2))  
        for combi in combination:
            new_feature_name = combi[0]+'_and_'+combi[1]+tag
            data[new_feature_name] = (data[combi[0]]+0.01)/(data[combi[1]]+0.01)
          
        data = data.interpolate(method='values')
        return data
